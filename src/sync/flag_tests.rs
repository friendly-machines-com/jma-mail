//! Stateful flag-sync tests using real Maildir files and SyncEngine cycles.

//! The server applies keyword patches independently of the client, rejects
//! false keyword values, and exposes its state through fresh Email/get calls.
//! A successful HTTP response alone is not a convergence assertion.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::config::FolderLayout;
use crate::config::{
    AccountConfig, AllowDestructiveFolderSync, Config, ConflictStrategy, StateConfig, SyncConfig,
};
use crate::state::{db, queries};
use crate::sync::engine::SyncEngine;
use rusqlite::Connection;
use serde_json::{Value, json};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const ACCOUNT_ID: &str = "u-test-account";
const SESSION_STATE: &str = "session-1";

/// Mutable mock-server state shared across the dispatcher closures so a
/// single test can stage multiple outcomes (e.g. pre-load mailboxes and
/// emails, then advance state, or rig the next Email/changes to error
/// with cannotCalculateChanges).
#[derive(Default)]
struct ServerState {
    mailboxes: Vec<RemoteMailbox>,
    mailbox_state: String,
    emails: Vec<RemoteEmail>,
    email_state: String,
    blobs: HashMap<String, Vec<u8>>,
    set_requests: Vec<Value>,
    reject_next_update: usize,
    reject_only_id: Option<String>,
    http_failures: usize,
    lost_replies: usize,
    hide_on_get: Option<(std::path::PathBuf, std::path::PathBuf)>,
    /// If set, the next Email/changes call returns this error instead
    /// of a normal response. Consumed on use so a follow-up call gets
    /// the regular code path.
    next_email_changes_error: Option<&'static str>,
    /// Mailbox/set { create } requests recorded in the order they
    /// arrived. Each entry is the (name, parent_id, role) triple
    /// the client sent; the handler appends the corresponding
    /// server-assigned id into `mailboxes` so a follow-up
    /// `Mailbox/get` would see it.
    mailbox_set_creates: Vec<(String, Option<String>, Option<String>)>,
    /// Names that `Mailbox/set { create }` must reject. Each
    /// matching create goes into `notCreated` with
    /// `invalidProperties` and is not appended to
    /// `mailbox_set_creates`. Pins the warn-and-continue path.
    mailbox_set_reject_names: Vec<String>,
    /// Mailbox/set { update } requests recorded in the order
    /// they arrived. Each entry is the (id, new_name,
    /// new_parent_id) triple the client sent; the handler
    /// rewrites the matching `RemoteMailbox` so a follow-up
    /// `Mailbox/get` reflects the rename.
    mailbox_set_updates: Vec<(String, String, Option<String>)>,
    /// Ids that `Mailbox/set { update }` must reject. Each
    /// matching update goes into `notUpdated` with
    /// `invalidProperties` and is not appended to
    /// `mailbox_set_updates` / does not rewrite the mailbox.
    mailbox_set_reject_update_ids: Vec<String>,
    /// Mailbox/set { destroy } ids recorded in the order they
    /// arrived. The handler removes the matching `RemoteMailbox`
    /// so a follow-up `Mailbox/get` no longer reflects it.
    mailbox_set_destroys: Vec<String>,
    /// Ids that `Mailbox/set { destroy }` must reject. Each
    /// matching destroy goes into `notDestroyed` with
    /// `mailboxHasEmail` and is not appended to
    /// `mailbox_set_destroys` / does not remove the mailbox.
    /// Pins the warn-and-continue path.
    mailbox_set_reject_destroy_ids: Vec<String>,
}

#[derive(Default, Clone)]
struct EmailChangesResp {
    created: Vec<String>,
    updated: Vec<String>,
    destroyed: Vec<String>,
}

#[derive(Clone)]
struct RemoteMailbox {
    id: String,
    name: String,
    role: Option<String>,
    parent_id: Option<String>,
}

#[derive(Clone)]
struct RemoteEmail {
    id: String,
    blob_id: String,
    thread_id: String,
    mailbox_ids: Vec<String>,
    keywords: Vec<String>,
    message_id: Option<String>,
}

fn session_doc(server_uri: &str) -> Value {
    json!({
        "capabilities": {
            "urn:ietf:params:jmap:core": {
                "maxSizeUpload": 50_000_000u64,
                "maxConcurrentUpload": 4u64,
                "maxSizeRequest": 10_000_000u64,
                "maxConcurrentRequests": 4u64,
                "maxCallsInRequest": 16u64,
                "maxObjectsInGet": 500u64,
                "maxObjectsInSet": 500u64,
                "collationAlgorithms": ["i;ascii-numeric"],
            },
            "urn:ietf:params:jmap:mail": {
                "maxMailboxesPerEmail": 1000u64,
                "maxMailboxDepth": null,
                "maxSizeMailboxName": 255u64,
                "maxSizeAttachmentsPerEmail": 50_000_000u64,
                "emailQuerySortOptions": [],
                "mayCreateTopLevelMailbox": true,
            }
        },
        "accounts": {
            ACCOUNT_ID: {
                "name": "test@example.com",
                "isPersonal": true,
                "isReadOnly": false,
                "accountCapabilities": {
                    "urn:ietf:params:jmap:mail": {}
                }
            }
        },
        "primaryAccounts": {
            "urn:ietf:params:jmap:mail": ACCOUNT_ID
        },
        "username": "test@example.com",
        "apiUrl": format!("{}/jmap", server_uri),
        "downloadUrl": format!(
            "{}/download/{{accountId}}/{{blobId}}/{{name}}?accept={{type}}",
            server_uri
        ),
        "uploadUrl": format!("{}/upload/{{accountId}}", server_uri),
        "eventSourceUrl": format!("{}/eventsource", server_uri),
        "state": "abc123",
    })
}

async fn mount_session(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/.well-known/jmap"))
        .respond_with(ResponseTemplate::new(200).set_body_json(session_doc(&server.uri())))
        .mount(server)
        .await;
}

/// Mount the `POST /jmap` dispatcher. Inspects the methodCalls in the
/// request body, dispatches each to the matching handler, and emits a
/// JMAP-shaped methodResponses array.
async fn mount_jmap(server: &MockServer, state: Arc<Mutex<ServerState>>) {
    let state_for_closure = state.clone();
    Mock::given(method("POST"))
        .and(path("/jmap"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).expect("valid JSON body");
            let calls = body["methodCalls"]
                .as_array()
                .expect("methodCalls array")
                .clone();
            if calls.iter().any(|call| call[0] == "Email/set") {
                let mut state = state_for_closure.lock().unwrap();
                state.set_requests.push(body.clone());
                if state.http_failures > 0 {
                    state.http_failures -= 1;
                    return ResponseTemplate::new(503);
                }
            }
            let mut responses = Vec::new();
            for call in calls {
                let arr = call.as_array().expect("methodCall is a triple");
                let method_name = arr[0].as_str().expect("method name").to_string();
                let args = arr[1].clone();
                let call_id = arr[2].as_str().expect("call id").to_string();
                let resp = handle_method(
                    &method_name,
                    &args,
                    &call_id,
                    &mut state_for_closure.lock().unwrap(),
                );
                responses.push(resp);
            }
            if responses.iter().any(|response| response[0] == "Email/set") {
                let mut state = state_for_closure.lock().unwrap();
                if state.lost_replies > 0 {
                    state.lost_replies -= 1;
                    return ResponseTemplate::new(503);
                }
            }
            ResponseTemplate::new(200).set_body_json(json!({
                "methodResponses": responses,
                "sessionState": SESSION_STATE,
            }))
        })
        .mount(server)
        .await;
}

/// Mount `GET /download/...` so `download_blob` finds the bytes that
/// were preloaded into `state.blobs`.
async fn mount_blob_downloads(server: &MockServer, state: Arc<Mutex<ServerState>>) {
    let state_for_closure = state.clone();
    Mock::given(method("GET"))
        .and(path_regex(r"^/download/u-test-account/"))
        .respond_with(move |req: &Request| {
            // URL shape: /download/{accountId}/{blobId}/{name}?accept=...
            let url = req.url.path();
            let segments: Vec<&str> = url.trim_start_matches('/').split('/').collect();
            // ["download", accountId, blobId, name]
            let blob_id = segments.get(2).copied().unwrap_or_default();
            let st = state_for_closure.lock().unwrap();
            match st.blobs.get(blob_id) {
                Some(bytes) => ResponseTemplate::new(200)
                    .set_body_bytes(bytes.clone())
                    .insert_header("content-type", "application/octet-stream"),
                None => ResponseTemplate::new(404).set_body_string("not found"),
            }
        })
        .mount(server)
        .await;
}

fn handle_method(name: &str, args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    match name {
        "Mailbox/get" => mailbox_get(args, call_id, state),
        "Mailbox/set" => mailbox_set(args, call_id, state),
        "Mailbox/changes" => mailbox_changes(call_id),
        "Email/query" => email_query(args, call_id, state),
        "Email/get" => email_get(args, call_id, state),
        "Email/changes" => email_changes(args, call_id, state),
        "Email/set" => email_set(args, call_id, state),
        other => json!([
            "error",
            { "type": "unknownMethod", "description": format!("unhandled: {}", other) },
            call_id
        ]),
    }
}

fn mailbox_get(_args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    let list: Vec<Value> = state
        .mailboxes
        .iter()
        .map(|mb| {
            json!({
                "id": mb.id,
                "name": mb.name,
                "parentId": mb.parent_id,
                "role": mb.role,
                "sortOrder": 0,
                "totalEmails": 0,
                "unreadEmails": 0,
                "totalThreads": 0,
                "unreadThreads": 0,
                "myRights": {
                    "mayReadItems": true, "mayAddItems": true,
                    "mayRemoveItems": true, "maySetSeen": true,
                    "maySetKeywords": true, "mayCreateChild": true,
                    "mayRename": true, "mayDelete": true, "maySubmit": true
                },
                "isSubscribed": true
            })
        })
        .collect();
    json!([
        "Mailbox/get",
        {
            "accountId": ACCOUNT_ID,
            "state": state.mailbox_state,
            "list": list,
            "notFound": []
        },
        call_id
    ])
}

/// Minimal `Mailbox/set { create | update | destroy }`
/// dispatcher. Records each create into `state.mailbox_set_creates`,
/// each update into `state.mailbox_set_updates`, and each destroy
/// into `state.mailbox_set_destroys`, rewriting / appending /
/// removing the matching `RemoteMailbox` so a follow-up
/// `Mailbox/get` reflects the change.
fn mailbox_set(args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    let mut created_resp = serde_json::Map::new();
    let mut not_created_resp = serde_json::Map::new();
    if let Some(creates) = args["create"].as_object() {
        for (creation_id, props) in creates {
            let name = props["name"].as_str().unwrap_or_default().to_string();
            let parent = props["parentId"].as_str().map(|s| s.to_string());
            let role = match &props["role"] {
                Value::String(s) => Some(s.clone()),
                _ => None,
            };
            if state.mailbox_set_reject_names.iter().any(|n| n == &name) {
                not_created_resp.insert(
                    creation_id.clone(),
                    json!({
                        "type": "invalidProperties",
                        "description": format!("rigged rejection for {:?}", name),
                    }),
                );
                continue;
            }
            let new_id = format!("MB-NEW-{}", state.mailboxes.len() + 1);
            state.mailboxes.push(RemoteMailbox {
                id: new_id.clone(),
                name: name.clone(),
                role: role.clone(),
                parent_id: parent.clone(),
            });
            state.mailbox_set_creates.push((name, parent, role));
            created_resp.insert(creation_id.clone(), json!({ "id": new_id }));
        }
    }
    let mut updated_resp = serde_json::Map::new();
    let mut not_updated_resp = serde_json::Map::new();
    if let Some(updates) = args["update"].as_object() {
        for (id, props) in updates {
            if state.mailbox_set_reject_update_ids.iter().any(|x| x == id) {
                not_updated_resp.insert(
                    id.clone(),
                    json!({
                        "type": "invalidProperties",
                        "description": format!("rigged rejection for {}", id),
                    }),
                );
                continue;
            }
            let new_name = props["name"].as_str().unwrap_or_default().to_string();
            let new_parent = props["parentId"].as_str().map(|s| s.to_string());
            if let Some(mb) = state.mailboxes.iter_mut().find(|m| m.id == *id) {
                mb.name = new_name.clone();
                mb.parent_id = new_parent.clone();
            }
            state
                .mailbox_set_updates
                .push((id.clone(), new_name, new_parent));
            updated_resp.insert(id.clone(), Value::Null);
        }
    }
    let mut destroyed_resp: Vec<Value> = Vec::new();
    let mut not_destroyed_resp = serde_json::Map::new();
    if let Some(destroys) = args["destroy"].as_array() {
        for id_val in destroys {
            let Some(id) = id_val.as_str() else {
                continue;
            };
            if state.mailbox_set_reject_destroy_ids.iter().any(|x| x == id) {
                not_destroyed_resp.insert(
                    id.to_string(),
                    json!({
                        "type": "mailboxHasEmail",
                        "description": format!("rigged rejection for {}", id),
                    }),
                );
                continue;
            }
            state.mailboxes.retain(|m| m.id != id);
            state.mailbox_set_destroys.push(id.to_string());
            destroyed_resp.push(Value::String(id.to_string()));
        }
    }
    let old_state = state.mailbox_state.clone();
    state.mailbox_state = format!("{}-set", old_state);
    json!([
        "Mailbox/set",
        {
            "accountId": ACCOUNT_ID,
            "oldState": old_state,
            "newState": state.mailbox_state,
            "created": Value::Object(created_resp),
            "updated": Value::Object(updated_resp),
            "destroyed": Value::Array(destroyed_resp),
            "notCreated": Value::Object(not_created_resp),
            "notUpdated": Value::Object(not_updated_resp),
            "notDestroyed": Value::Object(not_destroyed_resp),
        },
        call_id
    ])
}

fn email_query(args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    // Two filter shapes are supported, matching the two callers
    // in the engine + janitor today:
    //   - `{ "inMailbox": "<id>" }`: SyncEngine's per-mailbox query
    //   - `{ "operator": "OR", "conditions": [{ "header": ["Message-ID", "<id>"] }, ...] }`:
    //     rebindfolders' Message-ID probe
    // Anything else returns an empty result; tests using new
    // filter shapes need to extend this dispatcher.
    let filter = &args["filter"];
    let ids: Vec<String> = if let Some(mailbox_id) = filter["inMailbox"].as_str() {
        state
            .emails
            .iter()
            .filter(|e| e.mailbox_ids.iter().any(|m| m == mailbox_id))
            .map(|e| e.id.clone())
            .collect()
    } else if filter["operator"].as_str() == Some("OR") {
        // OR-of-headers: collect every `header[0]: header[1]`
        // pair, match emails whose `messageId` contains the value
        // (with our angle brackets stripped to compare against the
        // server's stored bracket-stripped form).
        let mut wanted: Vec<String> = Vec::new();
        if let Some(conditions) = filter["conditions"].as_array() {
            for cond in conditions {
                let Some(header) = cond["header"].as_array() else {
                    continue;
                };
                if header.len() != 2 {
                    continue;
                }
                let Some(name) = header[0].as_str() else {
                    continue;
                };
                let Some(value) = header[1].as_str() else {
                    continue;
                };
                if !name.eq_ignore_ascii_case("Message-ID") {
                    continue;
                }
                let stripped = value
                    .strip_prefix('<')
                    .and_then(|s| s.strip_suffix('>'))
                    .unwrap_or(value);
                wanted.push(stripped.to_string());
            }
        }
        state
            .emails
            .iter()
            .filter(|e| {
                e.message_id
                    .as_ref()
                    .is_some_and(|m| wanted.iter().any(|w| w == m))
            })
            .map(|e| e.id.clone())
            .collect()
    } else {
        Vec::new()
    };
    let total = ids.len();
    let position = args["position"].as_u64().unwrap_or(0) as usize;
    let limit = args["limit"].as_u64().unwrap_or(total as u64) as usize;
    let ids: Vec<_> = ids.into_iter().skip(position).take(limit).collect();
    json!([
        "Email/query",
        {
            "accountId": ACCOUNT_ID,
            "queryState": "q-1",
            "canCalculateChanges": false,
            "position": position,
            "ids": ids,
            "total": total,
            "limit": limit
        },
        call_id
    ])
}

fn email_get(args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    if let Some((source, held)) = state.hide_on_get.take() {
        std::fs::rename(source, held).unwrap();
    }
    let empty = vec![];
    let id_list = args["ids"].as_array().unwrap_or(&empty);
    let want_all_ids = args.get("ids").is_none();
    let list: Vec<Value> = state
        .emails
        .iter()
        .filter(|e| want_all_ids || id_list.iter().any(|v| v.as_str() == Some(e.id.as_str())))
        .map(|e| {
            let mut mailbox_map = serde_json::Map::new();
            for m in &e.mailbox_ids {
                mailbox_map.insert(m.clone(), Value::Bool(true));
            }
            let mut keyword_map = serde_json::Map::new();
            for k in &e.keywords {
                keyword_map.insert(k.clone(), Value::Bool(true));
            }
            json!({
                "id": e.id,
                "blobId": e.blob_id,
                "threadId": e.thread_id,
                "mailboxIds": Value::Object(mailbox_map),
                "keywords": Value::Object(keyword_map),
                "messageId": e.message_id.as_ref().map(|m| vec![m.clone()]),
            })
        })
        .collect();
    let not_found: Vec<_> = id_list
        .iter()
        .filter(|id| {
            !state
                .emails
                .iter()
                .any(|email| Some(email.id.as_str()) == id.as_str())
        })
        .cloned()
        .collect();
    json!([
        "Email/get",
        {
            "accountId": ACCOUNT_ID,
            "state": state.email_state,
            "list": list,
            "notFound": not_found
        },
        call_id
    ])
}

fn email_changes(args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    if let Some(err) = state.next_email_changes_error.take() {
        return json!([
            "error",
            { "type": err, "description": format!("rigged {}", err) },
            call_id
        ]);
    }
    let since = args["sinceState"].as_str().unwrap_or_default();
    let changes = if since == state.email_state {
        EmailChangesResp::default()
    } else {
        EmailChangesResp {
            updated: state.emails.iter().map(|email| email.id.clone()).collect(),
            ..Default::default()
        }
    };
    json!([
        "Email/changes",
        {
            "accountId": ACCOUNT_ID,
            "oldState": since,
            "newState": state.email_state,
            "hasMoreChanges": false,
            "created": changes.created,
            "updated": changes.updated,
            "destroyed": changes.destroyed,
        },
        call_id
    ])
}

/// jma's cache-reuse path calls `Mailbox/changes` once a `Mailbox`
/// cursor exists. Return `cannotCalculateChanges` so the engine falls
/// back to the full `Mailbox/get` this mock serves -- every test keeps
/// exercising the same server-structure path it did before the cursor
/// existed, rather than needing the mock to track mailbox-state deltas.
fn mailbox_changes(call_id: &str) -> Value {
    json!([
        "error",
        { "type": "cannotCalculateChanges", "description": "mock: full refetch" },
        call_id
    ])
}

/// Open a fresh on-disk state DB inside the given temp dir. Integration
/// tests can't use `db::open_in_memory` (gated on `cfg(test)` for unit
/// tests), so route through the public `open_or_recreate` against a
/// path the temp dir cleans up automatically.
fn fresh_db(temp: &tempfile::TempDir) -> Connection {
    let db_path = temp.path().join("state.db");
    db::open_or_recreate(&db_path).expect("open state DB")
}

fn test_config(
    server: &MockServer,
    maildir_root: &std::path::Path,
    mailboxes: Vec<String>,
) -> Config {
    Config {
        account: AccountConfig {
            email: "test@example.com".to_string(),
            token: Some("test-token".to_string()),
            // Force `session::connect` down the explicit-override path
            // so it `Client::connect`s the wiremock server instead of
            // doing real DNS/well-known autodiscovery.
            session_url: Some(server.uri()),
        },
        sync: SyncConfig {
            maildir_path: maildir_root.to_string_lossy().into_owned(),
            mailboxes,
            conflict_strategy: ConflictStrategy::ServerWins,
            case_insensitive_match: false,
            download_concurrency: 2,
            upload_concurrency: 2,
            retry_max_attempts: 1,
            retry_initial_backoff_ms: 1,
            retry_max_backoff_ms: 1,
            allow_destructive_folder_sync: AllowDestructiveFolderSync::None,
            folder_layout: FolderLayout::Fs,
            hierarchy_separator: '/',
        },
        state: StateConfig {
            db_path: Some(":memory:".to_string()),
        },
        ..Default::default()
    }
}

use crate::maildir_ops::flags::{flags_to_keywords, keywords_to_flags};
use std::collections::BTreeSet;
use std::path::PathBuf;

async fn assert_converged(fixture: &FlagFixture, flags: &str) {
    let snapshot = fixture.snapshot().await;
    assert_eq!(snapshot["file_flags"], flags, "{snapshot}");
    assert_eq!(snapshot["server_flags"], flags, "{snapshot}");
    assert_eq!(snapshot["cached_flags"], flags, "{snapshot}");
    assert_eq!(snapshot["local_baseline"], flags, "{snapshot}");
    assert_eq!(
        snapshot["cached_keywords"], snapshot["server_keywords"],
        "{snapshot}"
    );
}

#[test]
fn flag_sync_server_enforces_keyword_patch_semantics() {
    let email = RemoteEmail {
        id: "E1".into(),
        blob_id: "B1".into(),
        thread_id: "T1".into(),
        mailbox_ids: vec!["MB-INBOX".into()],
        keywords: vec!["$seen".into(), "$flagged".into(), "keep-me".into()],
        message_id: Some("audit@example.com".into()),
    };
    let mut state = ServerState {
        email_state: "e-1".into(),
        emails: vec![email.clone()],
        ..Default::default()
    };
    let response = email_set(
        &json!({"update":{"E1":{"keywords/$flagged":null}}}),
        "c1",
        &mut state,
    );
    assert!(response[1]["updated"].get("E1").is_some());
    assert!(!state.emails[0].keywords.contains(&"$flagged".into()));
    assert!(state.emails[0].keywords.contains(&"keep-me".into()));
    state.emails = vec![email.clone()];
    let before = state.email_state.clone();
    let response = email_set(
        &json!({"update":{"E1":{"keywords/$flagged":false,"mailboxIds":{"MB-TRASH":true}}}}),
        "c2",
        &mut state,
    );
    assert!(response[1]["notUpdated"].get("E1").is_some());
    assert_eq!(state.emails[0].keywords, email.keywords);
    assert_eq!(state.emails[0].mailbox_ids, email.mailbox_ids);
    assert_eq!(state.email_state, before);
    email_set(
        &json!({"update":{"E1":{"keywords/$seen":true}}}),
        "c3",
        &mut state,
    );
    assert_eq!(state.emails[0].keywords, email.keywords);
}

#[tokio::test]
async fn flag_sync_local_addition_converges_then_stays_idle() {
    let fixture = FlagFixture::new("", true, ConflictStrategy::ServerWins).await;
    fixture.local_change("S", false, false);
    let first = fixture.sync().await.unwrap();
    assert_eq!(first.flag_updates, 1);
    assert_converged(&fixture, "S").await;
    let requests = fixture.state.lock().unwrap().set_requests.len();
    let second = fixture.sync().await.unwrap();
    assert!(second.already_in_sync);
    assert_eq!(second.flag_updates, 0);
    assert_eq!(fixture.state.lock().unwrap().set_requests.len(), requests);
    assert_converged(&fixture, "S").await;
}

#[tokio::test]
async fn flag_sync_remote_addition_and_removal_preserve_custom_keywords() {
    let fixture = FlagFixture::new("S", true, ConflictStrategy::ServerWins).await;
    for flags in ["FS", "S"] {
        fixture.server_change(flags, true);
        assert_eq!(fixture.sync().await.unwrap().flag_updates, 1);
        assert_converged(&fixture, flags).await;
        assert!(
            fixture.snapshot().await["server_keywords"]
                .get("keep-me")
                .is_some()
        );
        assert!(fixture.sync().await.unwrap().already_in_sync);
    }
}

#[tokio::test]
async fn flag_sync_rejected_ordinary_push_retains_retryable_baseline() {
    let fixture = FlagFixture::new("", false, ConflictStrategy::ServerWins).await;
    fixture.local_change("S", false, false);
    fixture.state.lock().unwrap().reject_next_update = 1;
    assert_eq!(fixture.sync().await.unwrap().failed_remote_actions, 1);
    let failed = fixture.snapshot().await;
    assert_eq!(failed["file_flags"], "S");
    assert_eq!(failed["server_flags"], "");
    assert_eq!(failed["local_baseline"], "");
    assert_eq!(fixture.sync().await.unwrap().flag_updates, 1);
    assert_converged(&fixture, "S").await;
    assert!(fixture.sync().await.unwrap().already_in_sync);
}

#[tokio::test]
async fn flag_sync_rebuild_matching_views_reuses_existing_file() {
    let mut fixture = FlagFixture::new("FS", true, ConflictStrategy::ServerWins).await;
    let original = fixture.file();
    fixture.reset_db();
    let result = fixture.sync().await.unwrap();
    assert_eq!(result.downloaded, 0);
    assert_eq!(result.uploaded, 0);
    assert_eq!(fixture.file(), original);
    assert_converged(&fixture, "FS").await;
    assert!(fixture.sync().await.unwrap().already_in_sync);
}

fn advance_email_state(state: &mut ServerState) {
    let n: usize = state
        .email_state
        .strip_prefix("e-")
        .unwrap()
        .parse()
        .unwrap();
    state.email_state = format!("e-{}", n + 1);
}

// Independent RFC-style state transition: validate the entire object, then
// commit it atomically. Omission is not deletion; false is not a keyword value.
fn email_set(args: &Value, call_id: &str, state: &mut ServerState) -> Value {
    let old = state.email_state.clone();
    let mut updated = serde_json::Map::new();
    let mut not_updated = serde_json::Map::new();
    if let Some(updates) = args["update"].as_object() {
        for (id, patch) in updates {
            if state.reject_next_update > 0
                && state.reject_only_id.as_ref().is_none_or(|only| only == id)
            {
                state.reject_next_update -= 1;
                not_updated.insert(id.clone(), json!({"type":"forbidden"}));
                continue;
            }
            let Some(index) = state.emails.iter().position(|email| &email.id == id) else {
                not_updated.insert(id.clone(), json!({"type":"notFound"}));
                continue;
            };
            let mut candidate = state.emails[index].clone();
            let mut invalid = None;
            for (path, value) in patch.as_object().unwrap() {
                if let Some(keyword) = path.strip_prefix("keywords/") {
                    let keyword = keyword.replace("~1", "/").replace("~0", "~");
                    if value.is_null() {
                        candidate.keywords.retain(|key| key != &keyword);
                    } else if value == &json!(true) {
                        if !candidate.keywords.contains(&keyword) {
                            candidate.keywords.push(keyword);
                        }
                    } else {
                        invalid = Some(format!("{path} must be true or null, got {value}"));
                    }
                } else if path == "mailboxIds" {
                    match value.as_object() {
                        Some(ids) if !ids.is_empty() && ids.values().all(|v| v == &json!(true)) => {
                            candidate.mailbox_ids = ids.keys().cloned().collect();
                        }
                        _ => invalid = Some("invalid mailboxIds replacement".into()),
                    }
                } else {
                    invalid = Some(format!("unexpected patch path {path}"));
                }
            }
            if let Some(description) = invalid {
                not_updated.insert(
                    id.clone(),
                    json!({"type":"invalidProperties", "description":description}),
                );
                continue;
            }
            let before_keywords: BTreeSet<_> = state.emails[index].keywords.iter().collect();
            let after_keywords: BTreeSet<_> = candidate.keywords.iter().collect();
            let changed = before_keywords != after_keywords
                || state.emails[index].mailbox_ids != candidate.mailbox_ids;
            state.emails[index] = candidate;
            if changed {
                advance_email_state(state);
            }
            updated.insert(id.clone(), Value::Null);
        }
    }
    let mut destroyed = Vec::new();
    let mut not_destroyed = serde_json::Map::new();
    if let Some(ids) = args["destroy"].as_array() {
        for id in ids {
            if let Some(index) = state
                .emails
                .iter()
                .position(|email| Some(email.id.as_str()) == id.as_str())
            {
                state.emails.remove(index);
                destroyed.push(id.clone());
                advance_email_state(state);
            } else {
                not_destroyed.insert(id.as_str().unwrap().to_string(), json!({"type":"notFound"}));
            }
        }
    }
    json!(["Email/set", {"accountId":ACCOUNT_ID, "oldState":old, "newState":state.email_state,
        "updated":updated, "notUpdated":not_updated, "destroyed":destroyed, "notDestroyed":not_destroyed}, call_id])
}

struct FlagFixture {
    _server: MockServer,
    state: Arc<Mutex<ServerState>>,
    temp: tempfile::TempDir,
    conn: Connection,
    config: Config,
    verifier: jmap_client::client::Client,
}

impl FlagFixture {
    async fn new(flags: &str, custom: bool, strategy: ConflictStrategy) -> Self {
        let server = MockServer::start().await;
        mount_session(&server).await;
        let mut keywords: Vec<_> = flags_to_keywords(flags).into_keys().collect();
        if custom {
            keywords.push("keep-me".into());
        }
        let state = Arc::new(Mutex::new(ServerState {
            mailbox_state: "mb-1".into(),
            email_state: "e-1".into(),
            mailboxes: vec![
                RemoteMailbox {
                    id: "MB-INBOX".into(),
                    name: "Inbox".into(),
                    role: Some("inbox".into()),
                    parent_id: None,
                },
                RemoteMailbox {
                    id: "MB-TRASH".into(),
                    name: "Deleted Items".into(),
                    role: Some("trash".into()),
                    parent_id: None,
                },
            ],
            emails: vec![RemoteEmail {
                id: "E1".into(),
                blob_id: "B1".into(),
                thread_id: "T1".into(),
                mailbox_ids: vec!["MB-INBOX".into()],
                keywords,
                message_id: Some("audit@example.com".into()),
            }],
            blobs: HashMap::from([(
                "B1".into(),
                b"Message-ID: <audit@example.com>\r\nSubject: flags\r\n\r\nbody\r\n".to_vec(),
            )]),
            ..Default::default()
        }));
        mount_jmap(&server, state.clone()).await;
        mount_blob_downloads(&server, state.clone()).await;
        let verifier = jmap_client::client::Client::new()
            .credentials("test-token")
            .connect(&server.uri())
            .await
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let conn = fresh_db(&temp);
        let mut config = test_config(&server, temp.path(), vec![]);
        config.sync.conflict_strategy = strategy;
        let fixture = Self {
            _server: server,
            state,
            temp,
            conn,
            config,
            verifier,
        };
        let initial = fixture.sync().await.unwrap();
        assert_eq!(
            initial.downloaded, 1,
            "fixture must perform a real initial pull"
        );
        fixture
    }
    async fn sync(&self) -> anyhow::Result<crate::sync::engine::SyncOutcome> {
        SyncEngine::sync(&self.conn, &self.config, false).await
    }
    fn file(&self) -> PathBuf {
        let mut files = vec![];
        for folder in ["INBOX", "Deleted Items"] {
            for sub in ["cur", "new"] {
                let directory = self.temp.path().join(folder).join(sub);
                if let Ok(entries) = std::fs::read_dir(directory) {
                    files.extend(
                        entries
                            .map(|entry| entry.unwrap().path())
                            .filter(|path| path.is_file()),
                    );
                }
            }
        }
        assert_eq!(
            files.len(),
            1,
            "fixture must retain exactly one message: {files:?}"
        );
        files.remove(0)
    }
    fn local_change(&self, flags: &str, moved: bool, keep_new: bool) {
        let old = self.file();
        let id = old
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split(":2,")
            .next()
            .unwrap();
        let folder = if moved { "Deleted Items" } else { "INBOX" };
        let sub = if keep_new { "new" } else { "cur" };
        let target = self
            .temp
            .path()
            .join(folder)
            .join(sub)
            .join(format!("{id}:2,{flags}"));
        std::fs::rename(old, target).unwrap();
    }
    fn server_change(&self, flags: &str, custom: bool) {
        let mut state = self.state.lock().unwrap();
        state.emails[0].keywords = flags_to_keywords(flags).into_keys().collect();
        if custom {
            state.emails[0].keywords.push("keep-me".into());
        }
        advance_email_state(&mut state);
    }
    fn reset_db(&mut self) {
        self.conn = db::open_or_recreate(&self.temp.path().join("fresh.db")).unwrap();
    }
    async fn snapshot(&self) -> Value {
        // Fresh Email/get from a separate client, not the client's cached DB.
        let remote = crate::jmap::email::get_by_ids(&self.verifier, &["E1".into()])
            .await
            .unwrap()
            .remove(0);
        let path = self.file();
        let suffix = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .split_once(":2,")
            .map_or("", |(_, flags)| flags);
        let folder = path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        let subdir = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        let record = queries::get_message_by_jmap_id(&self.conn, &"E1".into())
            .unwrap()
            .unwrap();
        let cached: Value = serde_json::from_str(&record.jmap_keywords).unwrap();
        let local_state = queries::get_local_state_for_folder(&self.conn, folder).unwrap();
        let baseline = local_state
            .get(record.maildir_id.as_ref().unwrap())
            .map(|(_, flags)| flags.clone());
        let state = self.state.lock().unwrap();
        json!({"file_flags":suffix,"file_folder":folder,"subdir":subdir,
            "server_flags":keywords_to_flags(&remote.keywords),"server_keywords":remote.keywords,
            "cached_flags":record.flags,"cached_keywords":cached,"local_baseline":baseline,
            "cursor":queries::get_jmap_state(&self.conn,ACCOUNT_ID,"Email").unwrap(),
            "server_state":state.email_state,"set_requests":state.set_requests.len()})
    }
}
