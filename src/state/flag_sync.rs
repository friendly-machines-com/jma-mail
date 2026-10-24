//! Acknowledged flag baselines and receipts for ambiguous I/O outcomes.
//!
//! These are synchronization history, not mail data or a general job queue.
//! A DB rebuild discards them and invokes bootstrap policy; an ordinary
//! restart retains them so observation is not mistaken for acknowledgement.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ids::{JmapEmailId, MaildirId};
use crate::maildir_ops::flags::{flags_to_keywords, keywords_to_flags};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlagAttempt {
    pub source_flags: String,
    pub target_flags: String,
    pub source_maildir_id: Option<MaildirId>,
    pub source_folder: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlagSyncState {
    /// None means initial reconciliation has not completed.
    pub baseline: Option<String>,
    pub review: bool,
    pub attempt: Option<FlagAttempt>,
    pub blocked: bool,
    pub error: Option<String>,
    /// In-memory only: an old row has not yet been verified by Email/get.
    pub legacy: bool,
}

pub fn mapped_flags(flags: &str) -> String {
    keywords_to_flags(&flags_to_keywords(flags))
}

type StoredRow = (
    String,
    Option<String>,
    bool,
    Option<String>,
    bool,
    Option<String>,
);
fn decode(row: StoredRow) -> Result<(JmapEmailId, FlagSyncState)> {
    let (id, baseline, review, attempt, blocked, error) = row;
    let attempt = attempt
        .map(|json| serde_json::from_str(&json))
        .transpose()
        .context("Invalid stored flag-attempt receipt; refusing to discard it")?;
    Ok((
        id.into(),
        FlagSyncState {
            baseline,
            review,
            attempt,
            blocked,
            error,
            legacy: false,
        },
    ))
}

pub fn load(conn: &Connection) -> Result<HashMap<JmapEmailId, FlagSyncState>> {
    let mut stmt = conn.prepare(
        "SELECT jmap_email_id, baseline, review, attempt, blocked, error FROM message_flag_sync",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
            row.get(5)?,
        ))
    })?;
    rows.map(|row| decode(row?)).collect()
}

pub fn get(conn: &Connection, id: &JmapEmailId) -> Result<Option<FlagSyncState>> {
    let row: Option<StoredRow> = conn.query_row(
        "SELECT jmap_email_id, baseline, review, attempt, blocked, error FROM message_flag_sync WHERE jmap_email_id=?1",
        [id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
    ).optional()?;
    row.map(decode)
        .transpose()
        .map(|row| row.map(|(_, state)| state))
}

pub fn store(conn: &Connection, id: &JmapEmailId, state: &FlagSyncState) -> Result<()> {
    let attempt = state
        .attempt
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;
    conn.execute("INSERT INTO message_flag_sync (jmap_email_id,baseline,review,attempt,blocked,error)
        VALUES (?1,?2,?3,?4,?5,?6) ON CONFLICT(jmap_email_id) DO UPDATE SET
        baseline=excluded.baseline,review=excluded.review,attempt=excluded.attempt,blocked=excluded.blocked,error=excluded.error",
        params![id, state.baseline, state.review, attempt, state.blocked, state.error])?;
    Ok(())
}

pub fn acknowledge(conn: &Connection, id: &JmapEmailId, flags: &str) -> Result<()> {
    store(
        conn,
        id,
        &FlagSyncState {
            baseline: Some(mapped_flags(flags)),
            ..Default::default()
        },
    )
}

pub fn prepare_attempt(conn: &Connection, id: &JmapEmailId, attempt: FlagAttempt) -> Result<()> {
    // A missing row has no acknowledged generation. In particular a cold
    // adoption's observed server flags are not a baseline hint.
    let mut state = get(conn, id)?.unwrap_or_default();
    state.attempt = Some(attempt);
    state.blocked = false;
    state.error = None;
    store(conn, id, &state)
}

pub fn failure(
    conn: &Connection,
    id: &JmapEmailId,
    blocked: bool,
    rejected: bool,
    error: &str,
) -> Result<()> {
    let mut state = get(conn, id)?.unwrap_or_default();
    // A confirmed rejection cannot later serve as evidence of our own write.
    // Ambiguous transport/missing-ack outcomes retain their receipt.
    if rejected {
        state.attempt = None;
    }
    state.blocked = blocked;
    state.error = Some(error.into());
    store(conn, id, &state)
}

pub fn pending_ids(conn: &Connection, retryable_only: bool) -> Result<Vec<JmapEmailId>> {
    let mut stmt = conn.prepare("SELECT f.jmap_email_id,f.baseline,f.review,f.attempt,f.blocked,m.flags,l.flags
        FROM message_flag_sync f JOIN message_map m ON m.jmap_email_id=f.jmap_email_id
        LEFT JOIN local_state l ON l.maildir_id=m.maildir_id
        WHERE m.maildir_id IS NOT NULL AND (f.baseline IS NULL OR f.attempt IS NOT NULL OR f.review!=0
            OR l.flags IS NULL OR f.baseline!=m.flags OR f.baseline!=l.flags)
        AND (?1=0 OR (f.review=0 AND f.blocked=0))")?;
    let rows = stmt.query_map([retryable_only], |row| {
        Ok((
            row.get::<_, JmapEmailId>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, bool>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, bool>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, Option<String>>(6)?,
        ))
    })?;
    let mut pending = Vec::new();
    for row in rows {
        let (id, baseline, review, attempt, _blocked, remote, local) = row?;
        let unresolved = baseline.as_ref().is_none_or(|baseline| {
            baseline != &mapped_flags(&remote)
                || local
                    .as_ref()
                    .is_none_or(|flags| mapped_flags(flags) != *baseline)
        });
        if review || attempt.is_some() || unresolved {
            pending.push(id);
        }
    }
    Ok(pending)
}

pub fn has_unverified(conn: &Connection) -> Result<bool> {
    conn.query_row("SELECT EXISTS(SELECT 1 FROM message_map m LEFT JOIN message_flag_sync f
        ON f.jmap_email_id=m.jmap_email_id WHERE m.maildir_id IS NOT NULL AND f.jmap_email_id IS NULL)", [], |row| row.get(0)).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{db, queries};

    fn seed(conn: &Connection) {
        queries::upsert_message(
            conn,
            &queries::MessageRecord {
                jmap_email_id: "E1".into(),
                jmap_blob_id: None,
                jmap_thread_id: None,
                jmap_mailbox_id: "MB1".into(),
                maildir_id: Some("local".into()),
                message_id: "a@x".into(),
                flags: "S".into(),
                jmap_keywords: r#"{"$seen":true}"#.into(),
            },
        )
        .unwrap();
        queries::upsert_local_state(conn, &"local".into(), "INBOX", "SX", None).unwrap();
    }

    fn attempt() -> FlagAttempt {
        FlagAttempt {
            source_flags: "SX".into(),
            target_flags: "S".into(),
            source_maildir_id: Some("local".into()),
            source_folder: Some("INBOX".into()),
        }
    }

    #[test]
    fn observations_and_receipts_are_not_acknowledgements() {
        let conn = db::open_in_memory().unwrap();
        seed(&conn);
        assert!(has_unverified(&conn).unwrap());
        prepare_attempt(&conn, &"E1".into(), attempt()).unwrap();
        let state = get(&conn, &"E1".into()).unwrap().unwrap();
        assert!(state.baseline.is_none());
        assert_eq!(state.attempt, Some(attempt()));
        assert_eq!(
            pending_ids(&conn, true).unwrap(),
            vec![JmapEmailId::from("E1")]
        );
        acknowledge(&conn, &"E1".into(), "S").unwrap();
        assert!(
            pending_ids(&conn, false).unwrap().is_empty(),
            "unowned X is not flag work"
        );
    }

    #[test]
    fn confirmed_rejection_cannot_be_mistaken_for_an_ambiguous_write() {
        let conn = db::open_in_memory().unwrap();
        seed(&conn);
        acknowledge(&conn, &"E1".into(), "F").unwrap();
        prepare_attempt(&conn, &"E1".into(), attempt()).unwrap();
        failure(&conn, &"E1".into(), false, false, "missing ack").unwrap();
        assert!(get(&conn, &"E1".into()).unwrap().unwrap().attempt.is_some());
        failure(&conn, &"E1".into(), true, true, "forbidden").unwrap();
        let state = get(&conn, &"E1".into()).unwrap().unwrap();
        assert_eq!(state.baseline.as_deref(), Some("F"));
        assert!(state.attempt.is_none() && state.blocked);
        assert_eq!(pending_ids(&conn, false).unwrap().len(), 1);
        assert!(pending_ids(&conn, true).unwrap().is_empty());
    }

    #[test]
    fn deleting_a_message_cascades_history_but_invalid_receipts_are_not_ignored() {
        let conn = db::open_in_memory().unwrap();
        seed(&conn);
        prepare_attempt(&conn, &"E1".into(), attempt()).unwrap();
        conn.execute("UPDATE message_flag_sync SET attempt='invalid'", [])
            .unwrap();
        assert!(load(&conn).is_err());
        assert!(get(&conn, &"E1".into()).is_err());
        queries::delete_message_by_jmap_id(&conn, &"E1".into()).unwrap();
        assert!(load(&conn).unwrap().is_empty());
    }
}
