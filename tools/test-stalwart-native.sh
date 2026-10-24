#!/usr/bin/env bash
# Build outside the sandbox, then run the real flag suite with no host network
# or /var writes. Requires Linux user namespaces, Stalwart v0.16, Python 3,
# util-linux (unshare/mount), iproute2, and the normal Rust build environment.
set -euo pipefail

for tool in cargo python3 stalwart unshare mount ip realpath; do
    command -v "$tool" >/dev/null || { printf 'Missing native-test tool: %s\n' "$tool" >&2; exit 1; }
done

export JMA_TEST_NATIVE_STALWART
JMA_TEST_NATIVE_STALWART=$(realpath "$(command -v stalwart)")
# Cargo's structured output avoids guessing a target-directory/hash and avoids
# needing Cargo, downloads, or a compiler inside the private network namespace.
test_binary=$(
    cargo test --locked --test e2e_flag_sync --no-run --message-format=json |
        python3 -c 'import json,os,sys
paths=[]
for line in sys.stdin:
    item=json.loads(line)
    if item.get("reason")=="compiler-artifact" and item.get("target",{}).get("name")=="e2e_flag_sync" and item.get("executable"):
        paths.append(item["executable"])
if len(paths)!=1:
    raise SystemExit("Could not identify the compiled e2e_flag_sync executable")
print(os.path.realpath(paths[0]))'
)
case "$test_binary:$JMA_TEST_NATIVE_STALWART" in
    /var/*|*:/var/*) printf 'Native test binaries must live outside the isolated /var\n' >&2; exit 1 ;;
esac

exec unshare --user --map-root-user --net --mount --propagation private -- bash -c '
set -euo pipefail
mount -t tmpfs tmpfs /var
mkdir -p /var/log /var/lib /var/cache
ip link set lo up
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy DISPLAY
export NO_PROXY=127.0.0.1,localhost,::1
export DBUS_SESSION_BUS_ADDRESS=unix:path=/var/jma-no-session-bus
exec "$1" --ignored --nocapture --test-threads=1
' jma-native "$test_binary"
