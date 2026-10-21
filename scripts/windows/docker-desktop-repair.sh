#!/usr/bin/env bash
#
# Docker Desktop (WSL2 backend) repair — Windows ARM64 / Git Bash.
#
# The Docker Engine on this machine is the WSL2 backend of Docker Desktop.
# When it refuses to start, the cause is almost always one of:
#   1. the WSL2 virtual machine is still running, so Docker Desktop cannot
#      restart its own backend  -> fix: wsl --shutdown
#   2. a custom WSL kernel in .wslconfig; the Docker backend only starts on
#      the Microsoft stock kernel  -> fix: kernel-reset
#   3. a damaged "docker-desktop" distro  -> fix: rebuild-distro
#   4. stuck Docker Desktop processes  -> fix: force-restart
#
# Usage:
#   bash docker-desktop-repair.sh status
#   bash docker-desktop-repair.sh restart
#   bash docker-desktop-repair.sh kernel-reset
#   bash docker-desktop-repair.sh rebuild-distro
#   bash docker-desktop-repair.sh force-restart
#
# Every run appends to %TEMP%\auris-docker-repair.log and prints the same
# lines to the terminal.

set -u
export MSYS_NO_PATHCONV=1

LOG="${TEMP:-/tmp}/auris-docker-repair.log"
ENGINE_TIMEOUT="${AURIS_DOCKER_TIMEOUT:-600}"

log() {
    printf '%s %s\n' "$(date '+%H:%M:%S')" "$*"
    printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" >>"$LOG" 2>/dev/null || true
}

# The WSL tooling prints UTF-16; strip the NUL bytes so the output is readable.
wsl_l() { timeout 60 wsl.exe "$@" 2>&1 | tr -d '\000'; }

# Docker Desktop's own process names, in the order they must die.
DD_PROCESSES=(
    "Docker Desktop.exe"
    "com.docker.backend.exe"
    "com.docker.build.exe"
    "com.docker.dev-envs.exe"
    "docker-desktop.exe"
)

server_version() {
    timeout 30 docker version --format '{{.Server.Version}}' 2>/dev/null \
        | tr -d '\r' | head -1
}

engine_is_up() {
    local v
    v="$(server_version)"
    [ -n "$v" ] && ! printf '%s' "$v" | grep -qiE '500|error|cannot'
}

kill_dd_processes() {
    local p
    for p in "${DD_PROCESSES[@]}"; do
        if taskkill //IM "$p" //F //T >/dev/null 2>&1; then
            log "kilenced: $p"
        fi
    done
}

# Print everything needed to tell a healthy engine from a broken one.
show_status() {
    log "--- wsl --version ---"
    wsl_l --version | head -6
    log "--- wsl -l -v ---"
    wsl_l -l -v
    log "--- .wslconfig (kernel sor) ---"
    local cfg="${USERPROFILE:-/c/Users/$USERNAME}/.wslconfig"
    if [ -f "$cfg" ]; then
        grep -n -i "kernel" "$cfg" || log "(nincs kernel sor: a stock kernel fut)"
    else
        log "(nincs .wslconfig)"
    fi
    log "--- docker desktop status ---"
    timeout 60 docker desktop status 2>&1 | tr -d '\r' | head -4
    log "--- docker version ---"
    timeout 60 docker version 2>&1 | tr -d '\r' | head -22
    log "--- docker context ls ---"
    timeout 60 docker context ls 2>&1 | tr -d '\r' | head -6
}

# Wait until the engine answers, or give up after ENGINE_TIMEOUT seconds.
wait_for_engine() {
    local i v st
    for ((i = 1; i * 10 <= ENGINE_TIMEOUT; i++)); do
        sleep 10
        v="$(server_version)"
        st="$(timeout 30 docker desktop status 2>&1 | tr -d '\r' | grep -i 'status' | head -1)"
        log "poll[$i] $st | server=${v:-nincs}"
        if engine_is_up; then
            log "ENGINE_UP (${i}0 mp után)"
            timeout 60 docker version 2>&1 | tr -d '\r' | head -22
            timeout 60 docker info 2>&1 | tr -d '\r' \
                | grep -E 'Storage Driver|Docker Root Dir|Total Memory|Architecture|Operating System|Kernel Version' | head -10
            return 0
        fi
    done
    log "ENGINE_DOWN: ${ENGINE_TIMEOUT}s alatt nem indult el"
    timeout 60 docker desktop status 2>&1 | tr -d '\r' | head -5
    return 1
}

# 1. lépés — szelényes újraindítás.
do_restart() {
    log "=== restart ==="
    log "--- docker desktop stop ---"
    timeout 180 docker desktop stop 2>&1 | tr -d '\r' | head -3
    sleep 5
    log "--- wsl --shutdown ---"
    timeout 180 wsl.exe --shutdown 2>&1 | tr -d '\000'
    sleep 8
    log "--- docker desktop start ---"
    timeout 300 docker desktop start 2>&1 | tr -d '\r' | head -3
    wait_for_engine
}

# 2. lépés — a saját WSL kernel kikapcsolása, majd újraindítás.
do_kernel_reset() {
    local cfg="${USERPROFILE:-/c/Users/$USERNAME}/.wslconfig"
    log "=== kernel-reset ==="
    if [ ! -f "$cfg" ]; then
        log "nincs .wslconfig: a stock kernel eleve fut"
    elif grep -qiE '^\s*kernel\s*=' "$cfg"; then
        log "saját kernel be van kapcsolva -> kikapcsolom ($cfg)"
        cp "$cfg" "$cfg.auris-backup"
        # A kernel= sort megjegyzésbe teszem, hogy visszaállítható legyen.
        sed -i -E 's/^([[:space:]]*)kernel[[:space:]]*=/#\1kernel=/' "$cfg"
        log "a módosított sor: $(grep -n -i kernel "$cfg" | head -1)"
    else
        log "a .wslconfig-ban nincs aktív kernel sor"
    fi
    do_restart
}

# 3. lépés — a "docker-desktop" distro újraépítése.
# A "docker-desktop-data" (felhasználói image-ek) MEGMARAD.
do_rebuild_distro() {
    log "=== rebuild-distro ==="
    log "--- DD leállítás ---"
    timeout 180 docker desktop stop 2>&1 | tr -d '\r' | head -3
    sleep 5
    kill_dd_processes
    sleep 4
    log "--- wsl --shutdown ---"
    timeout 180 wsl.exe --shutdown 2>&1 | tr -d '\000'
    sleep 8
    log "--- distro állapot előtte ---"
    wsl_l -l -v
    log "--- wsl --unregister docker-desktop (a felhasználói adatok NEM ebben vannak) ---"
    timeout 300 wsl.exe --unregister docker-desktop 2>&1 | tr -d '\000'
    sleep 6
    log "--- distro állapot utána ---"
    wsl_l -l -v
    log "--- DD start ---"
    timeout 300 docker desktop start 2>&1 | tr -d '\r' | head -3
    wait_for_engine
}

# 4. lépés — beragadt processzek leállítása, majd újraindítás.
do_force_restart() {
    log "=== force-restart ==="
    kill_dd_processes
    sleep 6
    log "--- maradék wsl.exe processzek ---"
    local pids p
    pids="$(tasklist /FI "IMAGENAME eq wsl.exe" /FO CSV /NH 2>/dev/null \
        | tr -d '"' | cut -d, -f2 | tr -d '\r')"
    for p in $pids; do
        if taskkill //PID "$p" //F >/dev/null 2>&1; then
            log "kilenced: wsl.exe (PID $p)"
        fi
    done
    sleep 4
    log "--- wsl --shutdown ---"
    timeout 180 wsl.exe --shutdown 2>&1 | tr -d '\000'
    sleep 8
    log "--- docker desktop start ---"
    timeout 300 docker desktop start 2>&1 | tr -d '\r' | head -3
    wait_for_engine
}

case "${1:-status}" in
    status)         show_status ;;
    restart)        do_restart ;;
    kernel-reset)   do_kernel_reset ;;
    rebuild-distro) do_rebuild_distro ;;
    force-restart)  do_force_restart ;;
    *)
        echo "használat: bash $(basename "$0") {status|restart|kernel-reset|rebuild-distro|force-restart}" >&2
        exit 2
        ;;
esac
