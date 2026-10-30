#!/usr/bin/env bash
#
# External machine-checkable TRUST AUDIT for the standalone bench corpus.
#
# The library theorems are audited in-tree by VeriTile/Meta/TrustReport.lean
# (`lake build VeriTile.Meta.TrustReport`). This script covers the OTHER
# population — the standalone, single-file bench elaborations that are NOT
# importable:
#
#   * bench/tritonbench_g/*/*.lean   (the 173 TritonBench-G ports)
#   * bench/examples/**/*.lean       (the kernel showcases, including subdirectories)
#   * tests/lean/*.lean             (infra smoke tests / regression gates)
#
# For each file the shared comparator driver emits a snapshot copy that adds
# `import VeriTile.Meta.StatementAudit` and appends `#auditModuleAxioms` and
# `#auditModuleSpecs` for environment-based discovery, compiles it, and requires
# official comparator export/replay. The port files themselves are never modified —
# the corpus stays clean.
#
# A `#axiomsClean` failure means a `sorry`/smuggled axiom leaked into a proof
# the corpus calls complete: a REAL soundness finding. Do not paper it over.
#
# Independent single-file elaborations run in parallel (see scripts/bench/check_ports.sh
# for the same concurrency model). Override with AUDIT_TRUST_JOBS=N.
#
# Usage:
#   scripts/bench/audit_trust.sh                 # audit the whole bench corpus
#   scripts/bench/audit_trust.sh --ci-targets    # public entries, sources and tests
#   scripts/bench/audit_trust.sh <kernel> ...    # audit only named tritonbench_g kernels
#   scripts/bench/audit_trust.sh --catalog PATH [<kernel> ...]
#     Use the explicit current catalog for multi-entry/typed source layouts.
#   AUDIT_TRUST_JOBS=4 scripts/bench/audit_trust.sh
#   AUDIT_TRUST_SHARD_COUNT=4 AUDIT_TRUST_SHARD_INDEX=0 scripts/bench/audit_trust.sh
# Shards use zero-based indices; every shard must pass to audit the full corpus.
#
# Exit codes:
#   0 — every audited file passes its trust gates and official comparator replay
#   1 — at least one file failed; failing names are printed
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${PROJECT_ROOT}"

PORTS_ROOT="bench/tritonbench_g"
EXAMPLES_ROOT="bench/examples"
TESTS_ROOT="tests/lean"

catalog=''
catalog_sha256=''
final_inventory=false
inventory_targets='--targets'
if [ "$#" -eq 0 ]; then
  set -- --final-inventory
fi
if [ "${1-}" = '--final-inventory' ] || [ "${1-}" = '--ci-targets' ]; then
  final_inventory=true
  if [ "$1" = '--ci-targets' ]; then
    inventory_targets='--ci-targets'
  fi
  catalog='validation/source.json'
  catalog_sha256="$(sha256sum -- "${catalog}")" || exit "$?"
  shift
  if [ "$#" -ne 0 ]; then
    printf 'Final inventory modes cannot be combined with named targets\n' >&2
    exit 2
  fi
fi
if [ "${1-}" = '--catalog' ]; then
  if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" = --* ]]; then
    printf 'Usage: %s [--catalog PATH] [<kernel> ...]\n' "$0" >&2
    exit 2
  fi
  catalog="$2"
  catalog_sha256="$(sha256sum -- "${catalog}")" || exit "$?"
  shift 2
fi

select_targets() {
  if [ "${final_inventory}" = true ]; then
    python3 scripts/bench/final_inventory.py "${inventory_targets}" || return $?
    return
  fi
  if [ -n "${catalog}" ]; then
    python3 scripts/bench/current_source_catalog.py targets --root "${PROJECT_ROOT}" \
      --catalog "${catalog}" "$@" || return $?
    if [ "$#" -eq 0 ]; then
      find "${EXAMPLES_ROOT}" -type f -name '*.lean' | sort || return $?
      find "${TESTS_ROOT}" -maxdepth 1 -type f -name '*.lean' | sort || return $?
    fi
    return
  fi
  if [ "$#" -eq 0 ]; then
    python3 scripts/bench/port_layout.py all-proofs "${PORTS_ROOT}" || return $?
    find "${EXAMPLES_ROOT}" -type f -name '*.lean' | sort
    find "${TESTS_ROOT}" -maxdepth 1 -type f -name '*.lean' | sort
  else
    for name in "$@"; do
      local lean
      lean=$(python3 scripts/bench/port_layout.py proofs "${PORTS_ROOT}/${name}/") || return $?
      if [ -n "${lean}" ]; then
        printf '%s\n' "${lean}"
      else
        printf 'no .lean file for bench target: %s\n' "${name}" >&2
        exit 2
      fi
    done
  fi
}

default_jobs() {
  local cores mem_gb by_mem jobs
  cores=$(nproc 2>/dev/null || echo 1)
  mem_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 4)
  by_mem=$((mem_gb / 8))
  jobs=$((cores < by_mem ? cores : by_mem))
  [ "${jobs}" -gt 32 ] && jobs=32
  [ "${jobs}" -lt 1 ] && jobs=1
  printf '%d\n' "${jobs}"
}

JOBS="${AUDIT_TRUST_JOBS:-$(default_jobs)}"
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'AUDIT_TRUST_JOBS must be a positive integer: %s\n' "${JOBS}" >&2
  exit 2
fi

target_list="$(select_targets "$@")" || exit "$?"
if [ -z "${target_list}" ]; then
  printf 'No trust audit targets selected\n' >&2
  exit 2
fi

audit_scope='bench corpus'
if [[ -v AUDIT_TRUST_SHARD_COUNT || -v AUDIT_TRUST_SHARD_INDEX ]]; then
  if [ "$#" -ne 0 ]; then
    printf 'Trust audit sharding cannot be combined with named targets\n' >&2
    exit 2
  fi
  target_list="$(printf '%s\n' "${target_list}" | python3 "${SCRIPT_DIR}/select_audit_shard.py" \
    "${AUDIT_TRUST_SHARD_COUNT-}" "${AUDIT_TRUST_SHARD_INDEX-}")" || exit "$?"
  audit_scope="bench corpus shard ${AUDIT_TRUST_SHARD_INDEX}/${AUDIT_TRUST_SHARD_COUNT} (zero-based)"
  printf 'Selected %s: %s files\n' "${audit_scope}" "$(printf '%s\n' "${target_list}" | wc -l)"
fi

TMPDIR_AUDIT="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_AUDIT}"' EXIT

export PROJECT_ROOT TMPDIR_AUDIT

launcher_status=0
results="$(printf '%s\n' "${target_list}" | xargs -P "${JOBS}" -I{} bash -c '
  src="{}"
  rel="${src#'"${PROJECT_ROOT}"'/}"
  # Nested examples share names such as Proofs/FP.lean. Use the full path
  # so concurrent workers cannot overwrite another example audit log.
  tag="$(printf "%s" "${rel}" | sha256sum | cut -d " " -f 1)"
  tmp="${TMPDIR_AUDIT}/${tag}"
  if python3 "${PROJECT_ROOT}/scripts/check_comparator.py" --file "${src}" --trust \
      --workspace "${TMPDIR_AUDIT}/judge" > "${tmp}.log" 2>&1; then
    summary=$(sed -n "/^Axiom audit:/p; /^Spec audit:/p; /^Comparator:/p" "${tmp}.log" | paste -sd ";")
    printf "%s: %s\n" "${rel}" "${summary}" >&2
    printf "  ok    %s\n" "${rel}"
  else
    cat "${tmp}.log" >&2
    printf "  FAIL  %s\n" "${rel}"
  fi
')" || launcher_status=$?

if [ "${launcher_status}" -ne 0 ]; then
  printf 'Trust worker launcher failed (exit %s)\n' "${launcher_status}" >&2
  exit 1
fi

expected="$(printf '%s\n' "${target_list}" | sed "s|^${PROJECT_ROOT}/||")"
if ! python3 "${SCRIPT_DIR}/check_worker_results.py" "${expected}" "${results}"; then
  exit 1
fi

# A worker snapshot must not hide edits after initial catalog selection. The
# normal checks below still require every selected worker to pass independently.
if [ -n "${catalog}" ]; then
  if [ "$(sha256sum -- "${catalog}")" != "${catalog_sha256}" ]; then
    printf 'Current source catalog changed during trust audit\n' >&2
    exit 1
  fi
  if ! select_targets "$@" > /dev/null; then
    printf 'Current source catalog changed during trust audit\n' >&2
    exit 1
  fi
fi

printf '%s\n' "${results}" | sort

passed=$(printf '%s\n' "${results}" | grep -c '^  ok    ' || true)
failed=$(printf '%s\n' "${results}" | grep -c '^  FAIL  ' || true)

printf '\nTrust audit (%s): %d ok, %d fail\n' "${audit_scope}" "${passed}" "${failed}"

if [ "${failed}" -gt 0 ]; then
  printf 'Files whose trust gate FAILED (inspect diagnostics for proof or infrastructure failure):\n'
  printf '%s\n' "${results}" | awk '/^  FAIL  / {print "  - " $2}'
  exit 1
fi
