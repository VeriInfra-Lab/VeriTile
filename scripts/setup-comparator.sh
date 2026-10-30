#!/usr/bin/env bash
# Build the official judge/exporter for this project's pinned Lean version.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$#" -eq 1 ] || { echo "Usage: $0 <tools-directory>" >&2; exit 1; }
DEST="$(realpath -m "$1")"
COMPARATOR_REV=2a00b30df5e9173e70c4e4ec669fdf03da3163b9
# Keep the CLI v2 argument handling expected by the Lean 4.29 comparator.
LANDRUN_REV=5283024a2f49b28046c3b4a06d7d775c058d4d80
[ "$(cat "${ROOT}/lean-toolchain")" = 'leanprover/lean4:v4.29.0' ] || {
  echo 'Update the comparator/exporter pin for the new project toolchain first.' >&2; exit 1;
}
command -v go >/dev/null || { echo 'Go 1.24+ is required to build landrun.' >&2; exit 1; }
mkdir -p "${DEST}/bin"
for repo in comparator landrun; do
  if [ ! -d "${DEST}/${repo}" ]; then
    case "$repo" in
      comparator) url=https://github.com/leanprover/comparator.git ;;
      landrun) url=https://github.com/Zouuup/landrun.git ;;
    esac
    git clone "$url" "${DEST}/${repo}"
  fi
done
git -C "${DEST}/comparator" checkout --detach "${COMPARATOR_REV}"
git -C "${DEST}/landrun" checkout --detach "${LANDRUN_REV}"
# Materialize the locked dependencies before applying the upstream replay fix.
(cd "${DEST}/comparator" && lake env lean --version)
# Backport leanprover/lean4@94e0a0e4f6f4c3fa207aa7738fdd198067b4b3de
# to the Lean 4.29 checker. Both primitive definitions still go through the
# kernel; only their replay order changes. Reject unexpected checker sources.
checker_source="${DEST}/comparator/.lake/packages/Lean4Checker/Lean4Checker/Replay.lean"
checker_sha="$(sha256sum "${checker_source}" | cut -d ' ' -f 1)"
case "${checker_sha}" in
  7decb4f442215ffa9866d0801fc4eb8d21ba584334b5d662829bc595a848e60c)
    git -C "${DEST}/comparator" apply "${ROOT}/scripts/comparator-string-literals.patch" ;;
  b8a9cabc29f2fdd1cd34e626df072b3d84cffdab7efa704ec27a17aaa8ea988d) ;;
  *) echo 'Unexpected Lean4Checker replay source; review the backport before building.' >&2; exit 1 ;;
esac
printf '%s  %s\n' b8a9cabc29f2fdd1cd34e626df072b3d84cffdab7efa704ec27a17aaa8ea988d "${checker_source}" | sha256sum --check
(cd "${DEST}/comparator" && lake build comparator lean4export)
(cd "${DEST}/landrun" && go build -o "${DEST}/bin/landrun" ./cmd/landrun)
cp "${DEST}/comparator/.lake/build/bin/comparator" "${DEST}/bin/"
cp "${DEST}/comparator/.lake/packages/lean4export/.lake/build/bin/lean4export" "${DEST}/bin/"
printf 'Installed official comparator, lean4export and landrun. Add to PATH: %s/bin\n' "${DEST}"
