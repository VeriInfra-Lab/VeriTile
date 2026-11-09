#!/usr/bin/env bash
set -euo pipefail

# These assets match the repository toolchain and its locked Mathlib cache reader.
if [[ $(cat lean-toolchain) != leanprover/lean4:v4.29.0 || $(uname -sm) != 'Linux x86_64' ]]; then
  echo 'The release fallback requires Lean v4.29.0 on Linux x86-64.' >&2
  exit 1
fi

retry() {
  local delay
  for delay in 0 5 15; do
    sleep "$delay"
    if "$@"; then
      return 0
    fi
  done
  return 1
}

download_release() {
  local asset=$1 digest=$2 archive=$3
  timeout 300 gh api -H 'Accept: application/octet-stream' "$asset" >"$archive" &&
    printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
}

release_dir=$(mktemp -d "$RUNNER_TEMP/veritile-lean.XXXXXX")
archives=$(mktemp -d "$RUNNER_TEMP/veritile-lean-archives.XXXXXX")
trap 'rm -rf -- "$archives"' EXIT

retry download_release repos/leanprover/lean4/releases/assets/382932276 \
  089f7e513ed3e9436d191f4986330c0b53809c86b5997f5eb7f25550c374fa48 "$archives/lean.tar.zst"
tar --zstd -xf "$archives/lean.tar.zst" --strip-components=1 -C "$release_dir"

retry download_release repos/digama0/leangz/releases/assets/343278048 \
  46564c5a15b0a5919dee06b246ac1297399968c828269f675212104d7d1ac445 "$archives/leantar.tar.gz"
tar -xzf "$archives/leantar.tar.gz" --strip-components=1 -C "$archives"
export MATHLIB_CACHE_DIR=${MATHLIB_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/mathlib}
mkdir -p "$MATHLIB_CACHE_DIR"
install -m 0755 "$archives/leantar" "$MATHLIB_CACHE_DIR/leantar-0.1.17"

export PATH="$release_dir/bin:$PATH"
export MATHLIB_CACHE_GET_URL=${MATHLIB_CACHE_GET_URL:-https://cache.mathlib.org/mathlib4}
lean --version
"$MATHLIB_CACHE_DIR/leantar-0.1.17" --version
retry lake exe cache get
read -r -a build_targets <<< "$VERITILE_BUILD_ARGS"
lake build "${build_targets[@]}"
printf '%s\n' "$release_dir/bin" >> "$GITHUB_PATH"
