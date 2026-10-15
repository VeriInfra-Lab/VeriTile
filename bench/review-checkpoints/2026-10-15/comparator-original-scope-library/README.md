# Official library replay at 93d38e39

The unchanged official comparator accepted **118/118 theorem targets in eight
library modules**, with runner and comparator exit codes both zero. The exact
source is commit `93d38e399ac513eb98aa3624d9aeb1c21858757a` on `fp-examples`.
[original-result.json](original-result.json) is the untouched official result;
[target-mapping.json](target-mapping.json) lists the original manifest targets
and generated aliases. [verification.json](verification.json) records the run.

This is a library-only result. It does not certify the six new public entrypoint
files, the complete 30-module first extension batch, the subsequent LoRA/matrix
batch, the current corpus, or any of the 183 whole original programs.

The official runner executed as nonprivileged UID 2000 in an isolated Debian 13
VM with Landlock ABI 6 and systemd user services. The official preflight and
positive/negative isolation controls passed. The real comparator accepted the
valid control and rejected a changed statement, `sorryAx`, and a source-write
attempt. No isolation check was bypassed. Tools and frozen source media were
read-only, and the judge used its own copied cache.

[replay-evidence.tar.gz](replay-evidence.tar.gz) preserves the complete original
logs, official result/config/target files, before/after source guards, collection
scripts, pinned tool provenance, isolation controls and producer README. Its
SHA256 is `ffc0298ff6736c97fd3632235eb9bfbdd7df1f1fe7ab60a6088c2541247af3b0`.
No build cache, tool binaries, VM images or credentials are retained.

Root independently checked all **7,177 Git blobs** against the frozen manifest
and all **71 archive payload files** against the producer's `SHA256SUMS`.
Before/after source maps agree; the 4,303 official snapshot inputs also agree
after the collector normalizes the unchanged runner's generated library-target
registration. [commit-binding.json](commit-binding.json) and
[retention.json](retention.json) preserve these bindings and retention hashes.

To inspect and verify the full evidence from this directory:

```sh
evidence_dir=$(mktemp -d /tmp/veritile-library-evidence.XXXXXX)
tar -xzf replay-evidence.tar.gz -C "$evidence_dir"
cd "$evidence_dir/library-93d38e39-evidence"
sha256sum --check SHA256SUMS
```

The compressed retention avoids adding the expanded logs and source manifests
as separate working-tree files. It changes neither the original receipts nor
the current full-scope acceptance count, which remains 0/183.
