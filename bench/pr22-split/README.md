# PR #22 source stack

This stack splits the original body-copy work into 12 ordered review units.
Each part retains a previously verified source snapshot and its required support changes.
No new kernel migration or numerical experiment is added by splitting.

## Immutable archive

The complete original branch, including failed development attempts and full proof logs,
is retained on `codex/pr22-archive-20261029` at
[`5ea0e46126cc1c6eb05e4219ddc5dc13246ec3ce`](https://github.com/VeriInfra-Lab/VeriTile/tree/5ea0e46126cc1c6eb05e4219ddc5dc13246ec3ce).
The branch is an archive and must not receive later work. Commit links are authoritative.
Historical `bench/review-checkpoints/...` paths missing from this source stack resolve
against that commit. Base-branch evidence is retained in the fixed fp-examples
archive described in [the main integration](../main-integration/README.md).

`receipts/` contains verbatim verification fields extracted from archived reviews; each
excerpt pins the complete review by commit, path and SHA-256. These are historical
results for the named source snapshot, not fresh results for this split. Full semantic
reviews and detailed theorem/log inventories are available through the archive links.
Current task formula/IO/reader manifests remain local, and their normal guards still run.

`stages/` records source identities and separate integration checks for each part.
The newer `fp-examples` base is `25ad2949386cd0b992e2d3cb7cf26fac86f6a25c`. Its benchmark additions, guarded FP interface
and library targets are retained. Existing LogicalScheduledKernelIO definitions are
unchanged by its additive ImplementsOn API; current source pins reflect that base.

## Review and merge

Review and merge parts in numeric order. Each later PR targets the preceding branch.
Use merge commits or rebase the remaining stack after squash-merging; otherwise GitHub
can show already merged changes again. PRs #25–#36 were subsequently merged in numeric order into `fp-examples`;
the closed #22 PR records each resulting merge commit.

The final source coverage remains 124/183 selected tasks and 19/19 worked examples.
The remaining 59 tasks are outside this stopped batch. Archived comparator unavailability,
missing Torch/Triton and the known Structural receipt test failure remain explicit limits.

## Final integration and replay

The ordered PR links are maintained in the [#22 review index](https://github.com/VeriInfra-Lab/VeriTile/pull/22).
`final-integration.json` records the new full-stack build, joint import, compatibility
build of the newer base's benchmark libraries, Python results and archive checks.
The project cache is reused and affected dependencies are rebuilt. Compatibility
compilation does not discharge the new benchmark drafts' open proof/backend goals.

To replay the exact recorded root set and joint import after building dependencies:

```sh
python3 - <<'PY'
import gzip
from pathlib import Path
import subprocess
import tempfile
roots = gzip.decompress(Path("bench/pr22-split/final-roots.txt.gz").read_bytes()).decode().splitlines()
subprocess.run(["lake", "build", *roots], check=True)
with tempfile.TemporaryDirectory(prefix="veritile-pr22-") as temporary:
    joint = Path(temporary) / "Joint.lean"
    joint.write_text("".join(f"import {module}\n" for module in roots))
    subprocess.run(["lake", "env", "lean", str(joint)], check=True)
PY
lake build BugDetection PretrainKernels
```
