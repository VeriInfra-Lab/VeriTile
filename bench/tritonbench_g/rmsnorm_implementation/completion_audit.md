# Scoped proof completion audit

The final public-entry and separate audit build passed on 2026-10-13:

```bash
PATH=/tmp/veritile-lean-toolchain/bin:$PATH LEAN_NUM_THREADS=2 lake build \
  +bench.tritonbench_g.rmsnorm_implementation.Correct \
  +bench.tritonbench_g.rmsnorm_implementation.FPEquiv \
  +bench.tritonbench_g.rmsnorm_implementation.Proofs.Audit
```

The build completed all 3450 jobs. The log is
`/tmp/rmsnorm_implementation_final_public_build.log`. The explicit audit covers
20 key declarations and exhaustively checks all 894 theorem declarations in
the port, including private helpers, against the standard axiom set
`propext`, `Classical.choice`, `Quot.sound`. Both independent specifications pass
noncircularity checks, and the structural FP assumption list is empty.

The public entries reach 13 local modules; the separate audit reaches all 14,
with no orphan helpers. All 111 declaration headers and 50 docstrings from
`0ad438f5^:bench/tritonbench_g/rmsnorm_implementation/RmsnormImplementation.lean`
remain unchanged. The checkpoint `0ad438f5` also retains all 167 declaration
headers and 70 docstrings. No historical contract was weakened or deleted.

The specification sheets under `/tmp/rmsnorm_implementation_spec_sheets` report
zero self-reference flags and zero missing summaries. The five public mappings
are in `/tmp/rmsnorm_implementation_public_records.json`. `git diff --check` and
the local forbidden-proof-token scan pass.

The repeatable native check is:

```bash
python3 bench/tritonbench_g/rmsnorm_implementation/Proofs/native_lowering.py \
  --output /tmp/rmsnorm_implementation_native_lowering.json
```

The recorded run returns exit 0 for all 44 expected outcomes: 36 compiled
floating profiles, four floating profiles rejected for unsupported fp8/fp64
conversions, and four admitted nonfloating source profiles outside the formal
family. The stored receipt records exact scope and diagnostics. Expected
compiler rejection does not count as verified native execution.

The dimension-general mathematical and structural theorems, all-profile finite
allocation witness, and four complete typed numerical/frame witnesses are
checked. The numerical backend performs actual real sum, multiplication,
division and square root; casts preserve values, unused unsupported primitives
return bottom (undefined) rather than fabricated numeric values. These results do not prove IEEE
rounding or GPU behavior.

The whole source domain is still unfinished: `source_scope.md` lists integer and
Boolean storage, signed address geometry, native admission/width/conversion,
host-grid/race and numerical obligations. Independent comparator replay remains
pending because its environment is unavailable. Registry, manifest and ledger
integration are owned by the coordinating agent and are separate from this
per-port receipt.
