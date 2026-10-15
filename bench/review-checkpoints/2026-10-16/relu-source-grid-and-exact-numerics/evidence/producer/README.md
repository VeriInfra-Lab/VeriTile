# ReLU: checked bytes composed with the frozen typed source and full grid

This isolated producer package adds six Lean modules and two public contracts.
It preserves the twenty predecessor sources and all original Python bytes.
Whole-original-program acceptance remains false; this package adds zero completed
original programs and zero dtype profiles.

`SourceStages.statement` indexes the actual seven-statement body of frozen
`TypedValue.source encoding.profile`. Each trace retains the pid, block start,
offset, mask, loaded x, result, and final state. The proof supplies all seven
`FP.Structural.step` equations and derives actual original `exec` success.
The independent scalar/register reference does not call the source or interpreter.

The checked load hook consumes the actual offset and mask registers, reads each
active lane through the inherited allocation/read-permission byte interface, and
places the decoded values into the original x register. Inactive lanes retain
the incoming `State.undef` value. The original source load has no `other=0`;
the `.zero` padding argument does not replace that ordinary masked-load fallback.
The hook is proved equal to the original load step under the input byte/cell view.

The actual original comparison/conversion/where statement runs on every pid,
including all inactive lanes. Its complete result register is retained before
the gate. Numerical primitives remain pure total callbacks, so this is a value
and state trace, not evidence of native evaluation order or side effects.

The checked store hook extracts and evaluates the original source gate and
consumes the actual result/offset/mask registers. Each enabled fp32 output is
stored through allocation/write checks in lane and byte order. The logical
original store statement is also executed, and its exact source-step equation
is included in the same trace. An unavailable active input byte fails before
the result statement and gate, including for nonzero pids. Store failures retain
the partial byte memory supplied by the inherited checked-write interface.

`CheckedSchedule.runSchedule` passes each returned logical state and byte memory
to the next pid, rather than restarting programs from an unrelated initial view.
It retains one complete source trace for every schedule entry. Distinct logical
input/output names and nonoverlapping byte intervals preserve the input view
between blocks. Every complete permutation of the original ceil-grid succeeds
under the stated permissions and reaches the frozen independent word and byte
specifications, with a final LogicalView. The output tail is unchanged and needs
no allocation, read or write permission. N=0 yields an empty trace list.

The same nine integer/bool profiles are covered: signed/unsigned 8/16/32/64-bit
words and all 256 bool storage bytes. The two inherited numerical followups and
three callbacks are unchanged. No fp16/bf16/fp32 execution, fp64/fp8 route, frontend
admission/error family, host/module stage, native allocation/layout/provenance,
native compiler bridge, or instruction interleaving is newly closed. All address
arithmetic in these contracts is mathematical Nat arithmetic; inherited finite
address certificates remain separate. Whole-program permutations are not native
GPU schedules. The block contract binds the active input view; the grid contract
additionally proves the complete final input/output LogicalView under separation.

Verification uses the existing Lean toolchain and imported project oleans only.
`verify.py` records every real compile attempt, including failed attempts and a
deliberately terminated first attempt; each log retains its exact source snapshot.
`freeze.py before/after` binds transitive sources, imported project oleans,
twenty predecessor sources, thirteen inherited TTIR artifacts, the original
Python source, actual lakefile.toml/manifest/toolchain, and the declaration parser.
It does not rebuild external libraries, recompile frontend/native code, run a GPU,
or run the repository comparator. Root owns repository integration and acceptance.

Public declarations:

- `SourceBlockCorrect.relu_checked_source_block_correctness`
- `SourceGridCorrect.relu_checked_source_grid_correctness`

Final reproduction sequence, from the repository working directory:

```sh
python3 -B /tmp/veritile-standardization-20261015/relu-source-coupled-block-wip/freeze.py before
/tmp/veritile-lean-toolchain/bin/lake env python3 -B /tmp/veritile-standardization-20261015/relu-source-coupled-block-wip/verify.py bench/tritonbench_g/relu_triton_kernel/Proofs/SourceStages.lean bench/tritonbench_g/relu_triton_kernel/Proofs/CheckedTileIO.lean bench/tritonbench_g/relu_triton_kernel/Proofs/CheckedBlock.lean bench/tritonbench_g/relu_triton_kernel/SourceBlockCorrect.lean bench/tritonbench_g/relu_triton_kernel/Proofs/CheckedSchedule.lean bench/tritonbench_g/relu_triton_kernel/SourceGridCorrect.lean
python3 -B /tmp/veritile-standardization-20261015/relu-source-coupled-block-wip/freeze.py after
python3 -B /tmp/veritile-standardization-20261015/relu-source-coupled-block-wip/finalize.py
```

`finalize.py` is a one-time producer action and refuses to overwrite a receipt.
Reproduction of an already frozen package should compare the existing receipt
and run checks in a separate copy, preserving its original evidence.
