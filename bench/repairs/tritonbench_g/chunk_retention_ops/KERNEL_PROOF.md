# Kernel-only proof packet

This packet proves the four existing FP32 kernel quotations from
`chunk_retention_ops_repaired.py`. It adds no Python repair and changes no
guards, casts, optional-state behavior, or arithmetic order. The 51 files that
predate this packet remain byte-for-byte unchanged.

Import `KernelCorrect`, `KernelFPEquiv`, or the combined `KernelReader`.
Names below are in
`VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001`.

| Quotation / Python kernel suffix | Exact Real endpoint | Native FP32 endpoint |
| --- | --- | --- |
| `forwardH` / `fwd_kernel_h` | `KernelCorrect.forward_tensor` | `KernelFPEquiv.forward_tensor` |
| `forwardO` / `fwd_kernel_o` | `KernelCorrect.output_tensor` | `KernelFPEquiv.output_tensor` |
| `backwardH` / `bwd_kernel_dh` | `KernelCorrect.backward_tensor` | `KernelFPEquiv.backward_tensor` |
| `backwardQKV` / `bwd_kernel_dqkv` | `KernelCorrect.gradient_tensor` | `KernelFPEquiv.gradient_tensor` |

Each endpoint derives a state `t` with successful execution of the actual shared
quotation, its numerical observations, and `KernelTensor.Frame` for that same
state. There is no successful-execution, compiler-result, whole-source equality,
or desired-output premise. `KernelPacket.complete` also exposes the generic
execution, independent recurrence/scatter result, and exact logical frame.

The Real forward and reverse state targets are the dense mathematical
`MathTarget.state` / `finalState` and `future` / `initialGradient` recurrences.
Output and QKV use the independent scalar chunk contractions
`ScalarSource.outputTile`, `queryTile`, `keyTile`, and `valuePartial`.
For these standalone kernels, `h` and `dhistory` are genuine input buffers;
their equality to an earlier kernel's output is not assumed. The dV result has
shape `[keyBlocks,B,H,T,V]` and remains partial for each key block. No host dV
reduction or autograd composition is claimed by these endpoints.

Native observations are exact `Torch.FP32.logicalValue(...).bits` equalities to
the independently defined scalar `Reference.history`, `finalState`,
`reverseCarry`, `Body.output`, `Body.dq`, `Body.dk`, and `Body.dvPartial` formulas.
The forward scalar reference is reused from the sibling `chunk_retention`
packet. The bridge theorems `KernelNative.forward_target`, `output_target`,
`backward_target`, and `gradient_target` prove those formulas from the staged
input-only recurrence. In particular, `key_result_dot` and `key_result_eq`
prove the dK transposed q×dS dot's local materialization and separate final add.

`KernelTensor.View` binds an actual `Torch.FP32.Tensor`, its existing laws,
shape, contiguous layout, and resolved negative flag. `coordinates` proves the
storage-offset-aware address equals `dataPtr + 4*i`. `project` transfers only
the program's exact write footprint into the observed Tensor backing; all
other addresses keep the caller read value. Existing PID guards and descriptor
shapes imply every observed element bound through `footprint_bound` and its
per-role lemmas. `Frame` includes logical memory preservation, exact exterior
physical preservation, and preservation of every word of disjoint caller input
backing, including inactive storage-offset prefixes and tails. It uses ordinary
word observations, with no allocator, byte-heap, ownership-resource, or host
runtime protocol.

Exact Real arithmetic and native FP32 are separate interpretations. Native
theorems use the existing fixed-width `Numeric.ScalarLaws` and the local
`KernelStaging.LiteralLaws`; `KernelNative.reference_laws` witnesses the scalar
boundary's satisfiability. The staged dot interpretation retains source load
and cast boundaries, ordered FMA accumulation, and the selected local dK
materialization. These are explicit primitive/compiler-profile boundaries.
No equality of FP32 results to exact Real arithmetic, NaN normalization, or
global reassociation is introduced. Raw input bits remain unrestricted.

Saved evidence is under `.veritile-cache/kernel-evidence/chunk_retention_ops`:

- `quotation-check.json` verifies all four complete normalized quotation bodies
  against the selected unchanged Python source.
- `actual-binder.json` records 39/39 passing SM80 compilations using the installed
  Triton 3.7.1 binder and actual wrapper arguments, with Torch 2.13.0+cu130.
  These are offline compilation controls; GPU execution and native numerical
  conformance experiments were not performed.
- `frozen-packet.json` records source, proof, dependency, receipt, and object
  hashes together with the source-preservation and proof-audit results.
- `resume-public-reader-final.log` records the combined public theorem checks,
  standard-base axiom checks, and non-circularity checks.

Build the three entry modules using
`.veritile-cache/build-tools/workspace_lean_local_first_v2.py`. The packet does
not expand global grid-product compatibility or claim new host, autograd,
allocator, byte-heap, or runtime-resource correctness.
