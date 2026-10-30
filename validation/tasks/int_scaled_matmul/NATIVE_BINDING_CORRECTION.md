# Historical INT8 native-binding correction

The retained abstract typed theorems are valid in their stated modular32
interpreter. Their applicability to the historical native INT8 lowering is
contradicted by a concrete saturating-accumulator case. This correction records
the difference; it does not rewrite the archived source or review and does not
claim a GPU measurement.

The exact historical package is under
`/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/veritile-work/bench/tritonbench_g/int_scaled_matmul/revision001/`.

| Historical file | SHA256 |
| --- | --- |
| `int_scaled_matmul_repaired.py` | `c14ed9abf98543377ebb9883f14718b468e319d1b596db9ac7846ba5023a7fab` |
| `Correct.lean` | `6f83c847fa0a689a1c7fd73ec0814a4546247b8960b8968143bce941c3fe0aef` |
| `FPEquiv.lean` | `02e25f25997bd0477b1271522059f5467849b73cf497c4b7bc44acc7121a7e3b` |
| `Proofs/Host.lean` | `7aff51b5472f9adfaf7e2b9bb5226f62bb96b90caeeb75c8fb9f8171ca049536` |
| `Proofs/NativeCorrespondence.lean` | `82a22856d8804e14cfe26e8af8155476604b4fcb4a9fffe57d52b9680e2e3fc3` |

The affected historical public claims are `int_matmul_correct` and
`int_scaled_matmul_correct` in `Correct.lean`, and
`int_scaled_matmul_native_correct` in `FPEquiv.lean`, all in namespace
`VeriTile.Bench.Repairs.IntScaledMatmul.Revision001`. The last theorem's three
FP scalar laws do not cover or repair the integer-dot lowering discrepancy.
The underlying Lean results remain conditional results about their declared
typed interpreter; their native applicability must be corrected.

Use M=N=1, K=131072; A/B are all -128 in INT8. A has strides `(131072,1)`
and capacity 131072; B has strides `(1,1)` and capacity 131072; C is INT32
with strides `(1,1)` and capacity 1. All tensors use one CUDA device and
distinct regions. Config is BM=BN=16, BK=32, GROUP_M=8, warps=4, stages=2,
CTAs=1. These satisfy every historical `Host.Valid` field: matching shapes,
storage types/device, `ViewBounds`, output injectivity, separation and positive
tile dimensions. The input words satisfy `Int8Storage`. This witness also
satisfies the current checked index envelope; no added guard excludes it.

Each K block contributes 524288. The 4096-block exact sum is 2147483648.
The modular32 interpreter returns -2147483648. Carry-fed signed32 saturation
returns 2147483647. For the scaled theorem, use one FP32 row scale equal to
0.5. It satisfies `ScaleValid` and `FP32Storage`. The independent historical
logical result promotes -2147483648 exactly, multiplies by 0.5, and converts
to -1073741824. Thus its `LogicalDefined` premise holds. The observed native
integer recurrence instead supplies 2147483647; nearest-even FP32 promotion
produces 2147483648, multiplication by 0.5 produces 1073741824, and the final
conversion is also defined. The two INT32 outputs differ even when every
historical local FP correspondence holds.

The exact archived Python was compiled offline with Triton 3.7.1 for
`GPUTarget("cuda",90,32)`, i32 scalar metadata, the configuration above, and
`EVEN_K=True`/`ACC_TYPE=int32` for the scaled kernel. Both TTIR modules pass the
loop-carried accumulator into `tt.dot`. Both PTX modules contain
`mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` with that carry.
The scaled suffix contains `cvt.rn.f32.s32`, `mul.f32`, and
`cvt.rzi.s32.f32`. Saved local evidence is:

| File under `/tmp/veritile-int-scaled-lowering-audit/` | SHA256 |
| --- | --- |
| `exact_historical_scaled0.ttir` | `3246437de8d2808927a84e8b7db49d4ad101530c796f0e6f1fffef798614b19b` |
| `exact_historical_scaled0.ptx` | `b90079dd083d60dd5148343171805a943d57a9e84e0a4d04c82e309cb4af1d6a` |
| `exact_historical_scaled1.ttir` | `b7eaf4f4f413e27d8e98a8168114ab11b04f8d3cbb0ebe246bdd601b27329d10` |
| `exact_historical_scaled1.ptx` | `ceb990b304ad49511a2d0e5b9fa83581e59efc941378244b149f9801f07a3a9f` |

The companion JSON files `/tmp/veritile-int-scaled-historical-hashes.json` and
`/tmp/veritile-int-scaled-historical-lowering-hashes.json` also record source,
TTIR, LLVM IR, and PTX hashes. No historical evidence tree was copied.

The current separate candidate uses zero-seeded tile dots followed by explicit
`add.u32`. `SaturatingTile.source_dot_eq` proves that the checked signed8 tile
cannot saturate under any instruction partition; `add_u32` proves the wrapping
carry update. The [current revision documentation](../../../bench/tritonbench_g/int_scaled_matmul/README.md)
records its independent validation. Acceptance of that repaired current version
is separate from this substantive historical applicability correction.
