import bench.repairs.tritonbench_g.destindex_copy_kv1.Kernels

/-! Both selected Python function bodies are exactly identical to kv1, including
the final return, and share its actual pasted source in
bench/repairs/tritonbench_g/destindex_copy_kv1/Kernels.lean.
Both repaired launches use that complete masked multihead source. The two
payload planes retain independent head counts and feature widths. -/
namespace VeriTile.Bench.Repairs.DestindexCopy
abbrev Config := DestindexCopyKv1.Config
abbrev Layout := DestindexCopyKv1.Layout
abbrev kernel := DestindexCopyKv1.kernel
abbrev indexCommutedKernel := DestindexCopyKv1.indexCommutedKernel
abbrev flat := DestindexCopyKv1.flat
abbrev block := DestindexCopyKv1.block
inductive Side where | nope | rope
  deriving DecidableEq, Repr
end VeriTile.Bench.Repairs.DestindexCopy
