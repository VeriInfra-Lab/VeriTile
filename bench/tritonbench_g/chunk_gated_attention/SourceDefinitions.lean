import bench.tritonbench_g.chunk_gated_attention.Contract
import bench.tritonbench_g.chunk_gated_attention.Sources.Kernels

/-! Both semantics use the existing authoritative readable Triton DSL.
No parallel handwritten source AST, dtype variant, or Python repair is added. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton

abbrev innerKernel (r : Admission.Inner) (base : Nat) : ComputeKernel :=
  let c := r.config
  NativeSource.hOriginal .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal
abbrev prefixKernel (r : Admission.Prefix) (BS base : Nat) : ComputeKernel :=
  let c := r.config BS
  NativeSource.cumOriginal .fp32 .fp32 "s" "o"
    (c.PT*c.PS) c.PS 1 c.PT c.PS c.BT c.BS c.heads base

/-- Existing carry-sum candidate, with exactly the native entry's FP32
specialization and literal region roles. -/
def innerCandidate (c : HostFP.Inner) (base : Nat) : ComputeKernel :=
  NativeSource.hSumLeft .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal

/-- Existing integer-address candidate, including the repaired decoded-grid
and batch-exhaustion guard. -/
def prefixCandidate (c : HostFP.Cumulative) (base : Nat) : ComputeKernel :=
  NativeSource.cumIndexFirst .fp32 .fp32 "s" "o"
    (c.PT*c.PS) c.PS 1 c.PT c.PS c.BT c.BS c.heads base

end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
