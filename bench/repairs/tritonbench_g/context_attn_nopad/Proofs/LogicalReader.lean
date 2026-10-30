import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.IO
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.PublicRealResult

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- Exact equality of complete judgments, without discarding inactive inputs. -/
theorem logical_iff (c : Call) (r : Regions) :
    (attentionIO c r).ImplementsOn (Metadata c)
      (fun xs j => RealData.attention c xs.query xs.key xs.value j) ↔
      (RealData.io c r).ImplementsOn (RealData.domain c) (RealData.expected c) :=
  Iff.rfl

theorem logical_correct (c : Call) (r : Regions) (shape : Host.Shape c)
    (labels : RealData.Labels r) :
    (attentionIO c r).ImplementsOn (Metadata c)
      (fun xs j => RealData.attention c xs.query xs.key xs.value j) :=
  (logical_iff c r).mpr (RealData.complete c r shape labels)

end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData
