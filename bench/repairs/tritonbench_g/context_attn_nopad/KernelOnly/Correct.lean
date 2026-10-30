import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- FP32 source interpreted over Real values. Five logical input channels,
strided Q/K/V/O windows, fixed signed metadata, and the entire independent
ordinary grid are explicit. There is no private scratch allocation. -/
abbrev attentionIO (c : Call) (r : Regions) := LogicalData.attentionIO c r

/-- Independent dot-product score with the exact original binary64 host scale. -/
def attentionScore (c : Call) (q k : RealData.Coordinate c → ℝ) (b h t j : Nat) : ℝ :=
  (∑ d : Fin c.q.dims,
    RealData.sample c q b h t d.val * RealData.sample c k b h j d.val)*c.scale

/-- Self-inclusive causal softmax: exactly keys 0 through the query time. -/
def causalAttention (c : Call) (q k v : RealData.Coordinate c → ℝ) (j : RealData.Coordinate c) : ℝ :=
  (∑ t ∈ Finset.range (j.2.2.1.val+1),
    Real.exp (attentionScore c q k j.1.val j.2.1.val j.2.2.1.val t) *
      RealData.sample c v j.1.val j.2.1.val t j.2.2.2.1.val) /
  ∑ t ∈ Finset.range (j.2.2.1.val+1),
    Real.exp (attentionScore c q k j.1.val j.2.1.val j.2.2.1.val t)

def attentionExpected (c : Call) (xs : LogicalData.Inputs c) :
    RealData.Coordinate c → ℝ :=
  causalAttention c xs.query xs.key xs.value

/-- The complete-grid condition makes the active finite output window cover
every logical sequence element. Shared IO includes successful ordinary-grid
execution, actual source safety from window bounds, outputs and full frame. -/
specification correct (c : Call) (r : Regions) (shape : Host.Shape c)
    (labels : RealData.Labels r) (complete : c.completeGrid) :
    Spec.Real ((attentionIO c r) ⊨{LogicalData.Metadata c} (attentionExpected c)) :=
  LogicalData.logical_correct c r shape labels

#axiomsClean correct

#specNonCircular attentionExpected avoiding [kernel,VeriTile.Triton.exec]
#stmtSurfaceSubset correct ⊆
  [VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.Call,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData.Inputs,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.LogicalData.Metadata,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData.Coordinate,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData.Labels,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData.Values,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData.io,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData.laneCount,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.Regions,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.attentionExpected,
   VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.attentionIO,
   VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Call.completeGrid,
   VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Shape,
   VeriTile.Spec.Real,
   VeriTile.Triton.AlgorithmProgramIO.ImplementsOn,
   VeriTile.Triton.LogicalDomainIO.ImplementsOn]

end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
