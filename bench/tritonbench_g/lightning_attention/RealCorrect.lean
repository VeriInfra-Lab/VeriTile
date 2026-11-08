import bench.tritonbench_g.lightning_attention.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent causal attention over the complete logical sequence. -/
def causalFormula (d : Host.Dimensions) (x : LogicalIO.Inputs d)
    (h : Fin (d.B*d.H)) (t : Fin d.N) (c : Fin d.E) : ℝ :=
  ∑ u : Fin d.N, if u≤t then
    (∑ a : Fin d.D, x.query h t a*x.key h u a)*x.value h u c
  else 0

/-- Three named coordinate VJP tensors. -/
def vjpFormula (d : Host.Dimensions) (x : LogicalIO.Inputs d) : LogicalIO.Outputs d := ⟨
  fun h t a => ∑ u : Fin d.N, if u≤t then
    (∑ c : Fin d.E, x.incoming h t c*x.value h u c)*x.key h u a else 0,
  fun h u a => ∑ t : Fin d.N, if u≤t then
    x.query h t a*(∑ c : Fin d.E, x.incoming h t c*x.value h u c) else 0,
  fun h u c => ∑ t : Fin d.N, if u≤t then
    (∑ a : Fin d.D, x.query h t a*x.key h u a)*x.incoming h t c else 0⟩

/-- The complete original forward grid implements independent causal attention.
The shared judgment includes actual source access safety, success, typed outputs
and the exact exterior frame. Source precision stays FP32; this is Real math. -/
specification forward_correct (d : Host.Dimensions) (geometry : Host.Contract d) :
    Spec.Real (LogicalIO.forward d ⊨ causalFormula d) :=
  LogicalIO.forward_implements d geometry

/-- Both original backward grids implement the independent VJP. The inter grid
consumes the intra grid's actual gradients. All three outputs are coordinate
derivatives of the independent scalar loss, under the unchanged geometry. -/
specification backward_correct (d : Host.Dimensions) (geometry : Host.Contract d) :
    Spec.Real ((LogicalIO.io d ⊨ vjpFormula d) ∧ LogicalDerivatives d) :=
  ⟨LogicalIO.backward_implements d geometry,logical_derivatives d⟩

#specNonCircular causalFormula avoiding [FP.Structural.exec,FP.Structural.runSchedule]
#specNonCircular vjpFormula avoiding [FP.Structural.exec,FP.Structural.runSchedule]
#axiomsClean forward_correct
#axiomsClean backward_correct
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
