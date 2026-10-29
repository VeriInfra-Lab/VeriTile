import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole

/-! The literal Python programs have signed casts, local scalar assignments and
the original loop variable. Their ASTs are not equal to the former shortened
natural-index transcriptions. These connections instead check successful
execution of the actual programs and their complete memory observations.
The independent mathematical formulas are supplied by Logical and its proofs. -/
namespace VeriTile.Bench.Repairs.LayerNormLiger.Shape
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section

theorem forward_execution {α : Type} [Inhabited α]
    (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (s : State α) :
    ∃ t, exec (withSum M R) (Forward.kernel p) s = some t ∧
      t.mem = (Forward.reference M R p s).mem :=
  Forward.complete_run M R p s

theorem backward_execution {α : Type} [Inhabited α]
    (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (s : State α) :
    ∃ t, exec (withSum M R) (Backward.kernel p) s = some t ∧
      t.mem = (Backward.reference M R p s).mem :=
  Backward.complete_run M R p s

#axiomsClean forward_execution
#axiomsClean backward_execution
end
end VeriTile.Bench.Repairs.LayerNormLiger.Shape
