import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.ContextAttnFwd.FPEquiv
open VeriTile.Triton FP.Structural NativeFormula
open scoped VeriTile.Spec
noncomputable section
abbrev original := @PairIO.original
abbrev blockEndFactored := @PairIO.blockEndFactored

/-- Full FP32 sources: factor the common prompt offset out of the integer
minimum, saving one repeated scalar add. All masks, finite sentinel, base-2
exponential, probability conversion and live dot seed remain unchanged.
Arbitrary raw arrays, aliases, repetitions and all native consumers are retained. -/
specification block_end_factored_equiv {α : Type} [Inhabited α]
    (p : Layout) (r : Regions) (M : Algebra α) (dots : FPSuccess.DotSupport p M)
    (scale : ℝ) (jobs : List Launch) (positive : 0<p.BN) :
    original p r M dots scale jobs ≡[[]] blockEndFactored p r M dots scale jobs :=
  PairEquivalence.equivalent p r M dots scale jobs positive
#axiomsClean block_end_factored_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnFwd.FPEquiv
