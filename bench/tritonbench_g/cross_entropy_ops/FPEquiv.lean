import bench.tritonbench_g.cross_entropy_ops.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev forwardOriginal := @PairIO.forwardOriginal
abbrev forwardIndexCommuted := @PairIO.forwardIndexCommuted
abbrev backwardOriginal := @PairIO.backwardOriginal
abbrev backwardIndexCommuted := @PairIO.backwardIndexCommuted

/-- Complete repaired FP32 forward programs, actual runtime scalar binding,
signed labels, all active output cells and the split z-loss frame, successful safe execution and exterior frame.
Only the exact integer column product is commuted. -/
specification forward_index_commuted_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) (positive : 0<p.block)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (zLse : p.zloss≠p.lse) (zLoss : p.zloss≠p.loss) :
    forwardOriginal M p row block args ≡[[]]
      forwardIndexCommuted M p row block args :=
  PairEquivalence.forward M p row block args positive separate outputs zLse zLoss

/-- Complete repaired backward programs on the original unrestricted payload
domain, including ignored labels, in-place storage and all active output lanes. -/
specification backward_index_commuted_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (row block : Nat)
    (args : RuntimeArgs.Values α) :
    backwardOriginal M p row block args ≡[[]]
      backwardIndexCommuted M p row block args :=
  PairEquivalence.backward M p row block args

#axiomsClean forward_index_commuted_equiv
#axiomsClean backward_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv
