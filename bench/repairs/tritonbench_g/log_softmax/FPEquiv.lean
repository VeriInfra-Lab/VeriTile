import bench.repairs.tritonbench_g.log_softmax.Proofs.IndexEquivalence

namespace VeriTile.Bench.Repairs.LogSoftmax.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

/-- Only integer index multiplication changes; no floating rewrite law. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardOriginal (O X : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.forward O X M N K BM C,[]⟩
def forwardCommuted (O X : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.forwardCommuted O X M N K BM C,[]⟩
def backwardOriginal (Y G O : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.backward Y G O M N K BM C,[]⟩
def backwardCommuted (Y G O : RegionName) (M N K BM C : Nat) : Program :=
  ⟨Source.backwardCommuted Y G O M N K BM C,[]⟩

/-- Retains the general one-program equivalence with arbitrary named regions. -/
specification forward_index_equiv (O X : RegionName) (M N K BM C : Nat) (hc : 0<C) :
    forwardOriginal O X M N K BM C ≡[assumptions] forwardCommuted O X M N K BM C :=
  IndexEquivalence.forward_index_equiv O X M N K BM C hc

specification backward_index_equiv (Y G O : RegionName) (M N K BM C : Nat) :
    backwardOriginal Y G O M N K BM C ≡[assumptions] backwardCommuted Y G O M N K BM C :=
  IndexEquivalence.backward_index_equiv Y G O M N K BM C


#print_fp_assumptions forward_index_equiv
#print_fp_assumptions backward_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.FPEquiv
