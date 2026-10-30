import bench.tritonbench_g.geglu_tanh_triton.Proofs.RawNative

namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile.Triton FP.Structural FP.ProfiledMemory Native
open scoped VeriTile.Spec
noncomputable section

/-- The actual candidate changes only a signed integer address product. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (backward : Bool) (count : Nat) : Program := ⟨source backward 128 count,[]⟩
def indexCommuted (backward : Bool) (count : Nat) : Program :=
  ⟨indexCommutedSource backward 128 count,[]⟩

/-- Retained unrestricted typed-memory equivalence of the two actual sources.
The complete floating suffix is identical; no floating rewrite is assumed. -/
specification geglu_index_commuted_equiv (backward : Bool) (count : Nat) :
    original backward count ≡[assumptions] indexCommuted backward count := by
  simpa only [original,indexCommuted,assumptions,RawNative.original,RawNative.indexCommuted,
    RawNative.assumptions] using RawNative.geglu_index_commuted_equiv backward count


#print_fp_assumptions geglu_index_commuted_equiv
#stmtSurfaceSubset geglu_index_commuted_equiv ⊆
  [Spec.Equiv,Program,instProgramSyntaxProgram,original,indexCommuted,assumptions]
#axiomsClean geglu_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton
