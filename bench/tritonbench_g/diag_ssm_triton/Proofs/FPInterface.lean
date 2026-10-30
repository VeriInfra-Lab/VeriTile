import bench.tritonbench_g.diag_ssm_triton.Proofs.FPExecution
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_FP
import VeriTile.Triton.Float.ProfiledSoundness

/-! Successful full-memory equivalence for the existing index-commuted
candidates, on the same canonical typed DSL used by Correct. No floating
operation is changed: FP32 real arithmetic and both complex64 components,
conjugation signs, loops, branches, masks and stores are retained. The only
rewrite is pid*block = block*pid on the discrete column index. Hence no
floating numerical experiment or finite-operand premise is required. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.FPInterface
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
open FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section
set_option autoImplicit false

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardOriginal (s x transition y : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.forwardTyped s x transition y length batch dim block, []⟩

def forwardCandidate (s x transition y : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.forwardIndexCommuted s x transition y length batch dim block, []⟩

/-- Both complete typed sources succeed and agree on every memory cell.
The discrete index rewrite requires no numerical law or lane restriction. -/
theorem forward_equiv (s x transition y : RegionName) (length batch dim block : Nat) :
    forwardOriginal s x transition y length batch dim block ≡[assumptions] forwardCandidate s x transition y length batch dim block := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  obtain ⟨v, hv⟩ := FPExecution.forward_success (FP.ProfiledRewrite.engine M) s x transition y length batch dim block u
  have equal := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent .fp32 s x transition y length batch dim block) α M D hM u
    (by simp [FPProof.forwardProgram])
  exact ⟨v, v, hv, equal.symm.trans hv, rfl⟩

#print_fp_assumptions forward_equiv

def backwardOriginal (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.backwardTyped s transition y gs gx gl gy length batch dim block, []⟩

def backwardCandidate (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.backwardIndexCommuted s transition y gs gx gl gy length batch dim block, []⟩

/-- Both complete typed sources succeed and agree on every memory cell.
The discrete index rewrite requires no numerical law or lane restriction. -/
theorem backward_equiv (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    backwardOriginal s transition y gs gx gl gy length batch dim block ≡[assumptions] backwardCandidate s transition y gs gx gl gy length batch dim block := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  obtain ⟨v, hv⟩ := FPExecution.backward_success (FP.ProfiledRewrite.engine M) s transition y gs gx gl gy length batch dim block u
  have equal := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent .fp32 s transition y gs gx gl gy length batch dim block) α M D hM u
    (by simp [FPProof.backwardProgram])
  exact ⟨v, v, hv, equal.symm.trans hv, rfl⟩

#print_fp_assumptions backward_equiv

def complexForwardOriginal (s x y transition : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.complexForwardTyped s x y transition length batch dim block, []⟩

def complexForwardCandidate (s x y transition : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.complexForwardIndexCommuted s x y transition length batch dim block, []⟩

/-- Both complete typed sources succeed and agree on every memory cell.
The discrete index rewrite requires no numerical law or lane restriction. -/
theorem complexForward_equiv (s x y transition : RegionName) (length batch dim block : Nat) :
    complexForwardOriginal s x y transition length batch dim block ≡[assumptions] complexForwardCandidate s x y transition length batch dim block := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  obtain ⟨v, hv⟩ := FPExecution.complexForward_success (FP.ProfiledRewrite.engine M) s x y transition length batch dim block u
  have equal := FP.ProfiledRewrite.sound
    (FPProof.complexForward_equivalent s x y transition length batch dim block) α M D hM u
    (by simp [FPProof.complexForwardProgram])
  exact ⟨v, v, hv, equal.symm.trans hv, rfl⟩

#print_fp_assumptions complexForward_equiv

def complexBackwardOriginal (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.complexBackwardTyped s transition y gs gx gl gy length batch dim block, []⟩

def complexBackwardCandidate (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) : Program :=
  ⟨Kernels.complexBackwardIndexCommuted s transition y gs gx gl gy length batch dim block, []⟩

/-- Both complete typed sources succeed and agree on every memory cell.
The discrete index rewrite requires no numerical law or lane restriction. -/
theorem complexBackward_equiv (s transition y gs gx gl gy : RegionName) (length batch dim block : Nat) :
    complexBackwardOriginal s transition y gs gx gl gy length batch dim block ≡[assumptions] complexBackwardCandidate s transition y gs gx gl gy length batch dim block := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D hM u _
  obtain ⟨v, hv⟩ := FPExecution.complexBackward_success (FP.ProfiledRewrite.engine M) s transition y gs gx gl gy length batch dim block u
  have equal := FP.ProfiledRewrite.sound
    (FPProof.complexBackward_equivalent s transition y gs gx gl gy length batch dim block) α M D hM u
    (by simp [FPProof.complexBackwardProgram])
  exact ⟨v, v, hv, equal.symm.trans hv, rfl⟩

#print_fp_assumptions complexBackward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.FPInterface
