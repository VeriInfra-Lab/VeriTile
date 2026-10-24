import bench.tritonbench_g.swiglu_fwd.Proofs.WholeIndex
import bench.tritonbench_g.swiglu_fwd.Proofs.Execution
import bench.tritonbench_g.swiglu_fwd.Kernels
import bench.tritonbench_g.swiglu_fwd.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : Program where
  kernel := typedKernel input partner output v X Y OUT sx sy so N B
  domain := []

theorem decomposition (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody =
      sourcePrefix v X Y OUT sx sy so B ++
        (typedKernel input partner output Bool.false X Y OUT sx sy so N B).surfaceBody.drop 5 := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

/-- Both complete floating executions have equal state, memory and failure behavior;
only three natural row-address products commute, with all floating operations retained. -/
specification source_equivalence (input partner output : Storage)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    program input partner output Bool.false X Y OUT sx sy so N B ≡[assumptions]
      program input partner output Bool.true X Y OUT sx sy so N B := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> cases partner <;> cases output <;> rfl)
  intro α _ M D _ s _
  change run (engine M)
      (typedKernel input partner output Bool.false X Y OUT sx sy so N B).surfaceBody s =
    run (engine M)
      (typedKernel input partner output Bool.true X Y OUT sx sy so N B).surfaceBody s
  rw [decomposition input partner output Bool.false X Y OUT sx sy so N B,
    decomposition input partner output Bool.true X Y OUT sx sy so N B]
  exact prefix_run M s X Y OUT sx sy so B _

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.FPEquiv

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton FP.Structural
noncomputable section

/-- Thirty-two-bit payload carrier. Primitive implementations remain explicit;
there is no associativity, exact sigmoid, identity-cast, or finite-only law. -/
abbrev FP32Word := BitVec 32

/-- Original whole-entry finite-word result, with the source's FP32 load
casts, sigmoid and left-associated multiplications. No source/candidate
rewrite is used to infer original values, and no BF16 route is selected. -/
specification swiglu_fwd_fp32_source_outcomes (M : Algebra FP32Word)
    (rt : Runtime FP32Word input supplied entry preexisting)
    (_inputValid : InputDomain input)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p)
    (finite : ∀ p (hp : rt.prepare = .ok p) (hc : checks p = Bool.true),
      FiniteAdmission p (valid p hp hc)) :
    HostSpec (scalar M) rt (executeHost M rt) ∧
      ∀ p hp hc, IndexCorrespondence p (valid p hp hc) (finite p hp hc) := by
  exact ⟨host_outcomes M rt valid,fun p hp hc => index_correspondence p (valid p hp hc) (finite p hp hc)⟩

#axiomsClean swiglu_fwd_fp32_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
