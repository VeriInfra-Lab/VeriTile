import bench.tritonbench_g.swiglu_backward.Proofs.Observed
import bench.tritonbench_g.swiglu_backward.Proofs.WholeIndex
import bench.tritonbench_g.swiglu_backward.Proofs.Precision
import bench.tritonbench_g.swiglu_backward.Proofs.FP
import bench.tritonbench_g.swiglu_backward.Proofs.Execution
namespace VeriTile.Bench.TritonBenchG.SwigluBackward.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural
open FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) : Program where
  kernel := typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R
  domain := []

theorem decomposition (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody =
      sourcePrefix v X Y DOUT OUT DX DY sx sy sd so sdx sdy B R ++
        (typedKernel input upstream gradient output Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody.drop 9 := by
  cases v <;> rfl

/-- Complete source equivalence includes both flag outcomes, ordered stores,
all buffer aliases, masks and failure. Only natural row-address products commute;
no floating operation changes and no numerical assumptions are required. -/
specification source_equivalence (input upstream gradient output : Storage)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    program input upstream gradient output Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R ≡[assumptions]
      program input upstream gradient output Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by
      change (((typedKernel input upstream gradient output Bool.false X Y DOUT OUT DX DY
        sx sy sd so sdx sdy N B R).inputs,
        (typedKernel input upstream gradient output Bool.false X Y DOUT OUT DX DY
          sx sy sd so sdx sdy N B R).outputs),
        (program input upstream gradient output Bool.false X Y DOUT OUT DX DY
          sx sy sd so sdx sdy N B R).domain) =
        (((typedKernel input upstream gradient output Bool.true X Y DOUT OUT DX DY
          sx sy sd so sdx sdy N B R).inputs,
          (typedKernel input upstream gradient output Bool.true X Y DOUT OUT DX DY
            sx sy sd so sdx sdy N B R).outputs),
          (program input upstream gradient output Bool.true X Y DOUT OUT DX DY
            sx sy sd so sdx sdy N B R).domain)
      simp only [Precision.source_shape input upstream gradient output Bool.false
        X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R |>.2.1,
        Precision.source_shape input upstream gradient output Bool.false
          X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R |>.2.2,
        Precision.source_shape input upstream gradient output Bool.true
          X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R |>.2.1,
        Precision.source_shape input upstream gradient output Bool.true
          X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R |>.2.2]
      rfl)
  intro α _ M D _ s _
  change run (engine M)
      (typedKernel input upstream gradient output Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody s =
    run (engine M)
      (typedKernel input upstream gradient output Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody s
  rw [decomposition input upstream gradient output Bool.false X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R,
    decomposition input upstream gradient output Bool.true X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R]
  exact prefix_run M s X Y DOUT OUT DX DY sx sy sd so sdx sdy B R _

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.FPEquiv

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Observed
open VeriTile.Triton FP.Structural
open Whole
noncomputable section

/-- Whole original source interpreted as its pinned compiler-observed FP32
operation graph, including actual fused affine recurrence and operand order.
All pending numerical ports are scalar primitives; the full source, host,
trials, logical returns, entry frame and byte spans are already proved. -/
specification swiglu_backward_fp32_source_outcomes (M : NativeModel (BitVec 32))
    (rt : Whole.Runtime (Term (BitVec 32)) call (entryGraph entry) preexisting)
    (_inputValid : InputDomain call)
    (valid : ∀ p, rt.prepare = .ok p → checks p = Bool.true → Domain p)
    (finite : ∀ p (hp : rt.prepare = .ok p) (hc : checks p = Bool.true),
      FiniteAdmission p (valid p hp hc)) :
    HostSpec M rt (executeHost M rt) ∧
      ∀ p hp hc, IndexCorrespondence p (valid p hp hc) (finite p hp hc) := by
  exact ⟨host_outcomes M rt valid,
    fun p hp hc => index_correspondence p (valid p hp hc) (finite p hp hc)⟩

#axiomsClean swiglu_backward_fp32_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluBackward.Observed
