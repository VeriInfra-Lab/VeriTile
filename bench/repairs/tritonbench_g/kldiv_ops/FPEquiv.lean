import bench.repairs.tritonbench_g.kldiv_ops.Proofs.GridPair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! Actual complete FP32 forward/backward programs. The existing candidate
commutes the signed output-row pointer product; every floating instruction,
runtime epsilon binding, branch, chunk, reduction and store is retained. -/
namespace VeriTile.Bench.Repairs.KldivOps.Kernel.FPEquiv
open VeriTile Triton FP.Structural Interface
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def forwardOriginal (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (eps : α) :=
  ProgramIO.Implementation.original (forwardIO M p flag red eps id id) (fun _ => True)
def forwardCommuted (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (eps : α) :=
  ProgramIO.Implementation.ofVariant (forwardCandidate M p flag red eps id id) (fun _ => True)
def backwardOriginal (M : Algebra α) (p : Parameters) (flag : Bool) :=
  ProgramIO.Implementation.original (backwardIO M p flag id id) (fun _ => True)
def backwardCommuted (M : Algebra α) (p : Parameters) (flag : Bool) :=
  ProgramIO.Implementation.ofVariant (backwardCandidate M p flag id id) (fun _ => True)

/-- Complete row grid and all reduction codes; both empty axes and all input
and epsilon payloads are retained, including nonfinite native FP32 words. -/
specification forward_pair (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (eps : α)
    (block : 0<p.block) (predictionSeparate : p.Y≠p.O) (targetSeparate : p.T≠p.O) :
    forwardOriginal M p flag red eps ≡[[]] forwardCommuted M p flag red eps :=
  ProgramIO.Implementation.of_equivalent (forwardCandidate M p flag red eps id id)
    (forward_equivalence M p flag red eps block predictionSeparate targetSeparate id id)

/-- Complete standalone prediction derivative grid and unchanged exterior frame. -/
specification backward_pair (M : Algebra α) (p : Parameters) (flag : Bool)
    (block : 0<p.block) (targetSeparate : p.T≠p.O) :
    backwardOriginal M p flag ≡[[]] backwardCommuted M p flag :=
  ProgramIO.Implementation.of_equivalent (backwardCandidate M p flag id id)
    (backward_equivalence M p flag block targetSeparate id id)

#axiomsClean forward_pair
#axiomsClean backward_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.FPEquiv
