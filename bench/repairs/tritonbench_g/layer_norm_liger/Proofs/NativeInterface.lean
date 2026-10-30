import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan
import VeriTile.Triton.Float.ObservedProgramIO

namespace VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Native interpretation retains arbitrary symbolic input expressions,
primitive bindings, reduction trees and the selected local contraction plan. -/
def nativeForwardIO (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Forward.Parameters) (pid : Nat) :=
  forwardIO (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) p pid

def nativeBackwardIO (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Backward.Parameters) (pid : Nat) :=
  backwardIO (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) p pid

def forwardReference (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Forward.Parameters) (pid : Nat) :=
  Ordered.forward (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) p pid

def backwardReference (contraction : NativePlan.Contraction) (trees : NativePlan.Trees)
    (p : Backward.Parameters) (pid : Nat) :=
  Ordered.backward (NativePlan.algebra contraction trees) (NativePlan.sums contraction trees) p pid

/-- Evaluate logical output expressions under the explicit native primitives.
This observer does not restrict input expressions to words or finite values. -/
def forwardNative (P : NativePlan.Primitives) (contraction : NativePlan.Contraction)
    (trees : NativePlan.Trees) (p : Forward.Parameters) (pid : Nat) :=
  fun v => (nativeForwardIO contraction trees p pid).mapOutputs (fun _ _ => NativePlan.Expr.eval P)
    (forwardReference contraction trees p pid v)
def backwardNative (P : NativePlan.Primitives) (contraction : NativePlan.Contraction)
    (trees : NativePlan.Trees) (p : Backward.Parameters) (pid : Nat) :=
  fun v => (nativeBackwardIO contraction trees p pid).mapOutputs (fun _ _ => NativePlan.Expr.eval P)
    (backwardReference contraction trees p pid v)

theorem forward_native (P : NativePlan.Primitives) (contraction : NativePlan.Contraction)
    (trees : NativePlan.Trees) (p : Forward.Parameters) (pid : Nat) :
    (nativeForwardIO contraction trees p pid).Implements (forwardReference contraction trees p pid) ∧
    (nativeForwardIO contraction trees p pid).Observes (fun _ _ => NativePlan.Expr.eval P)
      (forwardNative P contraction trees p pid) := by
  have implementation := Ordered.forward_implements (NativePlan.algebra contraction trees)
    (NativePlan.sums contraction trees) p pid
  exact ⟨implementation,ProgramIO.Observes.of_implements implementation _⟩

theorem backward_native (P : NativePlan.Primitives) (contraction : NativePlan.Contraction)
    (trees : NativePlan.Trees) (p : Backward.Parameters) (pid : Nat) :
    (nativeBackwardIO contraction trees p pid).Implements (backwardReference contraction trees p pid) ∧
    (nativeBackwardIO contraction trees p pid).Observes (fun _ _ => NativePlan.Expr.eval P)
      (backwardNative P contraction trees p pid) := by
  have implementation := Ordered.backward_implements (NativePlan.algebra contraction trees)
    (NativePlan.sums contraction trees) p pid
  exact ⟨implementation,ProgramIO.Observes.of_implements implementation _⟩

#axiomsClean forward_native
#axiomsClean backward_native
end
end VeriTile.Bench.Repairs.LayerNormLiger.LayerIO
