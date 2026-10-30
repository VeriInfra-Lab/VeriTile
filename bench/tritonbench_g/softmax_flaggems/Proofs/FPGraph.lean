import bench.tritonbench_g.softmax_flaggems.Proofs.FPOneTile
import bench.tritonbench_g.softmax_flaggems.Proofs.FPBackwardInner
import bench.tritonbench_g.softmax_flaggems.Proofs.FPBackwardNoninner
import bench.tritonbench_g.softmax_flaggems.Proofs.FPForwardInner
import bench.tritonbench_g.softmax_flaggems.Proofs.FPForwardNoninner
import bench.tritonbench_g.softmax_flaggems.Proofs.FPGraphMemory

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGraph
open VeriTile Triton FP.Structural FPReference FPLoop FPGraphMemory
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem backward_inner_emit_congr (M : Algebra α) (P G Y : RegionName)
    (row rows N B C cur : Nat) (scale : Values α .real [C])
    (s t : State α) (h : s.mem=t.mem) :
    (FPBackwardInner.emit M P G Y row rows N B C scale cur s).mem =
    (FPBackwardInner.emit M P G Y row rows N B C scale cur t).mem := by
  unfold FPBackwardInner.emit
  apply scatter_congr _ _ _ _ _ _ _ h
  funext i
  simp only [FPBackwardInner.storedValue,FPBackwardInner.load,FPReference.read,h]

theorem backward_inner_congr (M : Algebra α) (P G Y : RegionName)
    (row rows N B C : Nat) (s t : State α) (h : s.mem=t.mem) :
    (FPBackwardInner.graph M s P G Y row rows N B C).mem =
    (FPBackwardInner.graph M t P G Y row rows N B C).mem := by
  have ha : FPBackwardInner.accumulate M s P G row rows N B C =
      FPBackwardInner.accumulate M t P G row rows N B C := by
    funext cur scale i
    simp only [FPBackwardInner.accumulate,FPBackwardInner.load,FPReference.read,h]
  unfold FPBackwardInner.graph
  rw [ha]
  exact walk_rel _ _ (fun s t : State α => s.mem=t.mem)
    (fun cur s t h => backward_inner_emit_congr M P G Y row rows N B C cur _ s t h) _ _ _ s t h

theorem backward_noninner_emit_congr (M : Algebra α) (P G Y : RegionName)
    (row col rows N K B C cur : Nat) (scale : Values α .real [C])
    (s t : State α) (h : s.mem=t.mem) :
    (FPBackwardNoninner.emit M P G Y row col rows N K B C scale cur s).mem =
    (FPBackwardNoninner.emit M P G Y row col rows N K B C scale cur t).mem := by
  unfold FPBackwardNoninner.emit
  apply scatter_congr _ _ _ _ _ _ _ h
  funext i
  simp only [FPBackwardNoninner.storedValue,FPBackwardNoninner.load,FPReference.read,h]

theorem backward_noninner_congr (M : Algebra α) (P G Y : RegionName)
    (row col rows N K B C : Nat) (s t : State α) (h : s.mem=t.mem) :
    (FPBackwardNoninner.graph M s P G Y row col rows N K B C).mem =
    (FPBackwardNoninner.graph M t P G Y row col rows N K B C).mem := by
  have ha : FPBackwardNoninner.accumulate M s P G row col rows N K B C =
      FPBackwardNoninner.accumulate M t P G row col rows N K B C := by
    funext cur scale i
    simp only [FPBackwardNoninner.accumulate,FPBackwardNoninner.load,FPReference.read,h]
  unfold FPBackwardNoninner.graph
  rw [ha]
  exact walk_rel _ _ (fun s t : State α => s.mem=t.mem)
    (fun cur s t h => backward_noninner_emit_congr M P G Y row col rows N K B C cur _ s t h) _ _ _ s t h

theorem forward_noninner_emit_congr (M : Algebra α) (X Y : RegionName)
    (row col N K B C cur : Nat) (mx z : Values α .real [C])
    (s t : State α) (h : s.mem=t.mem) :
    (FPForwardNoninner.emit M X Y row col N K B C mx z cur s).mem =
    (FPForwardNoninner.emit M X Y row col N K B C mx z cur t).mem := by
  unfold FPForwardNoninner.emit
  apply scatter_congr _ _ _ _ _ _ _ h
  funext i
  simp only [FPForwardNoninner.storedValue,FPForwardNoninner.load,FPReference.read,h]

theorem forward_noninner_congr (M : Algebra α) (X Y : RegionName)
    (row col N K B C : Nat) (s t : State α) (h : s.mem=t.mem) :
    (FPForwardNoninner.graph M s X Y row col N K B C).mem =
    (FPForwardNoninner.graph M t X Y row col N K B C).mem := by
  have ha : FPForwardNoninner.accumulate M s X row col N K B C =
      FPForwardNoninner.accumulate M t X row col N K B C := by
    funext cur v
    apply Prod.ext <;> funext i
    all_goals simp only [FPForwardNoninner.accumulate,FPForwardNoninner.load,FPReference.read,h]
  unfold FPForwardNoninner.graph
  rw [ha]
  exact walk_rel _ _ (fun s t : State α => s.mem=t.mem)
    (fun cur s t h => forward_noninner_emit_congr M X Y row col N K B C cur _ _ s t h) _ _ _ s t h

theorem forward_inner_emit_congr (M : Algebra α) (X Y : RegionName) (masked : Bool)
    (row N B cur : Nat) (mx z : α) (s t : State α) (h : s.mem=t.mem) :
    (FPForwardInner.emit M X Y masked row N B mx z cur s).mem =
    (FPForwardInner.emit M X Y masked row N B mx z cur t).mem := by
  unfold FPForwardInner.emit
  apply scatter_congr _ _ _ _ _ _ _ h
  funext i
  simp only [FPForwardInner.storedValue,FPForwardInner.load,FPReference.read,h]

theorem forward_inner_congr (M : Algebra α) (X Y : RegionName)
    (row N B : Nat) (s t : State α) (h : s.mem=t.mem) :
    (FPForwardInner.graph M s X Y row N B).mem =
    (FPForwardInner.graph M t X Y row N B).mem := by
  have ha (masked : Bool) : FPForwardInner.accumulate M s X masked row N B =
      FPForwardInner.accumulate M t X masked row N B := by
    funext cur v
    apply Prod.ext <;> funext i
    all_goals simp only [FPForwardInner.accumulate,FPForwardInner.load,FPReference.read,h]
  unfold FPForwardInner.graph
  rw [ha Bool.false,ha Bool.true]
  apply walk_rel _ _ (fun s t : State α => s.mem=t.mem)
    (fun cur s t h => forward_inner_emit_congr M X Y Bool.false row N B cur _ _ s t h)
  exact walk_rel _ _ (fun s t : State α => s.mem=t.mem)
    (fun cur s t h => forward_inner_emit_congr M X Y Bool.true row N B cur _ _ s t h) _ _ _ s t h

#axiomsClean backward_inner_congr
#axiomsClean backward_noninner_congr
#axiomsClean forward_inner_congr
#axiomsClean forward_noninner_congr
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPGraph
