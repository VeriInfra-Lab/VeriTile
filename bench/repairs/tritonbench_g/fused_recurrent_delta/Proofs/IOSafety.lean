import bench.repairs.tritonbench_g.fused_recurrent_delta.IO
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.SourceSafetyForwardComplete
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.SourceSafetyBackwardComplete
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.IOSafety
open VeriTile.Triton FP.Structural KernelProgram KernelRegisters KernelBackwardRegisters Interface
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem job_pids (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags)
    (scale : ℝ) (bh iv : Nat) (u : State α) :
    ((job M kind s f scale bh iv).initial u).pids 0=iv ∧
    ((job M kind s f scale bh iv).initial u).pids 1=0 ∧
    ((job M kind s f scale bh iv).initial u).pids 2=bh := by
  simp [job,Invocation.initial,bindRegisters]

theorem forward_bounds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : ForwardInputs α s) (bounds : RegionName → Nat)
    (hb : (forwardIO M s f scale bh iv).InBounds bounds vals) :
    SourceSafetyForward.Bounds s f bh iv bounds := by
  rcases hb with ⟨_,hin,hout,_⟩
  constructor
  · intro t ht i hi
    exact hin (0 : Fin 6) (⟨t,ht⟩,i,()) hi
  · intro t ht i hi
    exact hin (1 : Fin 6) (⟨t,ht⟩,i,()) hi
  · intro t ht j hj
    exact hin (2 : Fin 6) (⟨t,ht⟩,j,()) hj
  · intro flag t ht j hj
    exact hin (3 : Fin 6) (⟨t,ht⟩,j,()) ⟨flag,hj⟩
  · intro flag t ht
    exact hin (4 : Fin 6) (⟨t,ht⟩,()) flag
  · intro t ht j hj
    exact hout (0 : Fin 3) (⟨t,ht⟩,j,()) hj
  · intro flag i j hi hj
    exact hin (5 : Fin 6) (i,j,()) ⟨flag,hi,hj⟩
  · intro flag i j hi hj
    exact hout (2 : Fin 3) (i,j,()) ⟨flag,hi,hj⟩

theorem forward_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : ForwardInputs α s) (bounds : RegionName → Nat)
    (hb : (forwardIO M s f scale bh iv).InBounds bounds vals) (u : State α) :
    ScheduleSafe bounds ((forwardIO M s f scale bh iv).schedule vals) u := by
  change ScheduleSafe bounds [job M .forward s f scale bh iv] u
  rw [scheduleSafe_singleton]
  have pid := job_pids M .forward s f scale bh iv u
  exact SourceSafetyForward.whole M s f scale bh iv _ pid.1 pid.2.1 pid.2.2 bounds
    (forward_bounds M s f scale bh iv vals bounds hb)

theorem backward_bounds (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : BackwardInputs α s) (bounds : RegionName → Nat)
    (hb : (backwardIO M s f scale bh iv).InBounds bounds vals) :
    SourceSafetyBackward.Bounds s f bh iv bounds := by
  rcases hb with ⟨_,hin,hout,_⟩
  constructor
  · intro t ht i hi
    exact hin (0 : Fin 8) (⟨t,ht⟩,i,()) hi
  · intro t ht i hi
    exact hin (1 : Fin 8) (⟨t,ht⟩,i,()) hi
  · intro t ht j hj
    exact hin (2 : Fin 8) (⟨t,ht⟩,j,()) hj
  · intro t ht j hj
    exact hin (5 : Fin 8) (⟨t,ht⟩,j,()) hj
  · intro flag t ht j hj
    exact hin (3 : Fin 8) (⟨t,ht⟩,j,()) ⟨flag,hj⟩
  · intro flag t ht
    exact hin (4 : Fin 8) (⟨t,ht⟩,()) flag
  · intro t ht i hi
    exact hout (0 : Fin 6) (⟨t,ht⟩,i,()) hi
  · intro t ht i hi
    exact hout (1 : Fin 6) (⟨t,ht⟩,i,()) hi
  · intro t ht j hj
    exact hout (2 : Fin 6) (⟨t,ht⟩,j,()) hj
  · intro flag t ht j hj
    exact hout (3 : Fin 6) (⟨t,ht⟩,j,()) ⟨flag,hj⟩
  · intro flag t ht
    exact hout (4 : Fin 6) (⟨t,ht⟩,()) flag
  · intro flag i j hi hj
    exact hin (6 : Fin 8) (i,j,()) ⟨flag,hi,hj⟩
  · intro flag i j hi hj
    exact hin (7 : Fin 8) (i,j,()) ⟨flag,hi,hj⟩
  · intro flag i j hi hj
    exact hout (5 : Fin 6) (i,j,()) ⟨flag,hi,hj⟩

theorem backward_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (vals : BackwardInputs α s) (bounds : RegionName → Nat)
    (hb : (backwardIO M s f scale bh iv).InBounds bounds vals) (u : State α) :
    ScheduleSafe bounds ((backwardIO M s f scale bh iv).schedule vals) u := by
  change ScheduleSafe bounds [job M .backward s f scale bh iv] u
  rw [scheduleSafe_singleton]
  have pid := job_pids M .backward s f scale bh iv u
  exact SourceSafetyBackward.whole M s f scale bh iv _ pid.1 pid.2.1 pid.2.2 bounds
    (backward_bounds M s f scale bh iv vals bounds hb)

#axiomsClean forward_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.IOSafety
