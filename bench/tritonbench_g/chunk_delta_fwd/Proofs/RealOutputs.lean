import bench.tritonbench_g.chunk_delta_fwd.Proofs.GridOutputs
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealValues
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
noncomputable section
set_option maxHeartbeats 6000000

theorem history_value (p : Parameters) (h : Geometry.ShapeValid p) (input : Memory ℝ)
    (head c key value : Nat) (hk : key < p.K) (hv : value < p.V) :
    historyValue algebra dot p h input head c key value hk=target p head input c key value := by
  have hs := states_represents p (Ownership.pid p head value) input h.k_cover c
    (Ownership.hLane p h key value hk) (Ownership.h_active p h head key value hk hv)
  rw [Ownership.column_h] at hs
  exact hs

theorem corrected_value (p : Parameters) (h : Geometry.ShapeValid p) (input : Memory ℝ)
    (head t value : Nat) (ht : t < p.T) (hv : value < p.V) :
    correctedValue algebra dot p h input head t value=
      RealMath.corrected (arrays p head input) p.K (target p head input (t/p.BT)) t value := by
  have hs := corrected_state p (Ownership.pid p head value) input h.k_cover
    (t/p.BT) (Ownership.block p t) (Ownership.cLane p h t value)
    (Ownership.c_active p h head t value ht hv)
  rw [Ownership.time_c,Ownership.column_c] at hs
  exact hs

/-- Complete mathematical outputs and exact frame. The corrected row uses
the last visiting chunk floor(t/BT), even when earlier windows overlap it. -/
structure Spec (p : Parameters) (input before after : Memory ℝ) : Prop where
  history : ∀ head, head < p.B*p.H → ∀ c, c < p.NT → ∀ key, key < p.K →
    ∀ value, value < p.V → after "history" (((head*p.NT+c)*p.K+key)*p.V+value)=
      .mk .real (target p head input c key value)
  corrected : ∀ head, head < p.B*p.H → ∀ t, t < p.T → ∀ value, value < p.V →
    after "corrected" ((head*p.T+t)*p.V+value)=.mk .real
      (RealMath.corrected (arrays p head input) p.K (target p head input (t/p.BT)) t value)
  final : p.storeFinal=Bool.true → ∀ head, head < p.B*p.H → ∀ key, key < p.K →
    ∀ value, value < p.V → after "final" ((head*p.K+key)*p.V+value)=
      .mk .real (target p head input p.NT key value)
  frame : ∀ r o, ¬OutputCell p r o → after r o=before r o

theorem grid_to_real (p : Parameters) (h : Geometry.ShapeValid p) (input before after : Memory ℝ)
    (hs : GridSpec algebra dot p h input before after) : Spec p input before after := by
  constructor
  · intro head hh c hc key hk value hv
    rw [hs.history head hh c hc key hk value hv,history_value p h input head c key value hk hv]
  · intro head hh t ht value hv
    rw [hs.corrected head hh t ht value hv,corrected_value p h input head t value ht hv]
  · intro hp head hh key hk value hv
    rw [hs.final hp head hh key hk value hv]
    exact congrArg (Cell.mk .real) (history_value p h input head p.NT key value hk hv)
  · exact hs.frame

theorem whole (p : Parameters) (h : Geometry.ShapeValid p) (schedule : List PID)
    (hs : schedule.Perm (grid p)) (input before : Memory ℝ) (hi : SameInputs before input) :
    ∃ after, runGrid algebra dot p schedule before=some after ∧ Spec p input before after := by
  obtain ⟨after,he,hg⟩ := whole_grid algebra dot p h schedule hs input before hi
  exact ⟨after,he,grid_to_real p h input before after hg⟩

#axiomsClean whole
#specNonCircular Spec avoiding [run,FP.Structural.exec,runGrid,gridMemory,states]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
