import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelReference

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

theorem table_complete {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (value : ι → Cell α) (before after : Memory α)
    (read : ∀ i,after (region i) (index i)=value i)
    (frame : ∀ r o,(¬∃ i,region i=r ∧ index i=o) → after r o=before r o) :
    after=table region index value before := by
  classical
  funext r o
  unfold table
  split
  · rename_i h
    have hv := h.choose_spec
    exact (congrArg₂ after hv.1 hv.2).symm.trans (read h.choose)
  · rename_i h
    exact frame r o h

def stateStage (rev : Bool) : Stage := if rev then .backwardState else .forwardState

theorem state_history_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (input : Memory α)
    (h : Fin (p.B*p.H)) (t : Fin p.NT) (k : Fin p.K) (v : Fin p.V) :
    gridMemory M D (stateStage rev) p scale seed final (grid (stateStage rev) p) input
      (stateRegion rev (.history h t k v : StateCell p final))
      (stateIndex (.history h t k v : StateCell p final))=
      stateValue M D p shape scale rev seed input (.history h t k v : StateCell p final) := by
  let pid := Ownership.statePID p h.val k.val v.val
  let lane := Ownership.stateLane p shape k.val v.val
  have hp := Ownership.state_valid p shape h.val k.val v.val h.isLt k.isLt v.isLt
  have ha := Ownership.state_active p shape h.val k.val v.val k.isLt v.isLt
  have ht := Geometry.chunk_lt p rev t.val t.isLt
  have undo := Ownership.chunk_owner p rev t.val t.isLt
  cases rev with
  | «false» =>
    have hr := grid_forward_history M D p scale seed final input pid hp (chunk p Bool.false t.val) ht lane ha
    rw [undo,Ownership.history_offset] at hr
    exact hr
  | «true» =>
    have hr := grid_backward_history M D p scale seed final input pid hp (chunk p Bool.true t.val) ht lane ha
    rw [undo,Ownership.history_offset] at hr
    exact hr

theorem state_endpoint_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (input : Memory α)
    (enabled : final=Bool.true) (h : Fin (p.B*p.H)) (k : Fin p.K) (v : Fin p.V) :
    gridMemory M D (stateStage rev) p scale seed final (grid (stateStage rev) p) input
      (stateRegion rev (.endpoint enabled h k v)) (stateIndex (.endpoint enabled h k v))=
      stateValue M D p shape scale rev seed input (.endpoint enabled h k v) := by
  subst final
  let pid := Ownership.statePID p h.val k.val v.val
  let lane := Ownership.stateLane p shape k.val v.val
  have hp := Ownership.state_valid p shape h.val k.val v.val h.isLt k.isLt v.isLt
  have ha := Ownership.state_active p shape h.val k.val v.val k.isLt v.isLt
  cases rev with
  | «false» =>
    have hr := grid_forward_final M D p scale seed input pid hp lane ha
    rw [Ownership.final_offset] at hr
    exact hr
  | «true» =>
    have hr := grid_backward_final M D p scale seed input pid hp lane ha
    rw [Ownership.final_offset] at hr
    exact hr

theorem state_footprint (p : Parameters) (rev final : Bool) (pid : PID)
    (hp : ValidPID (stateStage rev) p pid) (r : RegionName) (o : Nat)
    (hf : Footprint (stateStage rev) p final pid r o) :
    ∃ i : StateCell p final,stateRegion rev i=r ∧ stateIndex i=o := by
  have head : pid.z<p.B*p.H := by cases rev <;> exact hp.2.2
  have footprint : StateFootprint p pid rev final r o := by cases rev <;> exact hf
  rcases footprint with ⟨hr,c,hc,i,hi,ho⟩|⟨hr,hfinal,i,hi,ho⟩
  · refine ⟨.history ⟨pid.z,head⟩ ⟨c,hc⟩ ⟨pid.x*p.BK+i.1.val,hi.1⟩
      ⟨pid.y*p.BV+i.2.1.val,hi.2⟩,hr.symm,?_⟩
    rw [ho,Geometry.history_address]
    rfl
  · refine ⟨.endpoint hfinal ⟨pid.z,head⟩ ⟨pid.x*p.BK+i.1.val,hi.1⟩
      ⟨pid.y*p.BV+i.2.1.val,hi.2⟩,hr.symm,?_⟩
    rw [ho,Geometry.final_address]
    rfl

theorem state_eq (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (input : Memory α) :
    gridMemory M D (stateStage rev) p scale seed final (grid (stateStage rev) p) input=
      state M D p shape scale rev seed final input := by
  apply table_complete
  · intro i
    cases i with
    | history h t k v => exact state_history_read M D p shape scale rev seed final input h t k v
    | endpoint enabled h k v => exact state_endpoint_read M D p shape scale rev seed final input enabled h k v
  · intro r o outside
    apply grid_frame
    intro pid hp hf
    exact outside (state_footprint p rev final pid ((mem_grid _ _ _).mp hp) r o hf)

#axiomsClean table_complete
#axiomsClean state_eq
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
