import bench.tritonbench_g.chunk_linear_attn.Proofs.ReferenceState

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

theorem output_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) (i : OutputCell p) :
    gridMemory M D .forwardOutput p scale Bool.false Bool.false (grid .forwardOutput p) input
      "output" (outputIndex i)=outputValue M D p shape scale input i := by
  have hp := Ownership.output_valid p shape i.1.val i.2.1.val i.2.2.val i.1.isLt i.2.1.isLt i.2.2.isLt
  have ha := Ownership.output_active p shape i.1.val i.2.1.val i.2.2.val i.2.1.isLt i.2.2.isLt
  have hr := grid_output M D p scale input _ hp _ ha
  rw [Ownership.output_offset] at hr
  exact hr

theorem output_footprint (p : Parameters) (pid : PID) (hp : ValidPID .forwardOutput p pid)
    (r : RegionName) (o : Nat) (hf : OutputFootprint p pid r o) :
    ∃ i : OutputCell p,("output" : RegionName)=r ∧ outputIndex i=o := by
  rcases hf with ⟨hr,i,hi,ho⟩
  refine ⟨⟨⟨pid.z,hp.2.2⟩,⟨pid.y*p.BT+i.1.val,hi.1⟩,
    ⟨pid.x*p.BV+i.2.1.val,hi.2⟩⟩,hr.symm,?_⟩
  rw [ho,Geometry.output_address]
  rfl

theorem output_eq (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) :
    gridMemory M D .forwardOutput p scale Bool.false Bool.false (grid .forwardOutput p) input=
      output M D p shape scale input := by
  apply table_complete
  · exact output_read M D p shape scale input
  · intro r o outside
    apply grid_frame
    intro pid hp hf
    exact outside (output_footprint p pid ((mem_grid _ _ _).mp hp) r o hf)

theorem gradient_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) (i : GradientCell p) :
    gridMemory M D .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) input
      (gradientRegion i) (gradientIndex i)=gradientValue M D p shape scale input i := by
  cases i with
  | query h t k =>
    have hp := Ownership.gradient_valid p shape h.val t.val k.val h.isLt t.isLt k.isLt
    have ha := Ownership.gradient_active p shape h.val t.val k.val t.isLt k.isLt
    have hr := grid_query_gradient M D p scale input _ hp _ ha
    rw [Ownership.gradient_offset] at hr
    exact hr
  | key h t k =>
    have hp := Ownership.gradient_valid p shape h.val t.val k.val h.isLt t.isLt k.isLt
    have ha := Ownership.gradient_active p shape h.val t.val k.val t.isLt k.isLt
    have hr := grid_key_gradient M D p scale input _ hp _ ha
    rw [Ownership.gradient_offset] at hr
    exact hr
  | valuePartial b h t v =>
    have hp := Ownership.partial_valid p shape b.val h.val t.val b.isLt h.isLt t.isLt
    have ha := Ownership.partial_active p shape b.val h.val t.val v.val t.isLt v.isLt
    have hv := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt shape.bv_pos v.isLt
    have hr := grid_value_partial M D p scale input _ hp (v.val/p.BV) hv _ ha
    rw [Ownership.partial_offset] at hr
    exact hr

theorem gradient_footprint (p : Parameters) (pid : PID) (hp : ValidPID .backwardGradient p pid)
    (r : RegionName) (o : Nat) (hf : GradientFootprint p pid r o) :
    ∃ i : GradientCell p,gradientRegion i=r ∧ gradientIndex i=o := by
  rcases hf with ⟨hr,i,hi,ho⟩|⟨hr,vb,hvb,i,hi,ho⟩
  · rcases hr with hr|hr
    · refine ⟨.query ⟨pid.z,hp.2.2⟩ ⟨pid.y*p.BT+i.1.val,hi.1⟩
        ⟨pid.x*p.BK+i.2.1.val,hi.2⟩,hr.symm,?_⟩
      rw [ho,Geometry.gradient_address]
      rfl
    · refine ⟨.key ⟨pid.z,hp.2.2⟩ ⟨pid.y*p.BT+i.1.val,hi.1⟩
        ⟨pid.x*p.BK+i.2.1.val,hi.2⟩,hr.symm,?_⟩
      rw [ho,Geometry.gradient_address]
      rfl
  · refine ⟨.valuePartial ⟨pid.x,hp.1⟩ ⟨pid.z,hp.2.2⟩ ⟨pid.y*p.BT+i.1.val,hi.1⟩
      ⟨vb*p.BV+i.2.1.val,hi.2⟩,hr.symm,?_⟩
    rw [ho,Geometry.partial_address]
    rfl

theorem gradient_eq (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) :
    gridMemory M D .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) input=
      gradient M D p shape scale input := by
  apply table_complete
  · exact gradient_read M D p shape scale input
  · intro r o outside
    apply grid_frame
    intro pid hp hf
    exact outside (gradient_footprint p pid ((mem_grid _ _ _).mp hp) r o hf)

theorem sum_eq (M : Algebra α) (p : Parameters) (selection : LibrarySelection p) (input : Memory α) :
    hostSumMemory M p selection input input=sum M p selection input := by
  apply table_complete
  · exact host_sum_read M p selection input input
  · intro r o outside
    apply host_sum_frame
    by_cases hr : r="valueGradient"
    · right
      by_contra h
      apply outside
      exact ⟨⟨o,by omega⟩,hr.symm,rfl⟩
    · exact Or.inl hr

theorem result_eq (M : Algebra α) (D : DotPrimitive α) (dims : Host.Dimensions) (valid : Host.Valid dims)
    (scale : ℝ) (seed final terminal : Bool) (selection : LibrarySelection (dims.parameters Bool.true))
    (input : Memory α) :
    completeMemory M D (dims.parameters Bool.false) (dims.parameters Bool.true)
      scale seed final terminal selection input=result M D dims valid scale seed final terminal selection input := by
  unfold completeMemory modelFour result
  dsimp only
  rw [show Stage.forwardState=stateStage Bool.false from rfl,
    state_eq M D _ (Host.shape_valid dims valid Bool.false),
    output_eq M D _ (Host.shape_valid dims valid Bool.false)]
  rw [show Stage.backwardState=stateStage Bool.true from rfl,
    state_eq M D _ (Host.shape_valid dims valid Bool.true),
    gradient_eq M D _ (Host.shape_valid dims valid Bool.true),sum_eq]

#axiomsClean output_eq
#axiomsClean gradient_eq
#axiomsClean sum_eq
#axiomsClean result_eq
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
