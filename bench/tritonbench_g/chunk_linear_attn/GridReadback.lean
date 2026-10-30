import bench.tritonbench_g.chunk_linear_attn.GridFreeze
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem grid_forward_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (m : Memory α) (pid : PID)
    (hp : ValidPID .forwardState p pid) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    gridMemory M D .forwardState p scale seed final (grid .forwardState p) m
      (historyName Bool.false) (hOffset p pid (chunk p Bool.false n) i)=
      .mk .real (cast M (stateValues M D p pid m scale Bool.false seed n i)) := by
  apply grid_read M D .forwardState p scale seed final _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inl ⟨rfl,chunk p Bool.false n,Geometry.chunk_lt p Bool.false n hn,i,hi,rfl⟩
  · intro mem
    exact state_read_history M D p pid m mem scale Bool.false seed final n i hn hi

theorem grid_forward_final (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (seed : Bool) (m : Memory α) (pid : PID)
    (hp : ValidPID .forwardState p pid) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    gridMemory M D .forwardState p scale seed Bool.true (grid .forwardState p) m
      (finalName Bool.false) (fOffset p pid i)=
      .mk .real (cast M (stateValues M D p pid m scale Bool.false seed p.NT i)) := by
  apply grid_read M D .forwardState p scale seed Bool.true _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inr ⟨rfl,rfl,i,hi,rfl⟩
  · intro mem
    exact state_read_final M D p pid m mem scale Bool.false seed i hi

theorem grid_backward_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (m : Memory α) (pid : PID)
    (hp : ValidPID .backwardState p pid) (n : Nat) (hn : n<p.NT)
    (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    gridMemory M D .backwardState p scale seed final (grid .backwardState p) m
      (historyName Bool.true) (hOffset p pid (chunk p Bool.true n) i)=
      .mk .real (cast M (stateValues M D p pid m scale Bool.true seed n i)) := by
  apply grid_read M D .backwardState p scale seed final _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inl ⟨rfl,chunk p Bool.true n,Geometry.chunk_lt p Bool.true n hn,i,hi,rfl⟩
  · intro mem
    exact state_read_history M D p pid m mem scale Bool.true seed final n i hn hi

theorem grid_backward_final (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (seed : Bool) (m : Memory α) (pid : PID)
    (hp : ValidPID .backwardState p pid) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    gridMemory M D .backwardState p scale seed Bool.true (grid .backwardState p) m
      (finalName Bool.true) (fOffset p pid i)=
      .mk .real (cast M (stateValues M D p pid m scale Bool.true seed p.NT i)) := by
  apply grid_read M D .backwardState p scale seed Bool.true _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inr ⟨rfl,rfl,i,hi,rfl⟩
  · intro mem
    exact state_read_final M D p pid m mem scale Bool.true seed i hi

theorem grid_output (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (m : Memory α) (pid : PID) (hp : ValidPID .forwardOutput p pid)
    (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    gridMemory M D .forwardOutput p scale Bool.false Bool.false (grid .forwardOutput p) m
      "output" (outputOffset p pid i)=
      .mk .real (cast M (outputResult M D p pid m scale (outputStates M D p pid m (keyBlocks p)) i)) := by
  apply grid_read M D .forwardOutput p scale Bool.false Bool.false _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact ⟨rfl,i,hi,rfl⟩
  · intro mem
    exact output_read M D p pid m mem scale i hi

theorem grid_query_gradient (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (m : Memory α) (pid : PID) (hp : ValidPID .backwardGradient p pid)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gridMemory M D .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "queryGradient" (gradKeyOffset p pid i)=
      .mk .real (cast M (gradQueryResult M D p pid m scale (gradStates M D p pid m scale (valueBlocks p)) i)) := by
  apply grid_read M D .backwardGradient p scale Bool.false Bool.false _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inl ⟨Or.inl rfl,i,hi,rfl⟩
  · intro mem
    exact gradient_read_query M D p pid m mem scale i hi

theorem grid_key_gradient (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (m : Memory α) (pid : PID) (hp : ValidPID .backwardGradient p pid)
    (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gridMemory M D .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "keyGradient" (gradKeyOffset p pid i)=
      .mk .real (cast M (gradKeyResult M D p pid m scale (gradStates M D p pid m scale (valueBlocks p)) i)) := by
  apply grid_read M D .backwardGradient p scale Bool.false Bool.false _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inl ⟨Or.inr rfl,i,hi,rfl⟩
  · intro mem
    exact gradient_read_key M D p pid m mem scale i hi

theorem grid_value_partial (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (scale : ℝ) (m : Memory α) (pid : PID) (hp : ValidPID .backwardGradient p pid)
    (vb : Nat) (hv : vb<valueBlocks p) (i : TileIndex [p.BT,p.BV]) (hi : gradValueActive p pid vb i) :
    gridMemory M D .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i)=
      .mk .real (cast M (gradValue M D p pid m scale vb i)) := by
  apply grid_read M D .backwardGradient p scale Bool.false Bool.false _ m pid _ _ _
    (fun x hx => (mem_grid _ _ _).mp hx) ((mem_grid _ _ _).mpr hp)
  · exact Or.inr ⟨rfl,vb,hv,i,hi,rfl⟩
  · intro mem
    exact gradient_read_value M D p pid m mem scale vb i hp.2.2 hv hi

#axiomsClean grid_forward_history
#axiomsClean grid_forward_final
#axiomsClean grid_backward_history
#axiomsClean grid_backward_final
#axiomsClean grid_output
#axiomsClean grid_query_gradient
#axiomsClean grid_key_gradient
#axiomsClean grid_value_partial
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
