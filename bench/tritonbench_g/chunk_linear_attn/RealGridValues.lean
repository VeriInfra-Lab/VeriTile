import bench.tritonbench_g.chunk_linear_attn.RealGradientQK
import bench.tritonbench_g.chunk_linear_attn.RealGradientV
import bench.tritonbench_g.chunk_linear_attn.RealGridHistory
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (owner_lt packed_lt)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem grid_output_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed : Bool) (hcover : p.T≤p.NT*p.BT) (hkcover : p.K≤keyBlocks p*p.BK)
    (head : Nat) (hh : head<p.B*p.H) (t : Fin p.T) (j : Fin p.V)
    (history : HistoryAt p head (t.val/p.BT) m seed) :
    gridMemory algebra dotPrimitive .forwardOutput p scale Bool.false Bool.false (grid .forwardOutput p) m
      "output" ((head*p.T+t.val)*p.V+j.val)=
      .mk .real (output scale (inputs p head m seed) t j) := by
  let owner := Ownership.outputPID p head t.val j.val
  let lane := Ownership.outputLane p shape t.val j.val
  have hv : ValidPID .forwardOutput p owner := Ownership.output_valid p shape head t.val j.val hh t.isLt j.isLt
  have ha : outputActive p owner lane := Ownership.output_active p shape head t.val j.val t.isLt j.isLt
  have hr := grid_output algebra dotPrimitive p scale m owner hv lane ha
  rw [Ownership.output_offset] at hr
  rw [output_result_correct p owner m seed scale history hkcover shape.bt_pos hcover hv.2.1 lane ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.outputPID,Ownership.outputLane,Nat.div_add_mod'] using hr

theorem grid_query_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed terminal : Bool) (hcover : p.T≤p.NT*p.BT) (hvcover : p.V≤valueBlocks p*p.BV)
    (head : Nat) (hh : head<p.B*p.H) (t : Fin p.T) (e : Fin p.K)
    (history : HistoryAt p head (t.val/p.BT) m seed) :
    gridMemory algebra dotPrimitive .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "queryGradient" ((head*p.T+t.val)*p.K+e.val)=
      .mk .real ((gradients scale (inputs p head m seed)
        (fun t j => incoming p head m t.val j.val)
        (fun e j => endpoint p head m Bool.true terminal e.val j.val)).q t e) := by
  let owner := Ownership.gradientPID p head t.val e.val
  let lane := Ownership.gradientLane p shape t.val e.val
  have hv : ValidPID .backwardGradient p owner := Ownership.gradient_valid p shape head t.val e.val hh t.isLt e.isLt
  have ha : gradKeyActive p owner lane := Ownership.gradient_active p shape head t.val e.val t.isLt e.isLt
  have hr := grid_query_gradient algebra dotPrimitive p scale m owner hv lane ha
  rw [Ownership.gradient_offset] at hr
  rw [grad_query_correct p owner m seed terminal scale history hvcover shape.bt_pos hcover hv.2.1 lane ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.gradientPID,Ownership.gradientLane,Nat.div_add_mod'] using hr

theorem grid_key_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed terminal : Bool) (hcover : p.T≤p.NT*p.BT) (hvcover : p.V≤valueBlocks p*p.BV)
    (head : Nat) (hh : head<p.B*p.H) (t : Fin p.T) (e : Fin p.K)
    (history : ReverseHistoryAt p head (t.val/p.BT) m scale terminal) :
    gridMemory algebra dotPrimitive .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "keyGradient" ((head*p.T+t.val)*p.K+e.val)=
      .mk .real ((gradients scale (inputs p head m seed)
        (fun t j => incoming p head m t.val j.val)
        (fun e j => endpoint p head m Bool.true terminal e.val j.val)).k t e) := by
  let owner := Ownership.gradientPID p head t.val e.val
  let lane := Ownership.gradientLane p shape t.val e.val
  have hv : ValidPID .backwardGradient p owner := Ownership.gradient_valid p shape head t.val e.val hh t.isLt e.isLt
  have ha : gradKeyActive p owner lane := Ownership.gradient_active p shape head t.val e.val t.isLt e.isLt
  have hr := grid_key_gradient algebra dotPrimitive p scale m owner hv lane ha
  rw [Ownership.gradient_offset] at hr
  rw [grad_key_correct p owner m seed terminal scale history hvcover shape.bt_pos hcover hv.2.1 lane ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.gradientPID,Ownership.gradientLane,Nat.div_add_mod'] using hr

theorem grid_partial_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (terminal : Bool) (hcover : p.T≤p.NT*p.BT)
    (head : Nat) (hh : head<p.B*p.H) (t : Fin p.T) (j : Fin p.V) (kb : Fin (keyBlocks p))
    (history : ReverseHistoryAt p head (t.val/p.BT) m scale terminal) :
    gridMemory algebra dotPrimitive .backwardGradient p scale Bool.false Bool.false (grid .backwardGradient p) m
      "valuePartials" (((kb.val*(p.B*p.H)+head)*p.T+t.val)*p.V+j.val)=
      .mk .real (valuePartial p head (t.val/p.BT) m terminal scale kb.val t.val j.val) := by
  let owner := Ownership.partialPID p kb.val head t.val
  let lane := Ownership.outputLane p shape t.val j.val
  have hv : ValidPID .backwardGradient p owner := Ownership.partial_valid p shape kb.val head t.val kb.isLt hh t.isLt
  have ha : gradValueActive p owner (j.val/p.BV) lane := Ownership.partial_active p shape kb.val head t.val j.val t.isLt j.isLt
  have hj : j.val/p.BV<valueBlocks p := owner_lt shape.bv_pos j.isLt
  have hr := grid_value_partial algebra dotPrimitive p scale m owner hv (j.val/p.BV) hj lane ha
  rw [Ownership.partial_offset] at hr
  rw [grad_value_partial p owner m scale terminal history shape.bt_pos hcover hv.2.1 (j.val/p.BV) lane ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.partialPID,Ownership.outputLane,Nat.div_add_mod'] using hr

theorem grid_forward_final_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed : Bool) (hcover : p.T≤p.NT*p.BT)
    (head : Nat) (hh : head<p.B*p.H) (e : Fin p.K) (j : Fin p.V) :
    gridMemory algebra dotPrimitive .forwardState p scale seed Bool.true (grid .forwardState p) m
      "final" ((head*p.K+e.val)*p.V+j.val)=
      .mk .real (finalState (inputs p head m seed) e j) := by
  let owner := Ownership.statePID p head e.val j.val
  let lane := Ownership.stateLane p shape e.val j.val
  have hv : ValidPID .forwardState p owner := Ownership.state_valid p shape head e.val j.val hh e.isLt j.isLt
  have ha : hActive p owner lane := Ownership.state_active p shape head e.val j.val e.isLt j.isLt
  have hr := grid_forward_final algebra dotPrimitive p scale seed m owner hv lane ha
  change gridMemory _ _ _ _ _ _ _ _ _ "final" _=_ at hr
  rw [Ownership.final_offset] at hr
  rw [forward_final_eq p owner m scale seed lane shape.bt_pos hcover ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.statePID,Ownership.stateLane,Nat.div_add_mod'] using hr

theorem grid_backward_final_real (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed terminal : Bool) (hcover : p.T≤p.NT*p.BT)
    (head : Nat) (hh : head<p.B*p.H) (e : Fin p.K) (j : Fin p.V) :
    gridMemory algebra dotPrimitive .backwardState p scale terminal Bool.true (grid .backwardState p) m
      "initialGradient" ((head*p.K+e.val)*p.V+j.val)=
      .mk .real ((gradients scale (inputs p head m seed)
        (fun t j => incoming p head m t.val j.val)
        (fun e j => endpoint p head m Bool.true terminal e.val j.val)).initial e j) := by
  let owner := Ownership.statePID p head e.val j.val
  let lane := Ownership.stateLane p shape e.val j.val
  have hv : ValidPID .backwardState p owner := Ownership.state_valid p shape head e.val j.val hh e.isLt j.isLt
  have ha : hActive p owner lane := Ownership.state_active p shape head e.val j.val e.isLt j.isLt
  have hr := grid_backward_final algebra dotPrimitive p scale terminal m owner hv lane ha
  change gridMemory _ _ _ _ _ _ _ _ _ "initialGradient" _=_ at hr
  rw [Ownership.final_offset] at hr
  rw [backward_final_eq p owner m scale seed terminal lane shape.bt_pos hcover ha] at hr
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,owner,lane,
    Ownership.statePID,Ownership.stateLane,Nat.div_add_mod'] using hr

#axiomsClean grid_output_real
#axiomsClean grid_query_real
#axiomsClean grid_key_real
#axiomsClean grid_partial_real
#axiomsClean grid_forward_final_real
#axiomsClean grid_backward_final_real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
