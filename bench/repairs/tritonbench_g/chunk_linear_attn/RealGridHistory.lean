import bench.repairs.tritonbench_g.chunk_linear_attn.RealOutput
import bench.repairs.tritonbench_g.chunk_linear_attn.WholeFrame
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem query_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    query p head a=query p head b := by
  funext t e
  unfold query
  rw [h "query" (by simp)]

theorem key_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    key p head a=key p head b := by
  funext t e
  unfold key
  rw [h "key" (by simp)]

theorem value_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    value p head a=value p head b := by
  funext t j
  unfold value
  rw [h "value" (by simp)]

theorem incoming_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    incoming p head a=incoming p head b := by
  funext t j
  unfold incoming
  rw [h "incoming" (by simp)]

theorem endpoint_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    endpoint p head a=endpoint p head b := by
  funext rev seed e j
  unfold endpoint
  have he : ∀ o, a (seedName rev) o=b (seedName rev) o := by
    apply h
    cases rev <;> simp [seedName]
  simp only [he]

theorem inputs_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (seed : Bool) (h : StateInputs a b) :
    inputs p head a seed=inputs p head b seed := by
  unfold inputs
  rw [query_congr p head a b h,key_congr p head a b h,value_congr p head a b h,endpoint_congr p head a b h]

theorem prior_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (seed : Bool) (h : StateInputs a b) :
    prior p head a seed=prior p head b seed := by
  funext c e j
  unfold prior
  rw [endpoint_congr p head a b h,key_congr p head a b h,value_congr p head a b h]

theorem posterior_congr (p : Parameters) (head : Nat) (a b : Memory ℝ) (scale : ℝ) (terminal : Bool)
    (h : StateInputs a b) : posterior p head a scale terminal=posterior p head b scale terminal := by
  funext c e j
  unfold posterior
  rw [endpoint_congr p head a b h,query_congr p head a b h,incoming_congr p head a b h]

theorem history_transfer (p : Parameters) (head c : Nat) (a b : Memory ℝ) (seed : Bool)
    (hd : StateInputs a b) (hr : ∀ o, a "history" o=b "history" o)
    (hh : HistoryAt p head c b seed) : HistoryAt p head c a seed := by
  intro e j he hj
  rw [hr,prior_congr p head a b seed hd]
  exact hh e j he hj

theorem reverse_history_transfer (p : Parameters) (head c : Nat) (a b : Memory ℝ) (scale : ℝ) (terminal : Bool)
    (hd : StateInputs a b) (hr : ∀ o, a "historyGradient" o=b "historyGradient" o)
    (hh : ReverseHistoryAt p head c b scale terminal) : ReverseHistoryAt p head c a scale terminal := by
  intro e j he hj
  rw [hr,posterior_congr p head a b scale terminal hd]
  exact hh e j he hj

theorem forward_grid_history (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (seed final : Bool) (head c : Nat) (hh : head<p.B*p.H) (hc : c<p.NT) :
    HistoryAt p head c
      (gridMemory algebra dotPrimitive .forwardState p scale seed final (grid .forwardState p) m) seed := by
  let result := gridMemory algebra dotPrimitive .forwardState p scale seed final (grid .forwardState p) m
  have hd : StateInputs result m := grid_data algebra dotPrimitive .forwardState p scale seed final _ m
  intro e j he hj
  rw [prior_congr p head result m seed hd]
  let owner := Ownership.statePID p head e j
  let lane := Ownership.stateLane p shape e j
  have hv : ValidPID .forwardState p owner := Ownership.state_valid p shape head e j hh he hj
  have ha : hActive p owner lane := Ownership.state_active p shape head e j he hj
  have hr := grid_forward_history algebra dotPrimitive p scale seed final m owner hv c hc lane ha
  change result "history" (hOffset p owner c lane)=_ at hr
  have ho : hOffset p owner c lane=head*(p.NT*p.K*p.V)+c*p.K*p.V+e*p.V+j := by
    rw [Ownership.history_offset]
    ring
  rw [ho] at hr
  have hx := congrArg (fun a : Cell ℝ => a.read .real) hr
  rw [forward_state_eq p owner m scale seed c lane shape.bt_pos ha] at hx
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,Cell.read,
    before,inputs,prior,owner,lane,Ownership.statePID,Ownership.stateLane,Nat.div_add_mod'] using hx

theorem backward_grid_history (p : Parameters) (shape : Geometry.ShapeValid p) (m : Memory ℝ)
    (scale : ℝ) (terminal final : Bool) (hcover : p.T≤p.NT*p.BT)
    (head c : Nat) (hh : head<p.B*p.H) (hc : c<p.NT) :
    ReverseHistoryAt p head c
      (gridMemory algebra dotPrimitive .backwardState p scale terminal final (grid .backwardState p) m) scale terminal := by
  let result := gridMemory algebra dotPrimitive .backwardState p scale terminal final (grid .backwardState p) m
  have hd : StateInputs result m := grid_data algebra dotPrimitive .backwardState p scale terminal final _ m
  intro e j he hj
  rw [posterior_congr p head result m scale terminal hd]
  let owner := Ownership.statePID p head e j
  let lane := Ownership.stateLane p shape e j
  have hv : ValidPID .backwardState p owner := Ownership.state_valid p shape head e j hh he hj
  have ha : hActive p owner lane := Ownership.state_active p shape head e j he hj
  have hn := Geometry.chunk_lt p Bool.true c hc
  have hchunk := Ownership.chunk_owner p Bool.true c hc
  have hr := grid_backward_history algebra dotPrimitive p scale terminal final m owner hv (chunk p Bool.true c) hn lane ha
  rw [hchunk] at hr
  change result "historyGradient" (hOffset p owner c lane)=_ at hr
  have ho : hOffset p owner c lane=head*(p.NT*p.K*p.V)+c*p.K*p.V+e*p.V+j := by
    rw [Ownership.history_offset]
    ring
  rw [ho] at hr
  have hx := congrArg (fun a : Cell ℝ => a.read .real) hr
  rw [backward_history_eq p owner m scale Bool.false terminal (chunk p Bool.true c) lane shape.bt_pos hcover hn ha,
    hchunk] at hx
  simpa [cast,algebra,VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra,Cell.read,
    after,inputs,posterior,owner,lane,Ownership.statePID,Ownership.stateLane,Nat.div_add_mod'] using hx

#axiomsClean inputs_congr
#axiomsClean history_transfer
#axiomsClean reverse_history_transfer
#axiomsClean forward_grid_history
#axiomsClean backward_grid_history
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
