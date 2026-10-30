import bench.repairs.tritonbench_g.chunk_linear_attn.RealSumValues
import bench.repairs.tritonbench_g.chunk_linear_attn.HostMetadata
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Real
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

structure Call where
  dims : Host.Dimensions
  scale : ℝ
  (seed final terminal : Bool)
  selection : LibrarySelection (dims.parameters Bool.true)
def Call.forward (c : Call) := c.dims.parameters Bool.false
def Call.backward (c : Call) := c.dims.parameters Bool.true
def Call.hf (c : Call) (m : Memory ℝ) := gridMemory algebra dotPrimitive .forwardState c.forward c.scale c.seed c.final (grid .forwardState c.forward) m
def Call.fo (c : Call) (m : Memory ℝ) := gridMemory algebra dotPrimitive .forwardOutput c.forward c.scale Bool.false Bool.false (grid .forwardOutput c.forward) (c.hf m)
def Call.hb (c : Call) (m : Memory ℝ) := gridMemory algebra dotPrimitive .backwardState c.backward c.scale c.terminal c.seed (grid .backwardState c.backward) (c.fo m)
def Call.bg (c : Call) (m : Memory ℝ) := gridMemory algebra dotPrimitive .backwardGradient c.backward c.scale Bool.false Bool.false (grid .backwardGradient c.backward) (c.hb m)
def Call.result (c : Call) (m : Memory ℝ) := hostSumMemory algebra c.backward c.selection (c.bg m) (c.bg m)
def Call.target (c : Call) (head : Nat) (m : Memory ℝ) : Inputs c.dims.T c.dims.K c.dims.V :=
  gradients c.scale (inputs c.backward head m c.seed)
    (fun t j => incoming c.backward head m t.val j.val)
    (fun e j => endpoint c.backward head m Bool.true c.terminal e.val j.val)

theorem Call.result_is_complete (c : Call) (m : Memory ℝ) :
    c.result m=completeMemory algebra dotPrimitive c.forward c.backward c.scale c.seed c.final c.terminal c.selection m := rfl

theorem Call.hf_data (c : Call) (m : Memory ℝ) : StateInputs (c.hf m) m :=
  grid_data _ _ _ _ _ _ _ _ _
theorem Call.fo_data (c : Call) (m : Memory ℝ) : StateInputs (c.fo m) m :=
  (grid_data _ _ _ _ _ _ _ _ _).trans (c.hf_data m)
theorem Call.hb_data (c : Call) (m : Memory ℝ) : StateInputs (c.hb m) m :=
  (grid_data _ _ _ _ _ _ _ _ _).trans (c.fo_data m)
theorem Call.bg_data (c : Call) (m : Memory ℝ) : StateInputs (c.bg m) m :=
  (grid_data _ _ _ _ _ _ _ _ _).trans (c.hb_data m)

theorem Call.target_congr (c : Call) (head : Nat) (a b : Memory ℝ) (h : StateInputs a b) :
    c.target head a=c.target head b := by
  unfold Call.target
  rw [inputs_congr c.backward head a b c.seed h,incoming_congr c.backward head a b h,
    endpoint_congr c.backward head a b h]

theorem Call.forward_history (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head n : Nat) (hh : head<c.dims.B*c.dims.H) (hn : n<c.forward.NT) :
    HistoryAt c.forward head n (c.hf m) c.seed :=
  forward_grid_history c.forward (Host.shape_valid c.dims valid Bool.false) m c.scale c.seed c.final head n hh hn

theorem Call.backward_history (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head n : Nat) (hh : head<c.dims.B*c.dims.H) (hn : n<c.backward.NT) :
    ReverseHistoryAt c.backward head n (c.hb m) c.scale c.terminal :=
  backward_grid_history c.backward (Host.shape_valid c.dims valid Bool.true) (c.fo m)
    c.scale c.terminal c.seed (Host.covers c.dims valid Bool.true).1 head n hh hn

theorem Call.forward_history_at_backward (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head n : Nat) (hh : head<c.dims.B*c.dims.H) (hn : n<c.backward.NT) :
    HistoryAt c.backward head n (c.hb m) c.seed := by
  have hd : StateInputs (c.hb m) (c.hf m) :=
    (grid_data _ _ _ _ _ _ _ _ _).trans (grid_data _ _ _ _ _ _ _ _ _)
  have hr : ∀ o, c.hb m "history" o=c.hf m "history" o := by
    intro o
    unfold Call.hb
    rw [grid_region_frame algebra dotPrimitive .backwardState c.backward c.scale c.terminal c.seed _ _ "history" (by decide)]
    unfold Call.fo
    exact grid_region_frame algebra dotPrimitive .forwardOutput c.forward c.scale Bool.false Bool.false _ _ "history" (by decide) o
  apply history_transfer c.backward head n (c.hb m) (c.hf m) c.seed hd hr
  exact c.forward_history valid m head n hh hn

theorem Call.after_hb_frame (c : Call) (m : Memory ℝ) (r : RegionName) (o : Nat)
    (hs : r≠"valueGradient") (hg : r∉writeRegions .backwardGradient) :
    c.result m r o=c.hb m r o := by
  unfold Call.result
  rw [host_sum_frame algebra c.backward c.selection _ _ r o (Or.inl hs)]
  unfold Call.bg
  exact grid_region_frame algebra dotPrimitive .backwardGradient c.backward c.scale Bool.false Bool.false _ _ r hg o

theorem Call.after_fo_frame (c : Call) (m : Memory ℝ) (r : RegionName) (o : Nat)
    (hs : r≠"valueGradient") (hg : r∉writeRegions .backwardGradient) (hb : r∉writeRegions .backwardState) :
    c.result m r o=c.fo m r o := by
  rw [c.after_hb_frame m r o hs hg]
  unfold Call.hb
  exact grid_region_frame algebra dotPrimitive .backwardState c.backward c.scale c.terminal c.seed _ _ r hb o

theorem Call.output_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V) :
    c.result m "output" ((head*c.dims.T+t.val)*c.dims.V+j.val)=
      .mk .real (output c.scale (inputs c.forward head m c.seed) t j) := by
  rw [c.after_fo_frame m "output" _ (by decide) (by decide) (by decide)]
  have hv := Host.covers c.dims valid Bool.false
  have hs := Host.shape_valid c.dims valid Bool.false
  have hn : t.val/c.forward.BT<c.forward.NT := by
    change t.val/64<VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.ceilDiv c.dims.T 64
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt (by decide) t.isLt
  have hr := grid_output_real c.forward hs (c.hf m) c.scale c.seed hv.1 hv.2.1 head hh t j
    (c.forward_history valid m head _ hh hn)
  rw [inputs_congr c.forward head (c.hf m) m c.seed (c.hf_data m)] at hr
  exact hr

theorem Call.final_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (hf : c.final=Bool.true) (head : Nat) (hh : head<c.dims.B*c.dims.H)
    (e : Fin c.dims.K) (j : Fin c.dims.V) :
    c.result m "final" ((head*c.dims.K+e.val)*c.dims.V+j.val)=
      .mk .real (finalState (inputs c.forward head m c.seed) e j) := by
  rw [c.after_fo_frame m "final" _ (by decide) (by decide) (by decide)]
  unfold Call.fo
  rw [grid_region_frame algebra dotPrimitive .forwardOutput c.forward c.scale Bool.false Bool.false _ _ "final" (by decide)]
  unfold Call.hf
  rw [hf]
  exact grid_forward_final_real c.forward (Host.shape_valid c.dims valid Bool.false) m c.scale c.seed
    (Host.covers c.dims valid Bool.false).1 head hh e j

theorem Call.query_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (e : Fin c.dims.K) :
    c.result m "queryGradient" ((head*c.dims.T+t.val)*c.dims.K+e.val)=
      .mk .real ((c.target head m).q t e) := by
  unfold Call.result
  rw [host_sum_frame algebra c.backward c.selection _ _ "queryGradient" _ (Or.inl (by decide))]
  have hv := Host.covers c.dims valid Bool.true
  have hs := Host.shape_valid c.dims valid Bool.true
  have hn : t.val/c.backward.BT<c.backward.NT := by
    change t.val/64<VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.ceilDiv c.dims.T 64
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt (by decide) t.isLt
  have hr := grid_query_real c.backward hs (c.hb m) c.scale c.seed c.terminal hv.1 hv.2.2 head hh t e
    (c.forward_history_at_backward valid m head _ hh hn)
  change c.bg m "queryGradient" _=Cell.mk .real ((c.target head (c.hb m)).q t e) at hr
  rw [c.target_congr head (c.hb m) m (c.hb_data m)] at hr
  exact hr

theorem Call.key_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (e : Fin c.dims.K) :
    c.result m "keyGradient" ((head*c.dims.T+t.val)*c.dims.K+e.val)=
      .mk .real ((c.target head m).k t e) := by
  unfold Call.result
  rw [host_sum_frame algebra c.backward c.selection _ _ "keyGradient" _ (Or.inl (by decide))]
  have hv := Host.covers c.dims valid Bool.true
  have hs := Host.shape_valid c.dims valid Bool.true
  have hn : t.val/c.backward.BT<c.backward.NT := by
    change t.val/64<VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.ceilDiv c.dims.T 64
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt (by decide) t.isLt
  have hr := grid_key_real c.backward hs (c.hb m) c.scale c.seed c.terminal hv.1 hv.2.2 head hh t e
    (c.backward_history valid m head _ hh hn)
  change c.bg m "keyGradient" _=Cell.mk .real ((c.target head (c.hb m)).k t e) at hr
  rw [c.target_congr head (c.hb m) m (c.hb_data m)] at hr
  exact hr

theorem Call.value_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (head : Nat) (hh : head<c.dims.B*c.dims.H) (t : Fin c.dims.T) (j : Fin c.dims.V) :
    c.result m "valueGradient" ((head*c.dims.T+t.val)*c.dims.V+j.val)=
      .mk .real ((c.target head m).v t j) := by
  have hv := Host.covers c.dims valid Bool.true
  have hs := Host.shape_valid c.dims valid Bool.true
  have hn : t.val/c.backward.BT<c.backward.NT := by
    change t.val/64<VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.ceilDiv c.dims.T 64
    exact VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt (by decide) t.isLt
  have hr := grid_sum_value_real c.backward hs (c.hb m) c.selection c.scale c.seed c.terminal hv.1 hv.2.1 head hh t j
    (c.backward_history valid m head _ hh hn)
  change c.result m "valueGradient" _=Cell.mk .real ((c.target head (c.hb m)).v t j) at hr
  rw [c.target_congr head (c.hb m) m (c.hb_data m)] at hr
  exact hr

theorem Call.initial_read (c : Call) (valid : Host.Valid c.dims) (m : Memory ℝ)
    (hseed : c.seed=Bool.true) (head : Nat) (hh : head<c.dims.B*c.dims.H)
    (e : Fin c.dims.K) (j : Fin c.dims.V) :
    c.result m "initialGradient" ((head*c.dims.K+e.val)*c.dims.V+j.val)=
      .mk .real ((c.target head m).initial e j) := by
  rw [c.after_hb_frame m "initialGradient" _ (by decide) (by decide)]
  have hr := grid_backward_final_real c.backward (Host.shape_valid c.dims valid Bool.true) (c.fo m)
    c.scale c.seed c.terminal (Host.covers c.dims valid Bool.true).1 head hh e j
  change gridMemory _ _ .backwardState _ _ _ Bool.true _ _ _ _=
    Cell.mk .real ((c.target head (c.fo m)).initial e j) at hr
  rw [c.target_congr head (c.fo m) m (c.fo_data m)] at hr
  simpa only [Call.hb,hseed] using hr

theorem Call.target_vjp (c : Call) (m : Memory ℝ) (head : Nat) :
    IsVJP c.scale (inputs c.backward head m c.seed)
      (fun t j => incoming c.backward head m t.val j.val)
      (fun e j => endpoint c.backward head m Bool.true c.terminal e.val j.val) (c.target head m) :=
  gradients_correct _ _ _ _

#axiomsClean Call.forward_history_at_backward
#axiomsClean Call.output_read
#axiomsClean Call.final_read
#axiomsClean Call.query_read
#axiomsClean Call.key_read
#axiomsClean Call.value_read
#axiomsClean Call.initial_read
#axiomsClean Call.target_vjp
#specNonCircular Call.target avoiding [run,FP.Structural.exec,Call.result,modelFour]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Real
