import bench.tritonbench_g.lightning_attention.PhaseRealTiles
import bench.tritonbench_g.lightning_attention.PhasePrograms
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
theorem forward_output (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E) :
    afterForward d m "output" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.output (logicalQ d.parameters head.val m) (logicalK d.parameters head.val m)
        (logicalV d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let t : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  let channel := feature.val/d.BM
  let c : Fin d.BM := ⟨feature.val%d.BM,Nat.mod_lt _ (Host.bm_positive d hc)⟩
  let pid : PID := ⟨head.val,channel⟩
  let i : TileIndex [64,d.BM] := (t,c,())
  have ht : n.val*64+t.val=time.val := by simpa only [n,t,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have hf : channel*d.BM+c.val=feature.val := by simpa only [channel,c,Nat.mul_comm] using Nat.div_add_mod feature.val d.BM
  have htime : n.val*64+i.1.val<d.N := by change n.val*64+t.val<d.N;rw [ht];exact time.isLt
  have h := forward_output_tile d hc m pid ⟨head.isLt,channel_owner d hc feature⟩ n i htime
  rw [Geometry.fwd_address] at h
  dsimp only [pid,i,Host.Dimensions.parameters] at h
  simpa [ht,hf,Geometry.dense,Nat.add_mul,Nat.add_assoc] using h

theorem backward_query (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D) :
    backwardMemory d m "queryGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dq (logicalK d.parameters head.val m) (logicalV d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := backward_query_tile d hc m head.val head.isLt n (lane,feature,()) htime
  dsimp only at h
  simpa only [ht] using h

theorem backward_key (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D) :
    backwardMemory d m "keyGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dk (logicalQ d.parameters head.val m) (logicalV d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := backward_key_tile d hc m head.val head.isLt n (feature,lane,()) htime
  dsimp only at h
  simpa only [ht] using h

theorem backward_value (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E) :
    backwardMemory d m "valueGradient" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.dv (logicalQ d.parameters head.val m) (logicalK d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := backward_value_tile d hc m head.val head.isLt n (lane,feature,()) htime
  dsimp only at h
  simpa only [ht] using h


structure ForwardPost (d : Host.Dimensions) (input result : Memory ℝ) : Prop where
  output : ∀ (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E),
    result "output" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.output (logicalQ d.parameters head.val input) (logicalK d.parameters head.val input)
        (logicalV d.parameters head.val input) time feature)
  inputs : Inputs result input
  exterior : ∀ (r : RegionName) o, (r≠"output" ∨ regionCapacity d r≤o) → result r o=input r o
structure BackwardPost (d : Host.Dimensions) (input result : Memory ℝ) : Prop where
  query : ∀ (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D),
    result "queryGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dq (logicalK d.parameters head.val input) (logicalV d.parameters head.val input)
        (logicalG d.parameters head.val input) time feature)
  key : ∀ (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D),
    result "keyGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dk (logicalQ d.parameters head.val input) (logicalV d.parameters head.val input)
        (logicalG d.parameters head.val input) time feature)
  value : ∀ (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E),
    result "valueGradient" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.dv (logicalQ d.parameters head.val input) (logicalK d.parameters head.val input)
        (logicalG d.parameters head.val input) time feature)
  inputs : Inputs result input
  output : ∀ o, result "output" o=input "output" o
  exterior : ∀ r o, regionCapacity d r≤o → result r o=input r o

theorem forward_correct (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ) :
    executeForward algebra dotPrimitive d m=some (afterForward d m) ∧ ForwardPost d m (afterForward d m) := by
  refine ⟨forward_execution _ realMasks _ _ _,forward_output d hc m,afterForward_inputs d m,?_⟩
  intro r o ho
  apply grid_frame
  intro pid hp hw
  rcases ho with hr|ho
  · have := hw.1; exact hr this
  · have := footprint_bound .forward d hc pid ((mem_grid _ _ _).mp hp) r o hw
    omega

theorem backward_correct (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ) :
    executeBackward algebra dotPrimitive d m=some (backwardMemory d m) ∧ BackwardPost d m (backwardMemory d m) := by
  refine ⟨backward_execution _ realMasks _ _ _,backward_query d hc m,backward_key d hc m,backward_value d hc m,?_,?_,?_⟩
  · exact (grid_inputs _ _ _ _ _ _).trans (backwardIntra_inputs d m)
  · intro o
    unfold backwardMemory backwardIntra
    rw [grid_frame,grid_frame]
    · intro pid hp; simp [Footprint,IntraFoot,RowsFoot]
    · intro pid hp; simp [Footprint,InterFoot,HeadFoot]
  · intro r o ho
    have safe : ∀ stage pid, pid∈grid stage d → ¬Footprint stage d pid r o := by
      intro stage pid hp hw
      have := footprint_bound stage d hc pid ((mem_grid stage d pid).mp hp) r o hw
      omega
    unfold backwardMemory backwardIntra
    rw [grid_frame _ _ _ _ _ _ _ _ (safe .inter),grid_frame _ _ _ _ _ _ _ _ (safe .intra)]
#axiomsClean forward_output
#axiomsClean backward_query
#axiomsClean backward_key
#axiomsClean backward_value
#axiomsClean forward_correct
#axiomsClean backward_correct
#specNonCircular ForwardPost avoiding [afterForward,modelForward,executeForward]
#specNonCircular BackwardPost avoiding [backwardMemory,modelBackward,executeBackward]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
