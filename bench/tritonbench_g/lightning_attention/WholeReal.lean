import bench.tritonbench_g.lightning_attention.Algebras
import bench.tritonbench_g.lightning_attention.WholeRealTiles
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

theorem time_owner (d : Host.Dimensions) (t : Fin d.N) : t.val/64<d.NT := by
  have ht := t.isLt
  have hc := Host.time_cover d
  omega

theorem channel_owner (d : Host.Dimensions) (h : Host.Contract d) (c : Fin d.E) : c.val/d.BM<d.NV := by
  apply (Nat.div_lt_iff_lt_mul (Host.bm_positive d h)).mpr
  simpa only [Geometry.channels_exact d h] using c.isLt

theorem result_output (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E) :
    resultMemory d m "output" (head.val*d.N*d.E+time.val*d.E+feature.val)=
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
  have h := result_output_tile d hc m pid ⟨head.isLt,channel_owner d hc feature⟩ n i htime
  rw [Geometry.fwd_address] at h
  dsimp only [pid,i,Host.Dimensions.parameters] at h
  simpa [ht,hf,Geometry.dense,Nat.add_mul,Nat.add_assoc] using h

theorem result_query (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D) :
    resultMemory d m "queryGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dq (logicalK d.parameters head.val m) (logicalV d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := result_query_tile d hc m head.val head.isLt n (lane,feature,()) htime
  dsimp only at h
  simpa only [ht] using h

theorem result_key (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.D) :
    resultMemory d m "keyGradient" (head.val*d.N*d.D+time.val*d.D+feature.val)=
      .mk .real (Target.dk (logicalQ d.parameters head.val m) (logicalV d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := result_key_tile d hc m head.val head.isLt n (feature,lane,()) htime
  dsimp only at h
  simpa only [ht] using h

theorem result_value (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E) :
    resultMemory d m "valueGradient" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.dv (logicalQ d.parameters head.val m) (logicalK d.parameters head.val m)
        (logicalG d.parameters head.val m) time feature) := by
  let n : Fin d.NT := ⟨time.val/64,time_owner d time⟩
  let lane : Fin 64 := ⟨time.val%64,Nat.mod_lt _ (by decide)⟩
  have ht : n.val*64+lane.val=time.val := by simpa only [n,lane,Nat.mul_comm] using Nat.div_add_mod time.val 64
  have htime : n.val*64+lane.val<d.N := by rw [ht];exact time.isLt
  have h := result_value_tile d hc m head.val head.isLt n (lane,feature,()) htime
  dsimp only at h
  simpa only [ht] using h

structure WholePost (d : Host.Dimensions) (input result : Memory ℝ) : Prop where
  output : ∀ (head : Fin (d.B*d.H)) (time : Fin d.N) (feature : Fin d.E),
    result "output" (head.val*d.N*d.E+time.val*d.E+feature.val)=
      .mk .real (Target.output (logicalQ d.parameters head.val input) (logicalK d.parameters head.val input)
        (logicalV d.parameters head.val input) time feature)
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
  exterior : ∀ r o, regionCapacity d r≤o → result r o=input r o

theorem whole_correct (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ) :
    ∃ result, executeThree algebra dotPrimitive d (grid .forward d) (grid .intra d) (grid .inter d) m=some result ∧
      WholePost d m result := by
  refine ⟨resultMemory d m,three_execution _ realMasks _ _ _ _ _ _,?_⟩
  refine ⟨result_output d hc m,result_query d hc m,result_key d hc m,result_value d hc m,?_,?_⟩
  · exact (grid_inputs _ _ _ _ _ _).trans (afterIntra_inputs d m)
  · exact modelThree_exterior _ _ _ hc m
#axiomsClean result_output
#axiomsClean result_query
#axiomsClean result_key
#axiomsClean result_value
#axiomsClean whole_correct
#specNonCircular WholePost avoiding [resultMemory,modelThree,executeThree]
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
