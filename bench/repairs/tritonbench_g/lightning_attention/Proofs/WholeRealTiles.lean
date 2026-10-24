import bench.repairs.tritonbench_g.lightning_attention.Proofs.GridInterValues
import bench.repairs.tritonbench_g.lightning_attention.Proofs.RealBridges
namespace VeriTile.Bench.Repairs.LightningAttention.Real
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def afterForward (d : Host.Dimensions) (m : Memory ℝ) : Memory ℝ :=
  gridMemory algebra dotPrimitive .forward d (grid .forward d) m
def afterIntra (d : Host.Dimensions) (m : Memory ℝ) : Memory ℝ :=
  gridMemory algebra dotPrimitive .intra d (grid .intra d) (afterForward d m)
def resultMemory (d : Host.Dimensions) (m : Memory ℝ) : Memory ℝ :=
  modelThree algebra dotPrimitive d (grid .forward d) (grid .intra d) (grid .inter d) m

theorem afterForward_inputs (d : Host.Dimensions) (m : Memory ℝ) : Inputs (afterForward d m) m :=
  grid_inputs _ _ _ _ _ _
theorem afterIntra_inputs (d : Host.Dimensions) (m : Memory ℝ) : Inputs (afterIntra d m) m :=
  (grid_inputs _ _ _ _ _ _).trans (afterForward_inputs d m)

theorem result_output_tile (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (pid : PID) (hp : ValidPID .forward d pid) (n : Fin d.NT) (i : TileIndex [64,d.BM])
    (ht : n.val*64+i.1.val<d.N) :
    resultMemory d m "output" (fwdOffset d.parameters pid n.val i)=
      .mk .real (Target.output (logicalQ d.parameters pid.head m) (logicalK d.parameters pid.head m)
        (logicalV d.parameters pid.head m) ⟨n.val*64+i.1.val,ht⟩
        ⟨pid.channel*d.BM+i.2.1.val,Geometry.channel_active d hc pid.channel hp.2 i.2.1⟩) := by
  unfold resultMemory modelThree
  rw [grid_frame]
  · rw [grid_frame]
    · rw [grid_forward_read algebra dotPrimitive d hc m m (Inputs.refl _) pid hp n.val n.isLt i ht]
      rw [cast_eq,forward_value d.parameters (by change 0<64;decide) (Host.time_cover d) pid m n.val n.isLt i ht (Geometry.channel_active d hc pid.channel hp.2 i.2.1)]
      rfl
    · intro id hid
      simp [Footprint,IntraFoot,RowsFoot]
  · intro id hid
    simp [Footprint,InterFoot,HeadFoot]

theorem result_query_tile (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (i : TileIndex [64,d.D])
    (ht : n.val*64+i.1.val<d.N) :
    resultMemory d m "queryGradient" (head*d.N*d.D+(n.val*64+i.1.val)*d.D+i.2.1.val)=
      .mk .real (Target.dq (logicalK d.parameters head m) (logicalV d.parameters head m)
        (logicalG d.parameters head m) ⟨n.val*64+i.1.val,ht⟩ i.2.1) := by
  let sub : Fin 2 := ⟨i.1.val/32,by have := i.1.isLt; omega⟩
  let lane : Fin 32 := ⟨i.1.val%32,Nat.mod_lt _ (by decide)⟩
  let j : TileIndex [32,d.D] := (lane,i.2.1,())
  have hrow : (n.val*2+sub.val)*32+j.1.val=n.val*64+i.1.val := by
    dsimp [sub,j,lane]
    omega
  have hj : (n.val*2+sub.val)*32+j.1.val<d.N := by rw [hrow];exact ht
  have hread := grid_inter_query_read algebra dotPrimitive d hc m (afterIntra d m)
    (afterIntra_inputs d m) head hh n sub j hj
  have hdot := query_sub_contribution d.parameters head ((n.val*2+sub.val)*32) m
    (queryCarry algebra dotPrimitive d.parameters head m n.val) j hj
  rw [hdot] at hread
  rw [hrow] at hread
  have hprev := grid_intra_query_read algebra dotPrimitive d hc m (afterForward d m)
    (afterForward_inputs d m) ⟨head,n.val⟩ ⟨hh,n.isLt⟩ i ht
  change afterIntra d m "queryGradient" (head*d.N*d.D+(n.val*64+i.1.val)*d.D+i.2.1.val)=_ at hprev
  change resultMemory d m "queryGradient" (head*d.N*d.D+(n.val*64+i.1.val)*d.D+i.2.1.val)=_ at hread
  dsimp only [j] at hread
  rw [hprev] at hread
  have hmath := query_value d.parameters (by change 0<64;decide) rfl (Host.time_cover d) head m n.val n.isLt i ht
  refine hread.trans ?_
  by_cases hz : n.val=0
  · simp only [hz,if_true,cast_eq]
    apply congrArg (Cell.mk .real)
    simpa [hz,queryCarry] using hmath
  · simp only [hz,if_false,cast_eq,add_eq,Cell.read,dite_true]
    apply congrArg (Cell.mk .real)
    simpa only [add_comm] using hmath

theorem result_key_tile (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (i : TileIndex [d.D,64])
    (ht : n.val*64+i.2.1.val<d.N) :
    resultMemory d m "keyGradient" (head*d.N*d.D+(n.val*64+i.2.1.val)*d.D+i.1.val)=
      .mk .real (Target.dk (logicalQ d.parameters head m) (logicalV d.parameters head m)
        (logicalG d.parameters head m) ⟨n.val*64+i.2.1.val,ht⟩ i.1) := by
  let rev : Fin d.NT := ⟨d.NT-1-n.val,by have := n.isLt;omega⟩
  let sub : Fin 2 := ⟨1-i.2.1.val/32,by have := i.2.1.isLt;omega⟩
  let lane : Fin 32 := ⟨i.2.1.val%32,Nat.mod_lt _ (by decide)⟩
  let j : TileIndex [d.D,32] := (i.1,lane,())
  have hrow : Current.reverseRow d.parameters rev.val sub.val+j.2.1.val=n.val*64+i.2.1.val := by
    have hn := n.isLt
    have hi := i.2.1.isLt
    dsimp [Current.reverseRow,Host.Dimensions.parameters,rev,sub,j,lane]
    omega
  have hj : Current.reverseRow d.parameters rev.val sub.val+j.2.1.val<d.N := by rw [hrow];exact ht
  have hread := grid_inter_key_read algebra dotPrimitive d hc m (afterIntra d m)
    (afterIntra_inputs d m) head hh rev sub j hj
  have hdot := key_sub_contribution d.parameters head (Current.reverseRow d.parameters rev.val sub.val) m
    (keyValueCarry algebra dotPrimitive d.parameters head m rev.val) j hj
  rw [hdot] at hread
  rw [hrow] at hread
  have hprev := grid_intra_key_read algebra dotPrimitive d hc m (afterForward d m)
    (afterForward_inputs d m) ⟨head,n.val⟩ ⟨hh,n.isLt⟩ i ht
  change afterIntra d m "keyGradient" (head*d.N*d.D+(n.val*64+i.2.1.val)*d.D+i.1.val)=_ at hprev
  change resultMemory d m "keyGradient" (head*d.N*d.D+(n.val*64+i.2.1.val)*d.D+i.1.val)=_ at hread
  dsimp only [j,rev] at hread
  rw [hprev] at hread
  have hmath := key_value d.parameters (by change 0<64;decide) rfl (Host.time_cover d) head m n.val n.isLt i ht
  refine hread.trans ?_
  by_cases hz : d.NT-1-n.val=0
  · simp only [hz,if_true,cast_eq]
    apply congrArg (Cell.mk .real)
    simpa [Host.Dimensions.parameters,hz,keyValueCarry] using hmath
  · simp only [hz,if_false,cast_eq,add_eq,Cell.read,dite_true]
    apply congrArg (Cell.mk .real)
    simpa only [add_comm] using hmath

theorem result_value_tile (d : Host.Dimensions) (hc : Host.Contract d) (m : Memory ℝ)
    (head : Nat) (hh : head<d.B*d.H) (n : Fin d.NT) (i : TileIndex [64,d.E])
    (ht : n.val*64+i.1.val<d.N) :
    resultMemory d m "valueGradient" (head*d.N*d.E+(n.val*64+i.1.val)*d.E+i.2.1.val)=
      .mk .real (Target.dv (logicalQ d.parameters head m) (logicalK d.parameters head m)
        (logicalG d.parameters head m) ⟨n.val*64+i.1.val,ht⟩ i.2.1) := by
  let rev : Fin d.NT := ⟨d.NT-1-n.val,by have := n.isLt;omega⟩
  let sub : Fin 2 := ⟨1-i.1.val/32,by have := i.1.isLt;omega⟩
  let lane : Fin 32 := ⟨i.1.val%32,Nat.mod_lt _ (by decide)⟩
  let j : TileIndex [32,d.E] := (lane,i.2.1,())
  have hrow : Current.reverseRow d.parameters rev.val sub.val+j.1.val=n.val*64+i.1.val := by
    have hn := n.isLt
    have hi := i.1.isLt
    dsimp [Current.reverseRow,Host.Dimensions.parameters,rev,sub,j,lane]
    omega
  have hj : Current.reverseRow d.parameters rev.val sub.val+j.1.val<d.N := by rw [hrow];exact ht
  have hread := grid_inter_value_read algebra dotPrimitive d hc m (afterIntra d m)
    (afterIntra_inputs d m) head hh rev sub j hj
  have hdot := value_sub_contribution d.parameters head (Current.reverseRow d.parameters rev.val sub.val) m
    (keyValueCarry algebra dotPrimitive d.parameters head m rev.val) j hj
  rw [hdot] at hread
  rw [hrow] at hread
  have hprev := grid_intra_value_read algebra dotPrimitive d hc m (afterForward d m)
    (afterForward_inputs d m) ⟨head,n.val⟩ ⟨hh,n.isLt⟩ i ht
  change afterIntra d m "valueGradient" (head*d.N*d.E+(n.val*64+i.1.val)*d.E+i.2.1.val)=_ at hprev
  change resultMemory d m "valueGradient" (head*d.N*d.E+(n.val*64+i.1.val)*d.E+i.2.1.val)=_ at hread
  dsimp only [j,rev] at hread
  rw [hprev] at hread
  have hmath := value_value d.parameters (by change 0<64;decide) rfl (Host.time_cover d) head m n.val n.isLt i ht
  refine hread.trans ?_
  by_cases hz : d.NT-1-n.val=0
  · simp only [hz,if_true,cast_eq]
    apply congrArg (Cell.mk .real)
    simpa [Host.Dimensions.parameters,hz,keyValueCarry] using hmath
  · simp only [hz,if_false,cast_eq,add_eq,Cell.read,dite_true]
    apply congrArg (Cell.mk .real)
    simpa only [add_comm] using hmath

#axiomsClean result_output_tile
#axiomsClean result_query_tile
#axiomsClean result_key_tile
#axiomsClean result_value_tile
end
end VeriTile.Bench.Repairs.LightningAttention.Real
