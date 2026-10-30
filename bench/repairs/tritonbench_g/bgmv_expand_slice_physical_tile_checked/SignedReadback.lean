import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedBody

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Readback
open VeriTile.Triton FP.Structural Program Direct.FPValues Setup Body
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem active_lanes : (TileShape.allIndices [256]).filter (fun i => decide (i.1.val<1))=
    [(⟨0,by decide⟩,PUnit.unit)] := by decide

theorem active_lanes_zero : (TileShape.allIndices [256]).filter (fun i => decide (i.1=0))=
    [(⟨0,by decide⟩,PUnit.unit)] := by decide

theorem iteration_memory (M : Algebra α) (reduction : Reduction α) (h : Call)
    (batch row slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int))
    (positiveN : 0<h.N) :
    (iteration M reduction h batch row slot data metadata).val.mem=
      ((entry h batch row data).write "Y" (batch*h.columns+row+h.slice)
        (.mk .real (SourceValues.result M reduction h batch slot row data ⟨0,by decide⟩))).mem := by
  have length := signed_length h positiveN
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  change ((N:Int)+N-1)/N=1 at length
  have nonzero : N≠0 := by change 0<N at positiveN;omega
  have length_cast : ((↑(N+N):Int)-1)/(N:Int)=1 := by simpa only [Int.natCast_add] using length
  have positive_sum : ¬((N:Int)+N<1) := by change 0<N at positiveN;omega
  have positive_source : ¬((N:Int)+(N-1)<0) := by change 0<N at positiveN;omega
  have source_length : ((N:Int)+(N-1))/N=1 := by
    convert length using 1 <;> congr 1 <;> omega
  have ptrCast : (N:Int)*K*slot+row*K=((N*K*slot+row*K:Nat):Int) := by push_cast; rfl
  have ptrToNat : ((N:Int)*K*slot+row*K).toNat=N*K*slot+row*K := by rw [ptrCast,Int.toNat_natCast]
  cases ev <;> cases adds
  all_goals
    simp (config := {maxSteps := 300000}) [iteration,firstEntry,preloop,preguard,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,entry,Data.cells,metadata,
      Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_ediv,length,length_cast,nonzero,positive_sum,positive_source,source_length,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul,
      SourceValues.result,sums,Direct.FPValues.products,loaded,weight,old,State.masked_scatter_eq_filter,active_lanes,active_lanes_zero,
      ptrToNat,Int.toNat_natCast]
    try simp only [State.masked_scatter_eq_filter,active_lanes_zero,List.foldl_cons,List.foldl_nil]
    simp (config := {maxSteps := 100000}) [State.write,Direct.FPValues.products,loaded,weight,bop,Broadcast.leftIndex,Broadcast.rightIndex,ptrToNat,Nat.mul_comm,length_cast,nonzero,positive_sum,positive_source,source_length,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    rw [active_lanes_zero]
    simp only [List.foldl_cons,List.foldl_nil,State.write]
    unfold Direct.FPValues.products FP.Structural.bop
    simp [loaded,weight,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,length_cast,nonzero,positive_sum,positive_source,source_length,←Int.natCast_add,←Int.natCast_mul,-Nat.cast_add,-Nat.cast_mul]
    try rfl

#axiomsClean iteration_memory
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Signed.Readback
