import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPBody

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.FPReadback
open VeriTile.Triton FP.Structural FPProgram FPValues FPSetup FPBody
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
      ((entry h batch row data).write "Y" (batch*(h.columns+255)+row+h.slice)
        (.mk .real (result M reduction h batch slot row data ⟨0,by decide⟩))).mem := by
  have length := Kernels.split_length h positiveN
  rcases h with ⟨batches,adapters,N,K,columns,slice,BK,positive,covered,inside,ev,evExact,adds,ity,wty,oty⟩
  have ptrCast : ((N:Int)+255)*BK*slot+row*BK=(((N+255)*BK*slot+row*BK:Nat):Int) := by push_cast; rfl
  have ptrToNat : (((N:Int)+255)*BK*slot+row*BK).toNat=(N+255)*BK*slot+row*BK := by rw [ptrCast,Int.toNat_natCast]
  cases ev <;> cases adds
  all_goals
    dsimp only at length
    simp (config := {maxSteps := 300000}) [iteration,firstEntry,preloop,preguard,
      machine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,entry,Data.cells,metadata,
      Broadcast.leftIndex,Broadcast.rightIndex,←Int.natCast_ediv,length,
      result,sums,FPValues.products,loaded,weight,old,State.masked_scatter_eq_filter,active_lanes,active_lanes_zero,
      ptrToNat,Int.toNat_natCast]
    try simp only [State.masked_scatter_eq_filter,active_lanes_zero,List.foldl_cons,List.foldl_nil]
    simp (config := {maxSteps := 100000}) [State.write,FPValues.products,loaded,weight,bop,Broadcast.leftIndex,Broadcast.rightIndex,ptrToNat,Nat.mul_comm]
    rw [active_lanes_zero]
    simp only [List.foldl_cons,List.foldl_nil,State.write]
    unfold FPValues.products FP.Structural.bop
    simp [loaded,weight,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
    try rfl

#axiomsClean iteration_memory
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.FPReadback
