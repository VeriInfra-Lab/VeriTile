import bench.tritonbench_g.attention_kernel.Proofs.IOSafetyBasics

namespace VeriTile.Bench.Repairs.AttentionKernel.IOSafety
open VeriTile Triton FP.Structural
open VeriTile.Triton.FP.ProfiledRewrite (engine)
open Whole001 Source Whole Machine
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body_accesses (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current) (guard : Guard p) (active : row<p.length)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts p).body.take 5)
      (current.setReg "start_n" .nat [] (fun _=>row)) := by
  have hk:=hb.key
  have hv:=hb.value
  have hbi:=hb.bias
  have column:=bias_index_bound p guard row active
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    have hquery:=ctx.query
    simp only [TypedSource.Storage.float,TypedSource.Storage.tile,FloatDType.toTileDType,ofFloat,id_eq] at hquery
    simp only [parts,partsAs,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte,List.take]
    simp only [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
    simp [safeOp_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,kptr,vptr,
      ctx.keyPointer,ctx.valuePointer,ctx.startM,ctx.rows,ctx.biasOffset,hquery,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
  all_goals
    refine ⟨?_,?_,?_⟩
    · intro i j hi
      change (decide (row+j.val<length) && Bool.true)=Bool.true at hi
      simp only [Bool.and_true,decide_eq_true_eq] at hi
      simpa [base,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using hk _ hi i.val i.isLt
    · intro i j hi
      change (decide (row+i.val<length) && Bool.true)=Bool.true at hi
      simp only [Bool.and_true,decide_eq_true_eq] at hi
      simpa [base,Nat.add_assoc] using hv _ hi j.val j.isLt
    · intro i hi
      simpa [base,Nat.add_assoc] using hbi _ hi (row/bn) column

theorem body_pure (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds ((parts p).body.drop 5) s := by
  apply pure_trace
  rcases p with ⟨storage,Q,K,V,Bias,Out,q,k,v,bias,o,batches,heads,length,dim,bm,bn,scale⟩
  cases storage
  all_goals
    simp only [parts,partsAs,kernelAs,TypedSource.repaired,TypedSource.FP16.repaired,TypedSource.BF16.repaired,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,↓reduceIte,List.drop,
      List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]
    simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [safeOp_unfold]

theorem body_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current) (guard : Guard p) (active : row<p.length)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts p).body
      (current.setReg "start_n" .nat [] (fun _=>row)) := by
  rw [←List.take_append_drop 5 (parts p).body,traceSafeList_append]
  exact ⟨body_accesses M dot p initial current row a ctx guard active bounds hb,
    fun t _=>body_pure M dot p bounds t⟩

theorem range_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial row a current) (guard : Guard p)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    rangeSafe (engine (withDot M dot)) bounds "start_n" row p.length p.bn (parts p).body current := by
  rw [rangeSafe_unfold]
  simp only [Nat.ne_of_gt guard.bn,if_false]
  by_cases active : row<p.length
  · simp only [active,if_true]
    refine ⟨body_safe M dot p initial current row a ctx guard active bounds hb,?_⟩
    rw [(bodyExec M dot p initial current row a guard.bn ctx).property]
    exact range_safe M dot p initial _ (row+p.bn) _
      (body_context M dot p initial current row a guard.bn ctx) guard bounds hb
  · simp [active]
termination_by p.length-row
decreasing_by have h:=guard.bn;omega

end
end VeriTile.Bench.Repairs.AttentionKernel.IOSafety
