import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedMemoryFree
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.AddressBounds

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedPostSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety AddressBounds TypedSafetyBasics
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem accesses (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : Context M c (Typed.prepare c s a) carry current)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).after.take 4) current := by
  change traceSafeList (engine (withDot M dot)) bounds
    (((kernel c).surfaceBody.drop 15).take 4) current
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.take]
  simp only [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold,
    addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Typed.prepare,Config.layout,
    ctx.key,ctx.time,ctx.head,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
  constructor
  · intro i j hi hj
    simpa [Nat.add_assoc] using query_bound c a bounds cap "k" (Or.inr (Or.inl rfl)) _ _ hi hj
  · intro i j hi hj
    simpa [Nat.add_assoc] using query_bound c a bounds cap "q" (Or.inl rfl) _ _ hi hj

theorem calculate (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : Context M c (Typed.prepare c s a) carry current)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).after.take 17) current := by
  have split : (parts c).after.take 17 = (parts c).after.take 4 ++
      (((kernel c).surfaceBody.drop 19).take 13) := rfl
  rw [split,traceSafeList_append]
  exact ⟨accesses M dot c s current a carry ctx bounds cap,
    fun t _=>TypedMemoryFree.post_suffix M dot c bounds t⟩

theorem stores_safe (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : AnswerContext M dot c (Typed.prepare c s a) carry current)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).after.drop 17) current := by
  change traceSafeList (engine (withDot M dot)) bounds ((kernel c).surfaceBody.drop 32) current
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop]
  simp [traceSafeList_unfold,stepSafe_unfold,safeAddress_unfold,safeMask_unfold,
    safeExpr_unfold,safeCompute_unfold,safeOp_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,
    resolvePrecision,ComputeDType.eraseDType,ofFloat,toFloat,
    ctx.dq,ctx.dk,ctx.dg,ctx.pq,ctx.pk,ctx.pg,masked_writes_regs,
    qPointer,gPointer,Typed.prepare,Config.layout,TileShape.indexToList,Nat.add_assoc]
  refine ⟨?_,?_,?_⟩
  · intro i j hi hj
    simpa [Nat.add_assoc] using query_bound c a bounds cap "dq" (Or.inr (Or.inr (Or.inl rfl))) _ _ hi hj
  · intro i j hi hj
    simpa [Nat.add_assoc] using query_bound c a bounds cap "dk" (Or.inr (Or.inr (Or.inr rfl))) _ _ hi hj
  · intro i hi
    simpa [Nat.add_assoc] using output_gate_bound c a bounds cap _ hi

theorem after_safe (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : Context M c (Typed.prepare c s a) carry current)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts c).after current := by
  rw [←List.take_append_drop 17 (parts c).after,traceSafeList_append]
  refine ⟨calculate M dot c s current a carry ctx bounds cap,?_⟩
  intro t ht
  have eq:=Option.some.inj (ht.symm.trans (contextPost M dot c _ current carry ctx).property)
  subst t
  exact stores_safe M dot c s _ a carry (contextPost_context M dot c _ current carry ctx) bounds cap

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedPostSafety
