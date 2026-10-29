import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedMemoryFree
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.AddressBounds

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedBodySafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety AddressBounds TypedSafetyBasics
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem accesses (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : Context M c (Typed.prepare c s a) carry current) (block : Nat)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).body.take 8)
      (current.setReg "i_v" .nat [] (fun _=>block)) := by
  rw [←TypedMemoryFree.body_eq]
  simp only [TypedMemoryFree.body,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.take,List.getElem?_cons_zero,List.getElem?_cons_succ]
  simp only [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold,
    addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Typed.prepare,Config.layout,
    ctx.key,ctx.time,ctx.head,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
  refine ⟨?_,?_,?_,?_⟩
  · intro i j hi hj
    simpa [Nat.add_assoc] using value_bound c a bounds cap "v" (Or.inl rfl) _ _ hi hj
  · intro i j hi hj
    simpa [Nat.add_assoc] using value_bound c a bounds cap "do" (Or.inr rfl) _ _ hi hj
  · intro i j hi hj
    simpa [Nat.add_assoc] using history_bound c a bounds cap "h" (Or.inl rfl) _ _ hi hj
  · intro i j hi hj
    simpa [Nat.add_assoc] using history_bound c a bounds cap "dh" (Or.inr rfl) _ _ hi hj

theorem body_safe (M : Algebra α) (dot : Dot α) (c : Config) (s current : State α) (a : CTA c)
    (carry : Carries α c) (ctx : Context M c (Typed.prepare c s a) carry current) (block : Nat)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts c).body
      (current.setReg "i_v" .nat [] (fun _=>block)) := by
  rw [←List.take_append_drop 8 (parts c).body,traceSafeList_append]
  refine ⟨accesses M dot c s current a carry ctx block bounds cap,?_⟩
  intro t _
  simpa only [TypedMemoryFree.body_eq] using TypedMemoryFree.body_suffix M dot c bounds t

theorem range_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (n start : Nat) (current : {s : State α // Ready c s})
    (ctx : Context M c (Typed.prepare c s a) (accumulated M dot c (Typed.prepare c s a) start) current.val)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    rangeSafe (engine (withDot M dot)) bounds "i_v" start (start+n) 1 (parts c).body current.val := by
  induction n generalizing start current with
  | zero=>rw [rangeSafe_unfold];simp
  | succ n ih=>
    rw [rangeSafe_unfold]
    simp only [show (1:Nat)≠0 from by decide,if_false,
      show start<start+(n+1) from by omega,if_true]
    refine ⟨body_safe M dot c s current.val a _ ctx start bounds cap,?_⟩
    rw [body_runs]
    have next:=body_context M dot c (Typed.prepare c s a) current.val start _ ctx current.property
    simpa only [Nat.add_assoc,Nat.add_comm 1 n] using
      ih (start+1) ⟨body M dot c current.val current.property start,
        body_ready M dot c current.val current.property start⟩ next

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedBodySafety
