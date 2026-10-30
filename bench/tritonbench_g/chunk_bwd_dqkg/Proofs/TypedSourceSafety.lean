import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedGrid
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.AddressBounds
import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedMemoryFree

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole Typed
open FP.Structural.Safety AddressBounds TypedSafetyBasics
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Constructive execution of only the six address-setup statements. -/
def addressSetup (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    {t : State α // run (engine (withDot M dot)) ((kernel c).surfaceBody.take 6) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.take]
  simp [run,step.eq_2,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    numeric,natLt,bop,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem address_setup_safe (M : Algebra α) (dot : Dot α) (c : Config)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds ((kernel c).surfaceBody.take 6) s := by
  apply pure_trace
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.take,List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]
  simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  simp only [safeBlockPtr1,safeBlockPtr2]
  simp [safeOp_unfold]

theorem entry_loads_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds
      (((kernel c).surfaceBody.drop 6).take 3) (addressSetup M dot c (Typed.prepare c s a)).val := by
  have hg := gate_bound c a bounds cap
  have hl : min (a.time.val*64+64) c.T-1<c.T := by have := T_pos c a; omega
  simp only [kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,List.drop,List.take]
  simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Typed.prepare,addressSetup,
    Broadcast.leftIndex,Broadcast.rightIndex,TileShape.indexToList,Nat.add_assoc]
  have hmin (u v : Nat) : min u v=(if u<v then u else v) := by split_ifs <;> omega
  exact ⟨fun i hi=>hg _ hi,by simpa [Nat.add_assoc,hmin] using hg _ hl⟩

theorem entry_prefix_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).before.take 9) (Typed.prepare c s a) := by
  have split : (parts c).before.take 9 = (kernel c).surfaceBody.take 6 ++
      (((kernel c).surfaceBody.drop 6).take 3) := rfl
  rw [split,traceSafeList_append]
  refine ⟨address_setup_safe M dot c bounds _,?_⟩
  intro t ht
  have eq := Option.some.inj (ht.symm.trans (addressSetup M dot c (Typed.prepare c s a)).property)
  subst t
  exact entry_loads_safe M dot c s a bounds cap

theorem entry_suffix_safe (M : Algebra α) (dot : Dot α) (c : Config)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (engine (withDot M dot)) bounds ((parts c).before.drop 9) s :=
  TypedMemoryFree.entry_suffix M dot c bounds s

theorem entry_safe (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : CTA c)
    (bounds : RegionName → Nat) (cap : Capacity c bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts c).before (Typed.prepare c s a) := by
  rw [←List.take_append_drop 9 (parts c).before,traceSafeList_append]
  exact ⟨entry_prefix_safe M dot c s a bounds cap,fun t _=>entry_suffix_safe M dot c bounds t⟩

end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TypedSourceSafety
