import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawSafety

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSafety
open VeriTile.Triton FP.Structural FPValue
noncomputable section
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : RawSource.Context p initial current)
    (acc : RawSource.HasAcc p a current)
    (ha : ∀ q (i : AIndex p), RawSource.innerMask p q i.2.1 = Bool.true →
      RawSource.addressA p initial q i < bounds p.A)
    (hb : ∀ q (i : BIndex p), RawSource.innerMask p q i.1 = Bool.true →
      RawSource.addressB p initial q i < bounds p.B) :
    Safety.rangeIntSafe (backend M D) bounds "k" (k : Int) (RawSource.blockCount p : Int) 1
      (RawSource.body p st) current := by
  by_cases inside : k < RawSource.blockCount p
  · rw [Safety.rangeIntSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide,if_false,Int.ofNat_lt,inside,if_true]
    refine ⟨body_safe M D p st bounds initial current k a ctx acc (ha k) (hb k),?_⟩
    rw [(RawSource.bodyExecution M D p st initial current k a ctx acc).property]
    simpa only [Int.natCast_add,Int.natCast_one] using
      range_safe M D p st bounds initial _ (k+1) (RawSource.update M D p st initial k a)
        (RawSource.bodyContext M D p st initial current k a ctx acc).1
        (RawSource.bodyContext M D p st initial current k a ctx acc).2 ha hb
  · rw [Safety.rangeIntSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide,if_false,Int.ofNat_lt,inside]
termination_by RawSource.blockCount p-k
decreasing_by omega

theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial : State α)
    (ha : ∀ q (i : AIndex p), RawSource.innerMask p q i.2.1 = Bool.true →
      RawSource.addressA p initial q i < bounds p.A)
    (hb : ∀ q (i : BIndex p), RawSource.innerMask p q i.1 = Bool.true →
      RawSource.addressB p initial q i < bounds p.B)
    (hc : ∀ i : CIndex p, RawSource.active p initial i = Bool.true →
      RawSource.outputOffset p initial i < bounds p.C) :
    TraceSafe (backend M D) bounds (Source.typed p st) initial := by
  unfold TraceSafe
  rw [RawSource.decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p st bounds initial,?_⟩
  intro start ran
  have equal : start = (RawSource.preludeExecution M D p st initial).val :=
    Option.some.inj (ran.symm.trans (RawSource.preludeExecution M D p st initial).property)
  subst start
  have context := (RawSource.preludeContext M D p st initial).1
  have hblocks := context.blocks
  have hstop : (RawSource.blockCount p : Int) = RawSource.signedCeil p.K (BK p) :=
    Int.toNat_of_nonneg (RawSource.signedCeil_nonnegative _ _)
  obtain ⟨middle,loop,ctx,acc⟩ := RawSource.loopExecution M D p st initial _ context
    (RawSource.preludeContext M D p st initial).2
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · have h := range_safe M D p st bounds initial _ 0 _ context
      (RawSource.preludeContext M D p st initial).2 ha hb
    rw [hstop] at h
    simpa [Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeOp,Safety.safeOp_unfold,
      evalExpr,evalOp_unfold,hblocks] using h
  · simp only [step,evalOp_unfold,hblocks]
    change (match rangeInt (backend M D) "k" 0 (RawSource.signedCeil p.K (BK p)) 1
        (RawSource.body p st) (RawSource.preludeExecution M D p st initial).val with
      | some t => Safety.traceSafeList (backend M D) bounds (RawSource.epilogue p st) t
      | none => True)
    rw [loop]
    exact epilogue_safe M D p st bounds initial middle ctx acc hc

#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSafety
