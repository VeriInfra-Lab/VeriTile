import bench.tritonbench_g.triton_matmul.Proofs.SourceSafetySteps
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName→Nat) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (acc : VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.HasAcc p a current)
    (ha : ∀ q (i : AIndex p),i.2.1.val+q*BK p<p.K→aOffset p initial q i<bounds p.A)
    (hb : ∀ q (i : BIndex p),i.1.val+q*BK p<p.K→bOffset p initial q i<bounds p.B) :
    FP.Structural.Safety.rangeIntSafe (backend M D) bounds "k" (k : Int) (p.numKBlocks : Int)
      1 (iteration p st) current := by
  by_cases inside:k<p.numKBlocks
  · have hi:(k : Int)<(p.numKBlocks : Int):=by exact_mod_cast inside
    rw [FP.Structural.Safety.rangeIntSafe_unfold]
    simp only [show ¬(1 : Nat)=0 by decide,if_false,hi,if_true]
    refine ⟨body_safe M D p st bounds initial current k a ctx acc (ha k) (hb k),?_⟩
    rw [(bodyExecution M D p st initial current k a ctx acc).property]
    simpa only [Int.natCast_add,Int.natCast_one] using
      range_safe M D p st bounds initial _ (k+1) (update M D p st initial k a)
        (bodyContext M D p st initial current k a ctx acc).1
        (bodyContext M D p st initial current k a ctx acc).2 ha hb
  · have hi:¬(k : Int)<(p.numKBlocks : Int):=by exact_mod_cast inside
    rw [FP.Structural.Safety.rangeIntSafe_unfold]
    simp only [show ¬(1 : Nat)=0 by decide,if_false,hi]
termination_by p.numKBlocks-k
decreasing_by omega

theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName→Nat) (initial : State α) (hm : 0<BM p) (hn : 0<BN p)
    (hf : SourceControl.first p initial≤SourceControl.rows p) (hk : ceilCount p.K (BK p)=p.numKBlocks)
    (ha : ∀ q (i : AIndex p),i.2.1.val+q*BK p<p.K→aOffset p initial q i<bounds p.A)
    (hb : ∀ q (i : BIndex p),i.1.val+q*BK p<p.K→bOffset p initial q i<bounds p.B)
    (hc : ∀ i : CIndex p,active p initial i→cOffset p initial i<bounds p.C) :
    TraceSafe (backend M D) bounds (VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.typed p st) initial := by
  unfold TraceSafe
  rw [decomposition,List.append_assoc,FP.Structural.Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p st bounds initial,?_⟩
  intro start ran
  have eq:start=(preludeExecution M D p st initial).val:=
    Option.some.inj (ran.symm.trans (preludeExecution M D p st initial).property)
  subst start
  have ctx:=preludeContext M D p st initial hm hn hf
  obtain ⟨middle,he,md,acc⟩:=loopExecution M D p st initial _ ctx.1 ctx.2
  rw [List.singleton_append,FP.Structural.Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · simpa only [FP.Structural.Safety.stepSafe_unfold,FP.Structural.Safety.safeOp_unfold,true_and,evalOp_unfold,ctx.1.blocks,hk,
      Option.bind_some,Option.bind_eq_bind,Option.pure_def] using
      range_safe M D p st bounds initial _ 0 _ ctx.1 ctx.2 ha hb
  · simp only [step,evalOp_unfold,ctx.1.blocks,hk,Option.bind_some,Option.bind_eq_bind,Option.pure_def]
    change (match rangeInt (backend M D) "k" 0 (p.numKBlocks : Int) 1 (iteration p st)
        (preludeExecution M D p st initial).val with
      | some after=>FP.Structural.Safety.traceSafeList (backend M D) bounds (VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.epilogue p st) after
      | none=>True)
    rw [he]
    exact epilogue_safe M D p st bounds initial middle md acc hc

#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
