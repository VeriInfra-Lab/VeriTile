import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Raw
import VeriTile.Triton.Float.StructuralSafety

/-! Access bounds follow the actual typed source and its existing loop context.
The statement walks use the shared Structural evaluator and introduce no
floating or hardware-conformance premise. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
open VeriTile.Triton FP.Structural FPValue Source
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (Source.prelude p st) s := by
  cases st <;> simp [Source.prelude,Source.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,dtype]

theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : Source.Context p initial current)
    (acc : HasAcc p a current)
    (ha : ∀ i : AIndex p, i.2.1.val+k*BK p<p.K → aOffset p initial k i<bounds p.A)
    (hb : ∀ i : BIndex p, i.1.val+k*BK p<p.K → bOffset p initial k i<bounds p.B) :
    Safety.traceSafeList (backend M D) bounds (Source.body p st)
      (current.setReg "kk" .nat [] (fun _ => k)) := by
  simp only [HasAcc] at acc
  cases st <;> simp [Source.body,Source.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.offk,ctx.am,ctx.bn,acc,backend,dtype]
  all_goals constructor
  all_goals intro i j rest active
  case half.left | e4m3.left =>
    have hk := (Source.mask_iff p.K (k*BK p) j.val).mp active
    have h := ha (i,j,rest) (by simpa only [Nat.add_comm] using hk)
    simp only [if_pos active]
    convert h using 1 <;> dsimp [aOffset] <;> ring
  case half.right | e4m3.right =>
    have hk := (Source.mask_iff p.K (k*BK p) i.val).mp active
    have h := hb (i,j,rest) (by simpa only [Nat.add_comm] using hk)
    simp only [if_pos active]
    convert h using 1 <;> dsimp [bOffset] <;> ring

theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (ctx : Source.Context p initial current)
    (acc : HasAcc p (accumulation M D p st initial) current)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.C) :
    Safety.traceSafeList (backend M D) bounds (Source.epilogue p st) current := by
  simp only [HasAcc] at acc
  cases st <;> simp [Source.epilogue,Source.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.am,ctx.bn,ctx.mm,ctx.nn,acc,backend,dtype]

  all_goals intro i j hr hn
  all_goals simpa [cOffset,aRow,bCol,hr,hn] using hc (i,j,PUnit.unit) ⟨hr,hn⟩

/-- Loop safety follows exactly the source runner's successor context. -/
theorem range_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : Source.Context p initial current)
    (acc : HasAcc p a current)
    (ha : ∀ q (i : AIndex p), i.2.1.val+q*BK p<p.K → aOffset p initial q i<bounds p.A)
    (hb : ∀ q (i : BIndex p), i.1.val+q*BK p<p.K → bOffset p initial q i<bounds p.B) :
    Safety.rangeSafe (backend M D) bounds "kk" k p.numKBlocks 1 (Source.body p st) current := by
  by_cases inside : k<p.numKBlocks
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside, if_true]
    refine ⟨body_safe M D p st bounds initial current k a ctx acc (ha k) (hb k), ?_⟩
    rw [(Source.bodyExecution M D p st initial current k a ctx acc).property]
    exact range_safe M D p st bounds initial _ (k+1) (update M D p st initial k a)
      (Source.bodyContext M D p st initial current k a ctx acc).1
      (Source.bodyContext M D p st initial current k a ctx acc).2 ha hb
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside]
termination_by p.numKBlocks-k
decreasing_by omega

/-- Source access safety for every actual load/store, including all iterations
and masked tails. The hypotheses are only active input/output window bounds. -/
theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Parameters) (st : Storage)
    (bounds : RegionName → Nat) (initial : State α)
    (ha : ∀ q (i : AIndex p), i.2.1.val+q*BK p<p.K → aOffset p initial q i<bounds p.A)
    (hb : ∀ q (i : BIndex p), i.1.val+q*BK p<p.K → bOffset p initial q i<bounds p.B)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.C) :
    FP.Structural.TraceSafe (backend M D) bounds (Source.typed p st) initial := by
  unfold FP.Structural.TraceSafe
  rw [Source.decomposition, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p st bounds initial, ?_⟩
  intro start ran
  have equal : start=(Source.preludeExecution M D p st initial).val :=
    Option.some.inj (ran.symm.trans (Source.preludeExecution M D p st initial).property)
  subst start
  obtain ⟨middle, loop, ctx, acc⟩ := Source.loopExecution M D p st initial _
    (Source.preludeContext M D p st initial).1 (Source.preludeContext M D p st initial).2
  rw [List.singleton_append, Safety.traceSafeList_unfold]
  refine ⟨?_, ?_⟩
  · simpa only [Safety.stepSafe_unfold] using range_safe M D p st bounds initial _ 0 _
      (Source.preludeContext M D p st initial).1 (Source.preludeContext M D p st initial).2 ha hb
  · simp only [FP.Structural.step]
    change (match FP.Structural.range (backend M D) "kk" 0 p.numKBlocks 1 (Source.body p st)
        (Source.preludeExecution M D p st initial).val with
      | some after => Safety.traceSafeList (backend M D) bounds (Source.epilogue p st) after
      | none => True)
    rw [loop]
    exact epilogue_safe M D p st bounds initial middle ctx acc hc

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.SourceSafety
