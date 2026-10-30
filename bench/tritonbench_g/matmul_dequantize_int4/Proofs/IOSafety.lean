import bench.tritonbench_g.matmul_dequantize_int4.Proofs.IOBounds
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem min_lt_form (a b : Nat) : min a b=if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

theorem prelude_safe (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (positive : 0<steps p)
    (access : IOBounds.Accesses p initial bounds) :
    Safety.traceSafeList M bounds (prelude p) initial := by
  by_cases hno : p.noGroups=Bool.true
  · have hs (j : Fin p.blockColumns) : inputColumn p initial j*p.scaleColumnStride<bounds p.scale := by
      simpa [scaleOffset,groupIndex,hno] using access.s 0 positive (j,())
    have hz (j : Fin p.blockColumns) : inputColumn p initial j/8*p.zeroColumnStride<bounds p.zero := by
      simpa [zeroOffset,groupIndex,hno] using access.z 0 positive (j,())
    simp only [inputColumn,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,min_lt_form] at hs hz
    simp [prelude,source,typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
      bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hno,backend,hs,hz]
  · simp [prelude,source,typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
      bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hno,backend]

theorem body_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (h : Invariant M D p initial k current) (hk : k<steps p)
    (access : IOBounds.Accesses p initial bounds) :
    Safety.traceSafeList (backend M D) bounds (body p)
      (current.setReg "k" .nat [] (fun _ => k)) := by
  have ha := access.a k hk
  have hb := access.q k hk
  have hs := access.s k hk
  have hz := access.z k hk
  by_cases hno : p.noGroups=Bool.true
  all_goals
    simp only [scaleOffset,zeroOffset,groupIndex,hno,if_true,if_false,Bool.false_eq_true] at hs hz
    simp [body,source,typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
      bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hno,backend,
      h.aptrs,h.bptrs,h.acc,h.memory,h.amask,h.sptrs,h.zptrs,h.shift,h.zshift,
      h.bits,h.inf,h.scales,h.zeros,ha,hb,hs,hz]
  all_goals first | exact fun a b u => ha (a,b,u) | exact ⟨fun a b u => ha (a,b,u),fun j=>hs (j,()),fun j=>hz (j,())⟩

theorem epilogue_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (h : Invariant M D p initial (steps p) current)
    (access : IOBounds.Accesses p initial bounds) :
    Safety.traceSafeList (backend M D) bounds (epilogue p) current := by
  simp [epilogue,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,h.pidm,h.pidn,h.acc,h.memory,backend]
  intro r n hr hn
  exact access.c (r,n,()) ⟨hr,hn⟩

theorem source_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (positive : 0<steps p)
    (access : IOBounds.Accesses p initial bounds) :
    TraceSafe (backend M D) bounds (source p) initial := by
  unfold TraceSafe
  rw [source_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe (backend M D) p bounds initial positive access,?_⟩
  intro start ran
  have eq : start=(preludeExecution M D p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution M D p initial).property)
  subst start
  let start := preludeExecution M D p initial
  have invariant := preludeInvariant M D p initial
  obtain ⟨middle,loop,hmid⟩ := loopExecution M D p initial start.val 0 (by omega) invariant
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,invariant.numk,
      Option.bind_some,true_and]
    exact Safety.rangeSafe_of_invariant (backend M D) bounds "k" (steps p) 1 (body p)
      (fun k t=>Invariant M D p initial k t)
      (fun k hk t ht=>body_safe M D p bounds initial t k ht hk access)
      (by
        intro k hk t ht u ran
        have eq : u=(bodyExecution M D p initial t k ht).val :=
          Option.some.inj (ran.symm.trans (bodyExecution M D p initial t k ht).property)
        subst u
        exact bodyInvariant M D p initial t k ht)
      0 start.val invariant
  · simp only [FP.Structural.step,evalOp_unfold,invariant.numk,Option.bind_some]
    change (match FP.Structural.range (backend M D) "k" 0 (steps p) 1 (body p) start.val with
      | some after=>Safety.traceSafeList (backend M D) bounds (epilogue p) after
      | none=>True)
    rw [loop]
    exact epilogue_safe M D p bounds initial middle hmid access

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOSafety
