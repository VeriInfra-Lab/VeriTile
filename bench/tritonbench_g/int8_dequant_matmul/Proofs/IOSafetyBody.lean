import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyCommon

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def loadArm (s : Shape) (c : Config) (hasBias unmasked : Bool) : List ComputeStmt :=
  match (body s c hasBias)[0]? with
  | some (ComputeStmt.ifThenElse _ yes no) => if unmasked then yes else no
  | _ => []
def loadPair (s : Shape) (c : Config) (hasBias unmasked : Bool) : List ComputeStmt :=
  if unmasked then loadArm s c hasBias unmasked else (loadArm s c hasBias unmasked).drop 1

theorem loadPair_safe (M : Algebra α) (arithmetic : IntegerDot.Arithmetic) (context : IntegerDot.Context)
    (s : Shape) (c : Config) (hasBias unmasked : Bool) (base state : State α) (t : Nat)
    (hA : state.regs .ptr [c.BM,c.BK] "A"=some (ap s c base t))
    (hB : state.regs .ptr [c.BK,c.BN] "B"=some (bp s c base t))
    (hK : state.regs .int [c.BK] "rk"=some (register base .int [c.BK] "rk"))
    (hR : unmasked=Bool.false → state.regs .nat [] "k_remaining"=some (fun _=>s.K-t*(c.BK*c.S)))
    (bounds : RegionName → Nat)
    (ha : ∀ i : TileIndex [c.BM,c.BK],
      (unmasked=Bool.true ∨ register base .int [c.BK] "rk" (i.2.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (ap s c base t i).2<bounds (ap s c base t i).1)
    (hb : ∀ i : TileIndex [c.BK,c.BN],
      (unmasked=Bool.true ∨ register base .int [c.BK] "rk" (i.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (bp s c base t i).2<bounds (bp s c base t i).1) :
    IntegerDot.traceSafeList M arithmetic context bounds (loadPair s c hasBias unmasked) state := by
  apply traceSafe_invariant M arithmetic context bounds (loadPair s c hasBias unmasked)
    (fun current => ∀ d sh name,name≠"a" → name≠"b" → current.regs d sh name=state.regs d sh name)
  · intro st member current keep
    have ptrA : current.regs .ptr [c.BM,c.BK] "A"=some (ap s c base t) :=
      (keep _ _ _ (by decide) (by decide)).trans hA
    have ptrB : current.regs .ptr [c.BK,c.BN] "B"=some (bp s c base t) :=
      (keep _ _ _ (by decide) (by decide)).trans hB
    have rk : current.regs .int [c.BK] "rk"=some (register base .int [c.BK] "rk") :=
      (keep _ _ _ (by decide) (by decide)).trans hK
    have remaining : unmasked=Bool.false → current.regs .nat [] "k_remaining"=some (fun _=>s.K-t*(c.BK*c.S)) :=
      fun h => (keep _ _ _ (by decide) (by decide)).trans (hR h)
    cases unmasked
    all_goals
      simp [loadPair,loadArm,body,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
        ComputeKernel.surfaceBody] at member
      rcases member with rfl|rfl
      all_goals
        simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
        simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,evalOp_unfold,
          ptrA,ptrB,rk,remaining,numericLt,bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals first
      | exact fun a b d => ha (a,b,d) (Or.inl rfl)
      | exact fun a b d => hb (a,b,d) (Or.inl rfl)
      | intro a b d mask evaluated active
        cases Option.some.inj evaluated
        first
        | exact ha (a,b,d) (Or.inr (of_decide_eq_true active))
        | exact hb (a,b,d) (Or.inr (of_decide_eq_true active))
  · intro st member current next keep executed
    cases unmasked
    all_goals
      simp [loadPair,loadArm,body,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
        ComputeKernel.surfaceBody] at member
      rcases member with rfl|rfl
      all_goals
        simp only [IntegerDot.step,IntegerDot.arithmeticStep,FP.Structural.step] at executed
        intro d sh name hna hnb
        first
        | exact (assigned_keeps_reg name d sh hna executed).trans (keep d sh name hna hnb)
        | exact (assigned_keeps_reg name d sh hnb executed).trans (keep d sh name hna hnb)
  · intro _ _ _ _ _
    rfl

/-- Dot/accumulator and pointer updates after both loads have no memory
accesses. Their intermediate numerical values are irrelevant to this fact. -/
theorem body_tail_safe (M : Algebra α) (arithmetic : IntegerDot.Arithmetic) (context : IntegerDot.Context)
    (s : Shape) (c : Config) (hasBias : Bool) (bounds : RegionName → Nat) (state : State α) :
    IntegerDot.traceSafeList M arithmetic context bounds ((body s c hasBias).drop 1) state := by
  apply traceSafe_of_pointwise
  simp [body,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,ComputeKernel.surfaceBody]
  all_goals simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals simp [Safety.safeOp_unfold]

#axiomsClean loadPair_safe
#axiomsClean body_tail_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
