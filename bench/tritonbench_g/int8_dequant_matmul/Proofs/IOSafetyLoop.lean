import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyBody
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyBounds
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IntegerDotBridge

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loadArm_safe (M : Algebra α) (arithmetic : IntegerDot.Arithmetic) (context : IntegerDot.Context)
    (s : Shape) (c : Config) (hasBias unmasked : Bool) (base current : State α) (t : Nat)
    (ctx : Context s c base t current) (bounds : RegionName → Nat)
    (ha : ∀ i : TileIndex [c.BM,c.BK],
      (unmasked=Bool.true ∨ register base .int [c.BK] "rk" (i.2.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (ap s c base t i).2<bounds (ap s c base t i).1)
    (hb : ∀ i : TileIndex [c.BK,c.BN],
      (unmasked=Bool.true ∨ register base .int [c.BK] "rk" (i.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (bp s c base t i).2<bounds (bp s c base t i).1) :
    IntegerDot.traceSafeList M arithmetic context bounds (loadArm s c hasBias unmasked)
      (current.setReg "k" .nat [] (fun _=>t)) := by
  cases unmasked
  · rw [←List.take_append_drop 1 (loadArm s c hasBias Bool.false),IntegerDot.traceSafeList_append]
    constructor
    · apply traceSafe_of_pointwise
      simp [loadArm,body,source,TypedSource.kernel,ComputeKernel.surfaceBody]
      simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
      simp [Safety.safeOp_unfold]
    · intro next executed
      simp [loadArm,body,source,TypedSource.kernel,ComputeKernel.surfaceBody,IntegerDot.run,IntegerDot.step,IntegerDot.arithmeticStep,
        FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,
        Broadcast.leftIndex,Broadcast.rightIndex] at executed
      cases executed
      apply loadPair_safe M arithmetic context s c hasBias Bool.false base _ t
        (by simpa using ctx.a) (by simpa using ctx.b) (by simpa using ctx.rk)
        (by intro _; rfl) bounds ha hb
  · apply loadPair_safe M arithmetic context s c hasBias Bool.true base _ t
      (by simpa using ctx.a) (by simpa using ctx.b) (by simpa using ctx.rk)
      (by intro h; cases h) bounds ha hb

theorem body_safe (M : Algebra α) (arithmetic : IntegerDot.Arithmetic) (context : IntegerDot.Context)
    (s : Shape) (c : Config) (hasBias : Bool) (base current : State α) (t : Nat)
    (ctx : Context s c base t current) (bounds : RegionName → Nat)
    (ha : ∀ i : TileIndex [c.BM,c.BK],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.2.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (ap s c base t i).2<bounds (ap s c base t i).1)
    (hb : ∀ i : TileIndex [c.BK,c.BN],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (bp s c base t i).2<bounds (bp s c base t i).1) :
    IntegerDot.traceSafeList M arithmetic context bounds (body s c hasBias)
      (current.setReg "k" .nat [] (fun _=>t)) := by
  rw [←List.take_append_drop 1 (body s c hasBias),IntegerDot.traceSafeList_append]
  refine ⟨?_,fun next _ => body_tail_safe M arithmetic context s c hasBias bounds next⟩
  change IntegerDot.traceSafeList M arithmetic context bounds
    [.ifThenElse (.constBool (s.K%(c.BK*c.S)==0))
      (loadArm s c hasBias Bool.true) (loadArm s c hasBias Bool.false)] _
  rw [IntegerDot.traceSafeList_cons]
  refine ⟨?_,fun next _ => IntegerDot.traceSafeList_nil _ _ _ _ next⟩
  simp only [IntegerDot.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,true_and]
  by_cases even : s.K%(c.BK*c.S)=0
  · simp only [even,beq_self_eq_true,↓reduceIte]
    exact loadArm_safe M arithmetic context s c hasBias Bool.true base current t ctx bounds
      (fun i _ => ha i (Or.inl even)) (fun i _ => hb i (Or.inl even))
  · simp only [show (s.K%(c.BK*c.S)==0)=Bool.false by simpa using even,↓reduceIte]
    apply loadArm_safe M arithmetic context s c hasBias Bool.false base current t ctx bounds
    · intro i h
      rcases h with h|h
      · cases h
      · exact ha i (Or.inr h)
    · intro i h
      rcases h with h|h
      · cases h
      · exact hb i (Or.inr h)

#axiomsClean loadArm_safe
#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
