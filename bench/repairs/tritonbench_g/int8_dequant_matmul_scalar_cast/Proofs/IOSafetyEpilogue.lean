import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetyCommon
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IntegerDotBridge
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteComplete

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private structure Coordinates (c : Config) (base state : State α) : Prop where
  rm : state.regs .int [c.BM] "rm" = some (register base .int [c.BM] "rm")
  rn : state.regs .int [c.BN] "rn" = some (register base .int [c.BN] "rn")
  pz : state.regs .int [] "pid_z" = some (register base .int [] "pid_z")

private theorem run_invariant (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (code : List ComputeStmt) (Inv : State α → Prop)
    (keep : ∀ st ∈ code, ∀ current next, Inv current →
      IntegerDot.step M a context st current = some next → Inv next)
    (current next : State α) (h : Inv current)
    (ran : IntegerDot.run M a context code current = some next) : Inv next := by
  induction code generalizing current with
  | nil =>
    simp only [IntegerDot.run,Option.some.injEq] at ran
    subst next
    exact h
  | cons st rest ih =>
    rw [IntegerDot.run] at ran
    cases hs : IntegerDot.step M a context st current with
    | none => simp [hs] at ran
    | some intermediate =>
      simp only [hs,Option.bind_some] at ran
      exact ih (fun st hst => keep st (by simp [hst])) intermediate
        (keep st (by simp) current intermediate h hs) ran

private def arithmeticPrefix (s : Shape) (c : Config) (bias : Bool) :=
  (epilogue s c bias).take 2
private def biasStage (s : Shape) (c : Config) (bias : Bool) :=
  ((epilogue s c bias).drop 2).take 1
private def writeStage (s : Shape) (c : Config) (bias : Bool) :=
  (epilogue s c bias).drop 3

private theorem arithmetic_safe (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (bias : Bool)
    (current : State α) (bounds : RegionName → Nat) :
    IntegerDot.traceSafeList M a context bounds (arithmeticPrefix s c bias) current := by
  apply traceSafe_of_pointwise
  simp [arithmeticPrefix,epilogue,source,TypedSource.kernel,TypedSource.skeleton,
    TypedSource.bindRuntime,TypedSource.runtimeScaled,ComputeKernel.surfaceBody]
  all_goals simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals simp [Safety.safeOp_unfold]

private theorem arithmetic_coordinates (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (bias : Bool)
    (base current next : State α) (coords : Coordinates c base current)
    (ran : IntegerDot.run M a context (arithmeticPrefix s c bias) current = some next) :
    Coordinates c base next := by
  apply run_invariant M a context (arithmeticPrefix s c bias) (Coordinates c base) ?_ current next coords ran
  intro st member before after h executed
  simp [arithmeticPrefix,epilogue,source,TypedSource.kernel,TypedSource.skeleton,
    TypedSource.bindRuntime,TypedSource.runtimeScaled,ComputeKernel.surfaceBody] at member
  rcases member with rfl | rfl
  all_goals
    simp only [IntegerDot.step,IntegerDot.arithmeticStep,FP.Structural.step] at executed
    constructor
    · exact (assigned_keeps_reg "rm" .int [c.BM] (by decide) executed).trans h.rm
    · exact (assigned_keeps_reg "rn" .int [c.BN] (by decide) executed).trans h.rn
    · exact (assigned_keeps_reg "pid_z" .int [] (by decide) executed).trans h.pz

private theorem bias_safe (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (mode : context.mode = .scalarCast)
    (s : Shape) (c : Config) (hasBias : Bool)
    (base current : State α) (bounds : RegionName → Nat) (coords : Coordinates c base current)
    (hbias : hasBias = Bool.true → ∀ i : Fin c.BN,
      register base .int [c.BN] "rn" (i, ()) < (s.N : Int) →
      (register base .int [c.BN] "rn" (i, ())).toNat < bounds "bias") :
    IntegerDot.traceSafeList M a context bounds (biasStage s c hasBias) current := by
  cases hasBias
  all_goals by_cases first : register base .int [] "pid_z" PUnit.unit = 0
  all_goals
    simp [biasStage,epilogue,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
      ComputeKernel.surfaceBody,IntegerDot.traceSafeList_cons,IntegerDot.stepSafe_unfold,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
      IntegerDot.step,IntegerDot.arithmeticStep,IntegerDot.run,FP.Structural.step,
      evalExpr,evalComputeOp,evalOp_unfold,bop,numericLt,numericEq,
      coords.rn,coords.pz,first,mode,Broadcast.leftIndex,Broadcast.rightIndex]
  exact fun i _ hi => hbias rfl i hi

private theorem bias_coordinates (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (mode : context.mode = .scalarCast)
    (s : Shape) (c : Config) (hasBias : Bool)
    (base current next : State α) (coords : Coordinates c base current)
    (ran : IntegerDot.run M a context (biasStage s c hasBias) current = some next) :
    Coordinates c base next := by
  cases hasBias
  all_goals by_cases first : register base .int [] "pid_z" PUnit.unit = 0
  all_goals
    simp [biasStage,epilogue,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
      ComputeKernel.surfaceBody,IntegerDot.step,IntegerDot.arithmeticStep,IntegerDot.run,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,bop,numericLt,numericEq,
      coords.rn,coords.pz,first,mode,Broadcast.leftIndex,Broadcast.rightIndex] at ran
  all_goals try (subst next; exact coords)
  all_goals
    cases hv : current.regs .fp16 [c.BM,c.BN] "out" with
    | none => simp [hv,State.setReg] at ran
    | some values =>
      simp [hv,State.setReg] at ran
      cases Option.some.inj ran
      constructor <;> simp [State.setReg,coords.rm,coords.rn,coords.pz]

private theorem write_safe (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (hasBias : Bool)
    (base current : State α) (bounds : RegionName → Nat) (coords : Coordinates c base current)
    (hout : ∀ i : TileIndex [c.BM,c.BN], active s c base i →
      offset s c base i < bounds "C") :
    IntegerDot.traceSafeList M a context bounds (writeStage s c hasBias) current := by
  by_cases split : c.S = 1
  all_goals
    simp [writeStage,epilogue,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
      ComputeKernel.surfaceBody,IntegerDot.traceSafeList_cons,IntegerDot.stepSafe_unfold,
      IntegerDot.step,IntegerDot.arithmeticStep,IntegerDot.run,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
      Option.bind,coords.rm,coords.rn,split,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [active,offset] at hout
  all_goals exact fun i j hi hj => hout (i,j,()) ⟨hi,hj⟩

/-- The epilogue accesses only the enabled bias lanes and active output lanes.
The arithmetic payload and the prior output contents are unrestricted. -/
theorem epilogue_safe (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool)
    (base current : State α) (bounds : RegionName → Nat)
    (ctx : Context s c base (steps s c) current)
    (hbias : hasBias = Bool.true → ∀ i : Fin c.BN,
      register base .int [c.BN] "rn" (i, ()) < (s.N : Int) →
      (register base .int [c.BN] "rn" (i, ())).toNat < bounds "bias")
    (hout : ∀ i : TileIndex [c.BM,c.BN], active s c base i →
      offset s c base i < bounds "C") :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic q)
      (SharedInteger.context q s.M s.N) bounds (epilogue s c hasBias) current := by
  have parts : epilogue s c hasBias = arithmeticPrefix s c hasBias ++
      (biasStage s c hasBias ++ writeStage s c hasBias) := rfl
  rw [parts,IntegerDot.traceSafeList_append]
  refine ⟨arithmetic_safe M _ _ s c hasBias current bounds,?_⟩
  intro afterArithmetic ranArithmetic
  have coordinates := arithmetic_coordinates M _ _ s c hasBias base current afterArithmetic
    ⟨ctx.rm,ctx.rn,ctx.pz⟩ ranArithmetic
  rw [IntegerDot.traceSafeList_append]
  refine ⟨bias_safe M _ _ rfl s c hasBias base afterArithmetic bounds coordinates hbias,?_⟩
  intro afterBias ranBias
  exact write_safe M _ _ s c hasBias base afterBias bounds
    (bias_coordinates M _ _ rfl s c hasBias base afterArithmetic afterBias coordinates ranBias) hout

#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
