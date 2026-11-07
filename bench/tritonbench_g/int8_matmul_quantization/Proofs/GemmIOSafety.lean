import bench.tritonbench_g.int8_matmul_quantization.IO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.GemmSafety
open VeriTile.Triton FP.Structural KernelExecution KernelGemm ControlArithmetic
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {F I H : Type} [Inhabited F]

/-- Actual evaluated addresses, before any input-cell or numerical assumptions. -/
structure Bounds (p : Parameters) (initial : State F) (bounds : RegionName→Nat) : Prop where
  a : ∀ t (i : AIndex p),inputInner p initial i.2.1<p.inner-t*(p.blockInner*p.split) →
    aOffset p initial t i<bounds p.input
  b : ∀ t (i : BIndex p),inputInner p initial i.1<p.inner-t*(p.blockInner*p.split) →
    bOffset p initial t i<bounds p.weight
  aScale : ∀ i : Fin p.blockRows,inputRow p initial i<p.rows →
    inputRow p initial i*p.inputScaleStride<bounds p.inputScale
  bScale : ∀ i : Fin p.blockColumns,inputColumn p initial i<p.columns →
    inputColumn p initial i*p.weightScaleStride<bounds p.weightScale
  output : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output

/-- Check both source scale loads after the complete actual pointer prelude. -/
theorem prelude_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (p : Parameters) (initial : State F) (bounds : RegionName→Nat) (hb : Bounds p initial bounds) (valid : firstRow p initial≤blocksM p) :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
      bounds (prelude rowScale output p) initial := by
  have minForm (a b : Nat) : min a b=if a<b then a else b:=by split_ifs <;> omega
  have validNatural:=valid
  simp only [firstRow,blocksM,blocksN] at validNatural
  by_cases small : (p.rows+p.blockRows-1)/p.blockRows -
      initial.pids 0/(p.groupRows*((p.columns+p.blockColumns-1)/p.blockColumns))*p.groupRows < p.groupRows
  all_goals cases rowScale <;> cases output
  all_goals
    simp (maxSteps:=1500000) [prelude,source,KernelSource.gemmHalfHalfKernel, gemmHalfHalfKernel,KernelSource.gemmHalfSingleKernel, gemmHalfSingleKernel,
      KernelSource.gemmSingleHalfKernel, gemmSingleHalfKernel,KernelSource.gemmSingleSingleKernel, gemmSingleSingleKernel,ComputeKernel.surfaceBody,
      IntegerDot.traceSafeList_unfold,IntegerDot.stepSafe_unfold,SharedInteger.step_eq,
      KernelExecution.step,KernelExecution.run,localStep,FP.Structural.step,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,numericEq,natLt,bop,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,integralFloorDiv,integralMod,
      FloatDType.toTileDType,FloatDType.numericDType,NumericDType.sub]
  all_goals try simp only [← Int.natCast_mul,← Int.natCast_ediv,
    signed_group_size _ _ _ validNatural,tmod_nat,tdiv_nat,Int.ofNat_lt,small,if_true,if_false]
  all_goals try simp only [← Int.natCast_emod,← Int.natCast_mul,← Int.natCast_add,
    ← Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]
  all_goals try simp only [tmod_nat,← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals
    simpa [inputRow,inputColumn,rowBlock,columnBlock,firstRow,rowsInGroup,blocksM,blocksN,minForm,small] using
      And.intro hb.aScale hb.bScale

/-- Actual masked integer operand loads and the carry-fed update, at every
source loop invariant. Arithmetic-only tail statements cannot add accesses. -/
theorem body_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale : Precision)
    (p : Parameters) (initial current : State F) (t : Nat)
    (h : Invariant M carrier ops plans rowScale p initial t current)
    (bounds : RegionName→Nat) (hb : Bounds p initial bounds) :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
      bounds (body p) (current.setReg "k" .int [] (fun _=>(t : Int))) := by
  simp (maxSteps:=1500000) [body,source,KernelSource.gemmHalfHalfKernel, gemmHalfHalfKernel,ComputeKernel.surfaceBody,
    IntegerDot.traceSafeList_unfold,IntegerDot.stepSafe_unfold,SharedInteger.step_eq,
    KernelExecution.step,localStep_dot,localStep_load,localStep_pointer,accumulate,FP.Structural.step,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,numericLt,numericEq,natLt,bop,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,integralFloorDiv,integralMod,
    FloatDType.toTileDType,FloatDType.numericDType,h.aptrs,h.bptrs,h.offsk,h.acc,h.pidm,h.pidn,h.pids,h.memory]
  exact ⟨fun a b u=>hb.a t (a,b,u),fun a b u=>hb.b t (a,b,u)⟩

/-- Both final store/atomic branches check the original active output mask. -/
theorem epilogue_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (p : Parameters) (initial current : State F)
    (h : Invariant M carrier ops plans rowScale p initial (KernelGemm.tripCount p) current)
    (bounds : RegionName→Nat) (hb : Bounds p initial bounds) :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic carrier ops zero plans) (SharedInteger.context plans)
      bounds (epilogue rowScale output p) current := by
  cases rowScale <;> cases output <;> by_cases hs : p.split=1
  all_goals
    simp (maxSteps:=1500000) [epilogue,source,KernelSource.gemmHalfHalfKernel, gemmHalfHalfKernel,KernelSource.gemmHalfSingleKernel, gemmHalfSingleKernel,
      KernelSource.gemmSingleHalfKernel, gemmSingleHalfKernel,KernelSource.gemmSingleSingleKernel, gemmSingleSingleKernel,ComputeKernel.surfaceBody,
      IntegerDot.traceSafeList_unfold,IntegerDot.stepSafe_unfold,SharedInteger.step_eq,
      KernelExecution.step,KernelExecution.run,localStep,FP.Structural.step,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,atomicAdd,atomicAddition,numeric,numericLt,numericEq,natLt,bop,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      FloatDType.toTileDType,FloatDType.numericDType,h.pidm,h.pidn,h.acc,h.ascale,h.bscale,hs]
  all_goals simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  all_goals repeat' constructor
  all_goals first
    | (intro row column rowBound columnBound
       exact hb.output (row,column,PUnit.unit) ⟨rowBound,columnBound⟩)
    | (split <;> trivial)

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.PublicIO.GemmSafety
