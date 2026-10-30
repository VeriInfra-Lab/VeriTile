import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndices
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
abbrev Parameters := KernelSource.GemmParameters

def original : Precision → Precision → Parameters → ComputeKernel
  | .half,.half=>Source.gemmHalfHalf | .half,.single=>Source.gemmHalfSingle
  | .single,.half=>Source.gemmSingleHalf | .single,.single=>Source.gemmSingleSingle
def candidate : Precision → Precision → Parameters → ComputeKernel
  | .half,.half=>Source.unmaskedGemmHalfHalf | .half,.single=>Source.unmaskedGemmHalfSingle
  | .single,.half=>Source.unmaskedGemmSingleHalf | .single,.single=>Source.unmaskedGemmSingleSingle

def oldSegment (rowScale output : Precision) (p : Parameters) :=
  ((original rowScale output p).surfaceBody.drop 10).take 9
def newSegment (rowScale output : Precision) (p : Parameters) :=
  ((candidate rowScale output p).surfaceBody.drop 10).take 9

theorem segment_run (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (s : State α) (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.run M a ctx (oldSegment rowScale output p) s=
      IntegerDot.run M a ctx (newSegment rowScale output p) s := by
  have modm (x : Nat) : x%p.rows<p.rows:=Nat.mod_lt x hm
  have modn (x : Nat) : x%p.columns<p.columns:=Nat.mod_lt x hn
  cases rowScale <;> cases output <;>
    cases hp : s.regs .nat [] "pid_m" <;>
    cases hq : s.regs .nat [] "pid_n" <;>
    cases hz : s.regs .nat [] "pid_sp_k"
  all_goals simp (maxSteps:=1500000) [oldSegment,newSegment,original,candidate,
    Source.gemmHalfHalf,Source.gemmHalfSingle,Source.gemmSingleHalf,Source.gemmSingleSingle,
    Source.unmaskedGemmHalfHalf,Source.unmaskedGemmHalfSingle,Source.unmaskedGemmSingleHalf,Source.unmaskedGemmSingleSingle,
    ComputeKernel.surfaceBody,IntegerDot.run,IntegerDot.step_unfold,IntegerDot.arithmeticStep,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    numeric,bop,natLt,numericLt,numericEq,integralFloorDiv,integralMod,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,FloatDType.toTileDType,FloatDType.numericDType,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,hq,hz,modm,modn]

theorem segment_safe (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (bounds : RegionName→Nat) (s : State α)
    (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.traceSafeList M a ctx bounds (oldSegment rowScale output p) s ↔
      IntegerDot.traceSafeList M a ctx bounds (newSegment rowScale output p) s := by
  have modm (x : Nat) : x%p.rows<p.rows:=Nat.mod_lt x hm
  have modn (x : Nat) : x%p.columns<p.columns:=Nat.mod_lt x hn
  cases rowScale <;> cases output <;>
    cases hp : s.regs .nat [] "pid_m" <;>
    cases hq : s.regs .nat [] "pid_n" <;>
    cases hz : s.regs .nat [] "pid_sp_k"
  all_goals simp (maxSteps:=1500000) [oldSegment,newSegment,original,candidate,
    Source.gemmHalfHalf,Source.gemmHalfSingle,Source.gemmSingleHalf,Source.gemmSingleSingle,
    Source.unmaskedGemmHalfHalf,Source.unmaskedGemmHalfSingle,Source.unmaskedGemmSingleHalf,Source.unmaskedGemmSingleSingle,
    ComputeKernel.surfaceBody,IntegerDot.traceSafeList_unfold,IntegerDot.stepSafe_unfold,
    IntegerDot.run,IntegerDot.step_unfold,IntegerDot.arithmeticStep,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    numeric,bop,natLt,numericLt,numericEq,integralFloorDiv,integralMod,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,FloatDType.toTileDType,FloatDType.numericDType,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,hq,hz,modm,modn]

theorem decomposition (rowScale output : Precision) (p : Parameters) :
    (original rowScale output p).surfaceBody=(original rowScale output p).surfaceBody.take 10 ++
      (oldSegment rowScale output p ++ (original rowScale output p).surfaceBody.drop 19) := by
  cases rowScale <;> cases output <;> rfl

theorem candidate_decomposition (rowScale output : Precision) (p : Parameters) :
    (candidate rowScale output p).surfaceBody=(original rowScale output p).surfaceBody.take 10 ++
      (newSegment rowScale output p ++ (original rowScale output p).surfaceBody.drop 19) := by
  cases rowScale <;> cases output <;> rfl

theorem execution (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (s : State α) (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.exec M a ctx (original rowScale output p) s=
      IntegerDot.exec M a ctx (candidate rowScale output p) s := by
  unfold IntegerDot.exec
  rw [decomposition,candidate_decomposition,IntegerDot.run_append,IntegerDot.run_append]
  cases IntegerDot.run M a ctx ((original rowScale output p).surfaceBody.take 10) s with
  | none=>rfl
  | some t=>simp only [Option.bind_some,IntegerDot.run_append,segment_run M a ctx rowScale output p t hm hn]

theorem safety (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (bounds : RegionName→Nat) (s : State α)
    (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.TraceSafe M a ctx bounds (original rowScale output p) s ↔
      IntegerDot.TraceSafe M a ctx bounds (candidate rowScale output p) s := by
  unfold IntegerDot.TraceSafe
  rw [decomposition,candidate_decomposition,IntegerDot.traceSafeList_append,IntegerDot.traceSafeList_append]
  simp only [IntegerDot.traceSafeList_append,segment_safe M a ctx rowScale output p bounds,
    segment_run M a ctx rowScale output p,hm,hn]

#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndices
