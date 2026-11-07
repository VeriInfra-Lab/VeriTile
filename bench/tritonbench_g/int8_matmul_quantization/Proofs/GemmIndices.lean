import bench.tritonbench_g.int8_matmul_quantization.Proofs.GemmIOResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndices
open VeriTile Triton FP.Structural ControlArithmetic
attribute [local simp 2000] FP.Structural.eval_truncDiv FP.Structural.eval_truncMod
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
abbrev Parameters := KernelSource.GemmParameters

def original : Precision → Precision → Parameters → ComputeKernel
  | .half,.half=>Source.gemmHalfHalfKernel | .half,.single=>Source.gemmHalfSingleKernel
  | .single,.half=>Source.gemmSingleHalfKernel | .single,.single=>Source.gemmSingleSingleKernel
def candidate : Precision → Precision → Parameters → ComputeKernel
  | .half,.half=>Source.unmaskedGemmHalfHalfKernel | .half,.single=>Source.unmaskedGemmHalfSingleKernel
  | .single,.half=>Source.unmaskedGemmSingleHalfKernel | .single,.single=>Source.unmaskedGemmSingleSingleKernel

def oldSegment (rowScale output : Precision) (p : Parameters) :=
  ((original rowScale output p).surfaceBody.drop 21).take 9
def newSegment (rowScale output : Precision) (p : Parameters) :=
  ((candidate rowScale output p).surfaceBody.drop 21).take 9

def addressPrefix (p : Parameters) := (oldSegment .half .half p).take 7
def oldLoads (rowScale : Precision) (p : Parameters) := (oldSegment rowScale .half p).drop 7
def newLoads (rowScale : Precision) (p : Parameters) := (newSegment rowScale .half p).drop 7

def GoodOffsets (p : Parameters) (s : State α) : Prop :=
  ∃ a : Values α .int [p.blockRows], ∃ b : Values α .int [p.blockColumns],
    s.regs .int [p.blockRows] "offs_am"=some a ∧ s.regs .int [p.blockColumns] "offs_bn"=some b ∧
      (∀ i,a i<(p.rows : Int)) ∧ (∀ i,b i<(p.columns : Int))

theorem old_segment_split (rowScale output : Precision) (p : Parameters) :
    oldSegment rowScale output p=addressPrefix p++oldLoads rowScale p := by
  cases rowScale <;> cases output <;> rfl

theorem new_segment_split (rowScale output : Precision) (p : Parameters) :
    newSegment rowScale output p=addressPrefix p++newLoads rowScale p := by
  cases rowScale <;> cases output <;> rfl

theorem prefix_good (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (p : Parameters) (s t : State α) (hm : 0<p.rows) (hn : 0<p.columns)
    (ran : IntegerDot.run M a ctx (addressPrefix p) s=some t) : GoodOffsets p t := by
  have modm (x : Int) : x.tmod (p.rows : Int)<(p.rows : Int):=Int.tmod_lt_of_pos x (by exact_mod_cast hm)
  have modn (x : Int) : x.tmod (p.columns : Int)<(p.columns : Int):=Int.tmod_lt_of_pos x (by exact_mod_cast hn)
  cases mode : ctx.mode <;>
    cases hp : s.regs .int [] "pid_m" <;> cases hq : s.regs .int [] "pid_n" <;>
    cases hz : s.regs .int [] "pid_sp_k"
  all_goals simp only [addressPrefix,oldSegment,original,Source.gemmHalfHalfKernel,
    KernelSource.gemmHalfHalfKernel,gemmHalfHalfKernel,ComputeKernel.surfaceBody,
    List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero] at ran
  all_goals simp [IntegerDot.run,IntegerDot.step_unfold,IntegerDot.arithmeticStep,
    IntegerDot.addExpression,IntegerDot.accumulatorStep,mode,FP.Structural.step,evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,natLt,numericLt,numericEq,
    integralFloorDiv,integralMod,NumericDType.sub,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,hq,hz] at ran
  all_goals cases ran
  all_goals unfold GoodOffsets
  all_goals
    let av : Values α .int [p.blockRows] := fun i =>
      (((s.regs .int [] "pid_m").getD (fun _=>0) PUnit.unit)*(p.blockRows : Int)+(i.1.val : Int)).tmod (p.rows : Int)
    let bv : Values α .int [p.blockColumns] := fun i =>
      (((s.regs .int [] "pid_n").getD (fun _=>0) PUnit.unit)*(p.blockColumns : Int)+(i.1.val : Int)).tmod (p.columns : Int)
    refine ⟨av,bv,?_,?_,?_,?_⟩
    all_goals try simp [av,bv,hp,hq]
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,modm,modn]
  all_goals rfl

theorem loads_run (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale : Precision) (p : Parameters) (s : State α) (good : GoodOffsets p s) :
    IntegerDot.run M a ctx (oldLoads rowScale p) s=IntegerDot.run M a ctx (newLoads rowScale p) s := by
  rcases good with ⟨av,bv,ha,hb,ab,bb⟩
  cases rowScale <;> cases hp : s.regs .ptr [p.blockRows] "as_ptrs" <;>
    cases hq : s.regs .ptr [p.blockColumns] "bs_ptrs"
  all_goals simp only [oldLoads,newLoads,oldSegment,newSegment,original,candidate,
    Source.gemmHalfHalfKernel,KernelSource.gemmHalfHalfKernel,gemmHalfHalfKernel,
    Source.gemmSingleHalfKernel,KernelSource.gemmSingleHalfKernel,gemmSingleHalfKernel,
    Source.unmaskedGemmHalfHalfKernel,unmaskedGemmHalfHalfKernel,
    Source.unmaskedGemmSingleHalfKernel,unmaskedGemmSingleHalfKernel,ComputeKernel.surfaceBody,
    List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  all_goals simp [IntegerDot.run,IntegerDot.step_unfold,IntegerDot.arithmeticStep,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,numericLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,FloatDType.toTileDType,FloatDType.numericDType,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,hq,ha,hb,ab,bb]

theorem segment_run (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (s : State α) (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.run M a ctx (oldSegment rowScale output p) s=
      IntegerDot.run M a ctx (newSegment rowScale output p) s := by
  rw [old_segment_split,new_segment_split,IntegerDot.run_append,IntegerDot.run_append]
  cases ran : IntegerDot.run M a ctx (addressPrefix p) s with
  | none => rfl
  | some next =>
    simp only [Option.bind_some]
    exact loads_run M a ctx rowScale p next (prefix_good M a ctx p s next hm hn ran)

theorem loads_safe (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale : Precision) (p : Parameters) (bounds : RegionName→Nat) (s : State α)
    (good : GoodOffsets p s) :
    IntegerDot.traceSafeList M a ctx bounds (oldLoads rowScale p) s ↔
      IntegerDot.traceSafeList M a ctx bounds (newLoads rowScale p) s := by
  rcases good with ⟨av,bv,ha,hb,ab,bb⟩
  cases rowScale <;> cases hp : s.regs .ptr [p.blockRows] "as_ptrs" <;>
    cases hq : s.regs .ptr [p.blockColumns] "bs_ptrs"
  all_goals simp only [oldLoads,newLoads,oldSegment,newSegment,original,candidate,
    Source.gemmHalfHalfKernel,KernelSource.gemmHalfHalfKernel,gemmHalfHalfKernel,
    Source.gemmSingleHalfKernel,KernelSource.gemmSingleHalfKernel,gemmSingleHalfKernel,
    Source.unmaskedGemmHalfHalfKernel,unmaskedGemmHalfHalfKernel,
    Source.unmaskedGemmSingleHalfKernel,unmaskedGemmSingleHalfKernel,ComputeKernel.surfaceBody,
    List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  all_goals simp [IntegerDot.traceSafeList_unfold,IntegerDot.stepSafe_unfold,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    IntegerDot.run,IntegerDot.step_unfold,IntegerDot.arithmeticStep,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,numeric,bop,numericLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,FloatDType.toTileDType,FloatDType.numericDType,
    Broadcast.leftIndex,Broadcast.rightIndex,hp,hq,ha,hb,ab,bb]

theorem segment_safe (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (bounds : RegionName→Nat) (s : State α)
    (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.traceSafeList M a ctx bounds (oldSegment rowScale output p) s ↔
      IntegerDot.traceSafeList M a ctx bounds (newSegment rowScale output p) s := by
  rw [old_segment_split,new_segment_split,IntegerDot.traceSafeList_append,IntegerDot.traceSafeList_append]
  constructor
  · rintro ⟨head,tail⟩
    refine ⟨head,?_⟩
    intro next ran
    exact (loads_safe M a ctx rowScale p bounds next (prefix_good M a ctx p s next hm hn ran)).mp (tail next ran)
  · rintro ⟨head,tail⟩
    refine ⟨head,?_⟩
    intro next ran
    exact (loads_safe M a ctx rowScale p bounds next (prefix_good M a ctx p s next hm hn ran)).mpr (tail next ran)

theorem decomposition (rowScale output : Precision) (p : Parameters) :
    (original rowScale output p).surfaceBody=(original rowScale output p).surfaceBody.take 21 ++
      (oldSegment rowScale output p ++ (original rowScale output p).surfaceBody.drop 30) := by
  cases rowScale <;> cases output <;> rfl

theorem candidate_decomposition (rowScale output : Precision) (p : Parameters) :
    (candidate rowScale output p).surfaceBody=(original rowScale output p).surfaceBody.take 21 ++
      (newSegment rowScale output p ++ (original rowScale output p).surfaceBody.drop 30) := by
  cases rowScale <;> cases output <;> rfl

theorem execution (M : Algebra α) (a : IntegerDot.Arithmetic) (ctx : IntegerDot.Context)
    (rowScale output : Precision) (p : Parameters) (s : State α) (hm : 0<p.rows) (hn : 0<p.columns) :
    IntegerDot.exec M a ctx (original rowScale output p) s=
      IntegerDot.exec M a ctx (candidate rowScale output p) s := by
  unfold IntegerDot.exec
  rw [decomposition,candidate_decomposition,IntegerDot.run_append,IntegerDot.run_append]
  cases IntegerDot.run M a ctx ((original rowScale output p).surfaceBody.take 21) s with
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
