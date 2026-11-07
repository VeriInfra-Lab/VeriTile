import bench.tritonbench_g.rms_rbe_matmul.IO
import bench.tritonbench_g.rbe_triton_transform.IO
import bench.tritonbench_g.rms_matmul_rbe.Proofs.PrefixValues
namespace VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryBridge
open VeriTile.Triton
open VeriTile.Bench.Repairs
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 120000
set_option linter.unusedSimpArgs false
variable (r : RbeTritonTransform.Request) (X Y : RegionName)
abbrev pasted := RotaryIO.source
open FP.Structural
open RbeTritonTransform
private theorem ceil_same (n : Nat) :
    (n:Int)/1024+(if (n:Int)-1024*((n:Int)/1024)=0 then 0 else 1)=((n:Int)+1023)/1024 := by
  split_ifs <;> omega
private theorem nonnegative (n : Nat) : ¬(n:Int)<0 := by omega
private theorem ceil_nonnegative (n : Nat) : ¬(n:Int)+1023<0 := by omega
private def statement : List ComputeStmt → Nat → ComputeStmt
  | st::_,0 => st
  | _::rest,n+1 => statement rest n
  | [],_ => .effectMarker "absent_statement"

variable (s : State ℝ) (M : Algebra ℝ)
theorem step11 :
    step M (statement (pasted r X Y).surfaceBody 11) s =
    step M (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 11) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    statement]
  have constant : (128.0:ℝ)=128 := by norm_num
  cases hpid : s.regs .int [] "pid" <;>
   simp (config := {maxSteps:=20000}) [hpid,constant,IntegerExpr.sameShape,pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,step,evalExpr,evalOp_unfold,evalComputeOp,
    Option.bind,numeric,numericEq,numericLt,integralFloorDiv,integralMod,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
    nonnegative,ceil_nonnegative,ceil_same]
theorem step12 :
    step M (statement (pasted r X Y).surfaceBody 12) s =
    step M (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 12) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    statement]
  have constant : (128.0:ℝ)=128 := by norm_num
  cases hpid : s.regs .int [] "pid" <;>
   simp (config := {maxSteps:=20000}) [hpid,constant,IntegerExpr.sameShape,pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,step,evalExpr,evalOp_unfold,evalComputeOp,
    Option.bind,numeric,numericEq,numericLt,integralFloorDiv,integralMod,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
    nonnegative,ceil_nonnegative,ceil_same]
  congr 1
  funext i
  simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,ceil_same]
theorem step25 :
    step ideal.backend (statement (pasted r X Y).surfaceBody 25) s =
    step ideal.backend (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 25) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    statement]
  have constant : (128.0:ℝ)=128 := by norm_num
  cases hpid : s.regs .int [] "pid" <;>
   simp (config := {maxSteps:=20000}) [hpid,constant,IntegerExpr.sameShape,pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,step,evalExpr,evalOp_unfold,evalComputeOp,
    Option.bind,numeric,numericEq,numericLt,integralFloorDiv,integralMod,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,
    nonnegative,ceil_nonnegative,ceil_same,Operations.backend,ideal]
theorem safe11 (bounds : RegionName → Nat) :
    Safety.stepSafe ideal.backend bounds (statement (pasted r X Y).surfaceBody 11) s ↔
    Safety.stepSafe ideal.backend bounds (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 11) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,statement]
  simp [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp_unfold]
theorem safe12 (bounds : RegionName → Nat) :
    Safety.stepSafe ideal.backend bounds (statement (pasted r X Y).surfaceBody 12) s ↔
    Safety.stepSafe ideal.backend bounds (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 12) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,statement]
  simp [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp_unfold]
theorem safe25 (bounds : RegionName → Nat) :
    Safety.stepSafe ideal.backend bounds (statement (pasted r X Y).surfaceBody 25) s ↔
    Safety.stepSafe ideal.backend bounds (statement (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody 25) s := by
  dsimp only [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,
    RbeTritonTransformCheckedLaunch.source,RbeTritonTransformCheckedLaunch.typedKernel,
    parameters,ComputeKernel.surfaceBody,statement]
  simp [pasted,RotaryIO.source,RmsRbeMatmul.TypedSource.rbeKernel,RbeTritonTransformCheckedLaunch.source,
    RbeTritonTransformCheckedLaunch.typedKernel,parameters,ComputeKernel.surfaceBody,
    Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,Safety.safeOp_unfold]
private def StatementRelated (a b : ComputeStmt) : Prop :=
  (∀ s : State ℝ, step ideal.backend a s = step ideal.backend b s) ∧
  (∀ bounds s, Safety.stepSafe ideal.backend bounds a s ↔ Safety.stepSafe ideal.backend bounds b s)

private theorem statements_related : List.Forall₂ StatementRelated
    (pasted r X Y).surfaceBody
    (RbeTritonTransformCheckedLaunch.source (parameters r X Y)).surfaceBody := by
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun s => step11 r X Y s ideal.backend,fun bounds s => safe11 r X Y s bounds⟩
  apply List.Forall₂.cons
  · exact ⟨fun s => step12 r X Y s ideal.backend,fun bounds s => safe12 r X Y s bounds⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨step25 r X Y,fun bounds s => safe25 r X Y s bounds⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  apply List.Forall₂.cons
  · exact ⟨fun _ => rfl,fun _ _ => Iff.rfl⟩
  exact List.Forall₂.nil

private theorem related_runs {xs ys : List ComputeStmt}
    (h : List.Forall₂ StatementRelated xs ys) :
    ∀ s : State ℝ, run ideal.backend xs s = run ideal.backend ys s := by
  induction h with
  | nil => intro s; rfl
  | cons h _ ih =>
    intro s
    simp only [run,h.1 s]
    cases step ideal.backend _ s <;> simp [ih]

private theorem related_safe {xs ys : List ComputeStmt}
    (h : List.Forall₂ StatementRelated xs ys) :
    ∀ bounds s, Safety.traceSafeList ideal.backend bounds xs s ↔
      Safety.traceSafeList ideal.backend bounds ys s := by
  induction h with
  | nil => intro bounds s; exact Iff.rfl
  | cons h _ ih =>
    intro bounds s
    simp only [Safety.traceSafeList,h.1 s,h.2 bounds s]
    cases step ideal.backend _ s <;> simp [ih]

theorem source_executes (s : State ℝ) :
    exec ideal.backend (pasted r X Y) s =
      exec ideal.backend (RbeTritonTransformCheckedLaunch.source (parameters r X Y)) s :=
  related_runs (statements_related r X Y) s

theorem source_safe (bounds : RegionName → Nat) (s : State ℝ) :
    TraceSafe ideal.backend bounds (pasted r X Y) s ↔
      TraceSafe ideal.backend bounds (RbeTritonTransformCheckedLaunch.source (parameters r X Y)) s :=
  related_safe (statements_related r X Y) bounds s

#axiomsClean source_executes
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.RotaryBridge
