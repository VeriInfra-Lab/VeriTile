import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.TypedControl
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.IndexCSE
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def candidate (s : Shape) (cu64 off64 : Bool) : ComputeKernel :=
  KernelSource.shared cu64 off64 (TypedExecution.outputRegion s) "x" "cos" "sin" "cu" "offsets"
    s.scalarOffset s.maximum (2*s.half) s.coefficientRows
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    (if s.varlen then 0 else s.maximum*s.heads*s.width) (s.heads*s.width) s.width 1
    s.blockK s.blockM s.offsetTensor s.varlen s.interleaved s.conjugate

theorem candidate_carrier (s : Shape) (cu64 off64 : Bool) :
    candidate s cu64 off64 = candidate s Bool.true Bool.true := by
  simp only [candidate,KernelSource.shared_signed_carrier]

/-- Proof-only slices of the two entire quotations. -/
def continuation (s : Shape) :=
  ((candidate s Bool.true Bool.true).surfaceBody.drop 6).flatMap TypedExecution.ifBody
def interleaved (s : Shape) :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.true)
def split (s : Shape) :=
  ((continuation s).drop 4).flatMap (TypedExecution.branchBody Bool.false)

theorem prefix_same (s : Shape) : (continuation s).take 4 = (TypedExecution.continuation s).take 4 := rfl

theorem split_same (s : Shape) : split s = TypedExecution.splitBody s := rfl

theorem interleaved_body (s : Shape) :
    interleaved s = (interleaved s).take 2 ++ (TypedExecution.interleavedBody s).drop 2 := rfl

theorem indices_rk (M : Algebra α) (s : Shape) (u t : State α)
    (ran : run M ((TypedExecution.continuation s).take 4) u = some t) :
    t.regs .nat [s.blockK] "rk" = some (fun i => i.1.val) := by
  have code : (TypedExecution.continuation s).take 4 =
      (TypedExecution.continuation s).take 2 ++
        ((TypedExecution.continuation s).drop 2).take 2 := rfl
  rw [code,run_append] at ran
  cases h : run M ((TypedExecution.continuation s).take 2) u with
  | none => simp [h] at ran
  | some v =>
    simp only [h,Option.bind_some] at ran
    have last : run M (((TypedExecution.continuation s).drop 2).take 2) v =
        some ((v.setReg "rk" .nat [s.blockK] (fun i => i.1.val)).setReg
          "rk_half" .nat [s.blockK/2] (fun i => i.1.val)) := by
      simp [TypedExecution.continuation,TypedExecution.ifBody,TypedExecution.kernel,
        KernelSource.source,KernelSource.source64_64,ComputeKernel.surfaceBody,List.drop,List.take,
        run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold]
    rw [last] at ran
    cases Option.some.inj ran
    simp

theorem interleaved_prefix_run (M : Algebra α) (s : Shape) (u : State α)
    (rk : u.regs .nat [s.blockK] "rk" = some (fun i => i.1.val)) :
    run M ((TypedExecution.interleavedBody s).take 2) u = run M ((interleaved s).take 2) u := by
  simp [interleaved,continuation,TypedExecution.interleavedBody,TypedExecution.continuation,
    TypedExecution.branchBody,TypedExecution.ifBody,TypedExecution.kernel,candidate,
    KernelSource.source,KernelSource.source64_64,KernelSource.shared,KernelSource.shared64_64,
    ComputeKernel.surfaceBody,List.drop,List.take,List.flatMap,run,FP.Structural.step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,numericLt,natLt,bop,integralFloorDiv,integralMod,Option.bind,rk]

theorem interleaved_prefix_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds ((TypedExecution.interleavedBody s).take 2) u ↔
      Safety.traceSafeList M bounds ((interleaved s).take 2) u := by
  simp [interleaved,continuation,TypedExecution.interleavedBody,TypedExecution.continuation,
    TypedExecution.branchBody,TypedExecution.ifBody,TypedExecution.kernel,candidate,
    KernelSource.source,KernelSource.source64_64,KernelSource.shared,KernelSource.shared64_64,
    ComputeKernel.surfaceBody,List.drop,List.take,List.flatMap,
    Safety.traceSafeList_cons,Safety.traceSafeList_nil,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem interleaved_run (M : Algebra α) (s : Shape) (u : State α)
    (rk : u.regs .nat [s.blockK] "rk" = some (fun i => i.1.val)) :
    run M (TypedExecution.interleavedBody s) u = run M (interleaved s) u := by
  rw [interleaved_body]
  conv_lhs => rw [← List.take_append_drop 2 (TypedExecution.interleavedBody s)]
  rw [run_append,run_append,interleaved_prefix_run M s u rk]

theorem interleaved_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α)
    (rk : u.regs .nat [s.blockK] "rk" = some (fun i => i.1.val)) :
    Safety.traceSafeList M bounds (TypedExecution.interleavedBody s) u ↔
      Safety.traceSafeList M bounds (interleaved s) u := by
  rw [interleaved_body]
  conv_lhs => rw [← List.take_append_drop 2 (TypedExecution.interleavedBody s)]
  simp only [Safety.traceSafeList_append,interleaved_prefix_run M s u rk,interleaved_prefix_safe]

theorem old_tail_code (s : Shape) : (TypedExecution.continuation s).drop 4 =
    [.ifThenElse (.boolNot (.constBool s.interleaved))
      (TypedExecution.splitBody s) (TypedExecution.interleavedBody s)] := rfl

theorem new_tail_code (s : Shape) : (continuation s).drop 4 =
    [.ifThenElse (.boolNot (.constBool s.interleaved)) (split s) (interleaved s)] := rfl

theorem tail_run (M : Algebra α) (s : Shape) (u : State α)
    (rk : u.regs .nat [s.blockK] "rk" = some (fun i => i.1.val)) :
    run M ((TypedExecution.continuation s).drop 4) u = run M ((continuation s).drop 4) u := by
  rw [old_tail_code,new_tail_code,TypedExecution.run_if_not,TypedExecution.run_if_not]
  cases hi : s.interleaved
  · simp only [hi,Bool.false_eq_true,if_false,split_same]
  · simpa only [hi,if_true] using interleaved_run M s u rk

theorem tail_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α)
    (rk : u.regs .nat [s.blockK] "rk" = some (fun i => i.1.val)) :
    Safety.traceSafeList M bounds ((TypedExecution.continuation s).drop 4) u ↔
      Safety.traceSafeList M bounds ((continuation s).drop 4) u := by
  rw [old_tail_code,new_tail_code]
  cases hi : s.interleaved <;>
    simp [Safety.traceSafeList_cons,Safety.traceSafeList_nil,Safety.stepSafe_unfold,
      Safety.safeOp_unfold,evalOp_unfold,hi,split_same,interleaved_safe M s bounds u rk]

theorem continuation_run (M : Algebra α) (s : Shape) (u : State α) :
    run M (TypedExecution.continuation s) u = run M (continuation s) u := by
  conv_lhs => rw [← List.take_append_drop 4 (TypedExecution.continuation s)]
  conv_rhs => rw [← List.take_append_drop 4 (continuation s)]
  rw [run_append,run_append,prefix_same]
  cases ran : run M ((TypedExecution.continuation s).take 4) u with
  | none => simp only [ran,Option.bind_none]
  | some t =>
    simp only [ran,Option.bind_some]
    exact tail_run M s t (indices_rk M s u t ran)

theorem continuation_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds (TypedExecution.continuation s) u ↔
      Safety.traceSafeList M bounds (continuation s) u := by
  conv_lhs => rw [← List.take_append_drop 4 (TypedExecution.continuation s)]
  conv_rhs => rw [← List.take_append_drop 4 (continuation s)]
  simp only [Safety.traceSafeList_append,prefix_same]
  exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right
    (fun ran => tail_safe M s bounds t (indices_rk M s u t ran))))

theorem guarded_run (M : Algebra α) (s : Shape) (u : State α) :
    run M [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] u =
      run M [.ifThen (TypedExecution.guard s) (continuation s)] u := by
  rw [TypedExecution.run_if,TypedExecution.run_if]
  simp only [continuation_run]

theorem guarded_safe (M : Algebra α) (s : Shape) (bounds : RegionName → Nat) (u : State α) :
    Safety.traceSafeList M bounds [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] u ↔
      Safety.traceSafeList M bounds [.ifThen (TypedExecution.guard s) (continuation s)] u := by
  simp only [Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,continuation_safe]
  simp

/-- Exact full-state equality, both signed metadata widths and every original
source flag. All payloads and initial register states remain unrestricted. -/
theorem execution (M : Algebra α) (s : Shape) (cu64 off64 : Bool) (u : State α) :
    exec M (TypedExecution.kernel s cu64 off64) u = exec M (candidate s cu64 off64) u := by
  rw [TypedExecution.kernel_carrier,candidate_carrier]
  have oldBody : (TypedExecution.kernel s Bool.true Bool.true).surfaceBody =
      (TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6 ++
        [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] := rfl
  have newBody : (candidate s Bool.true Bool.true).surfaceBody =
      (TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6 ++
        [.ifThen (TypedExecution.guard s) (continuation s)] := rfl
  change run M (TypedExecution.kernel s Bool.true Bool.true).surfaceBody u =
    run M (candidate s Bool.true Bool.true).surfaceBody u
  rw [oldBody,newBody,run_append,run_append]
  exact congrArg (fun f => (run M ((TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6) u).bind f)
    (funext (guarded_run M s))

/-- The same reached source addresses and masks are safe for both full programs. -/
theorem trace_safe (M : Algebra α) (s : Shape) (cu64 off64 : Bool)
    (bounds : RegionName → Nat) (u : State α) :
    TraceSafe M bounds (TypedExecution.kernel s cu64 off64) u ↔
      TraceSafe M bounds (candidate s cu64 off64) u := by
  rw [TypedExecution.kernel_carrier,candidate_carrier]
  have oldBody : (TypedExecution.kernel s Bool.true Bool.true).surfaceBody =
      (TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6 ++
        [.ifThen (TypedExecution.guard s) (TypedExecution.continuation s)] := rfl
  have newBody : (candidate s Bool.true Bool.true).surfaceBody =
      (TypedExecution.kernel s Bool.true Bool.true).surfaceBody.take 6 ++
        [.ifThen (TypedExecution.guard s) (continuation s)] := rfl
  unfold TraceSafe
  rw [oldBody,newBody]
  simp only [Safety.traceSafeList_append]
  exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right
    (fun _ => guarded_safe M s bounds t)))

#axiomsClean execution
#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.IndexCSE
