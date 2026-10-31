import bench.tritonbench_g.ksoftmax_triton.Proofs.RawKernel
import VeriTile.Triton.Float.StructuralSafety

/-! Safety follows the selected forward/backward quotation, actual causal
read mask, optional mask-address branch and unconditional logical output mask. -/
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section
set_option maxHeartbeats 14000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem source_safe {A : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (Q K D : Nat) (positive : 0 < D) (s : State A) (bounds : RegionName → Nat)
    (left : ∀ i : Fin D, Forward.readActive K (s.pids 1) op.causal i.val = Bool.true →
      Native.flat Q K (s.pids 0) (s.pids 1) i.val < bounds "left")
    (right : ∀ i : Fin D, op.readsRight = Bool.true →
      Forward.readActive K (s.pids 1) op.causal i.val = Bool.true →
      rightAddress op Q K (s.pids 0) (s.pids 1) i.val < bounds "right")
    (output : ∀ i : Fin D, i.val < K → Native.flat Q K (s.pids 0) (s.pids 1) i.val < bounds "dst") :
    FP.Structural.TraceSafe (FP.ProfiledRewrite.engine M) bounds (KernelExecution.source op Q K D) s := by
  cases op with
  | forward log causal mode =>
    cases mode <;> cases log <;> cases causal
    all_goals simp (maxSteps := 2200000) [FP.Structural.TraceSafe, KernelExecution.source,
      forward, Forward.Typed.fp32Original,VeriTile.Bench.Repairs.Ksoftmax.forward, backward, ComputeKernel.surfaceBody,
      Forward.MaskMode.python, Forward.MaskMode.present, Operation.causal, Operation.readsRight,
      Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
      Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
      Safety.maskActive, run, step, stepAlg, evalExpr, evalComputeOp, evalOp_unfold,
      FP.ProfiledRewrite.engine, Algebra.withDefaultPrecision, resolvePrecision, State.setReg, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, natLt, natLe, numericLt, numericLe, numericEq, Region.cast, ofFloat, toFloat,
      TileShape.axisDim, positive, Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals
      simp [Operation.causal, Operation.readsRight, Forward.MaskMode.present,
        Forward.readActive, Bool.and_eq_true, decide_eq_true_eq, rightAddress,
        Forward.maskAddress, Native.flat] at left right output
    all_goals repeat' constructor
    all_goals first
      | exact output
      | exact left
      | exact right
      | exact fun i active => right i rfl active
      | exact fun i hk hc => left i ⟨hk, hc⟩
      | exact fun i hk hc => right i rfl ⟨hk, hc⟩
  | backward log causal =>
    cases log <;> cases causal
    all_goals simp (maxSteps := 2200000) [FP.Structural.TraceSafe, KernelExecution.source,
      backward, ComputeKernel.surfaceBody,
      Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
      Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
      Safety.maskActive, run, step, stepAlg, evalExpr, evalComputeOp, evalOp_unfold,
      FP.ProfiledRewrite.engine, Algebra.withDefaultPrecision, resolvePrecision, State.setReg, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, natLt, natLe, numericLt, numericLe, numericEq, Region.cast, ofFloat, toFloat,
      TileShape.axisDim, positive, Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals
      simp [Operation.causal, Operation.readsRight, Forward.readActive, Bool.and_eq_true,
        decide_eq_true_eq, rightAddress, Native.flat] at left right output
    all_goals repeat' constructor
    all_goals first
      | exact output
      | exact left
      | exact right
      | exact fun i active => right i rfl active
      | exact fun i hk hc => left i ⟨hk, hc⟩
      | exact fun i hk hc => right i rfl ⟨hk, hc⟩

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked
