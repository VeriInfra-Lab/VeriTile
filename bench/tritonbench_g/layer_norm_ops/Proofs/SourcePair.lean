import bench.tritonbench_g.layer_norm_ops.Proofs.KernelExecution
import bench.tritonbench_g.layer_norm_ops.Proofs.SourceSafety
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormOps.SourcePair
open VeriTile.Triton FP.Structural Safety
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

private def backBefore (p : BackwardParameters) := (backwardKernel p).surfaceBody.take 14
private def backIndex (reuse : Bool) (p : BackwardParameters) :=
  ((if reuse then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody.drop 14).take 1
private def backAfter (p : BackwardParameters) := (backwardKernel p).surfaceBody.drop 15
private theorem back_split (reuse : Bool) (p : BackwardParameters) :
    (if reuse then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody =
      backBefore p ++ (backIndex reuse p ++ backAfter p) := by
  cases reuse <;> rfl

private theorem back_registers (M : Algebra α) (p : BackwardParameters) (s t : State α)
    (ran : run M (backBefore p) s=some t) :
    t.regs .nat [] "row_block_id"=some (fun _ => s.pids 0) ∧
    t.regs .nat [] "row_start"=some (fun _ => s.pids 0*p.rowsPerProgram) := by
  cases hd : p.hasDres <;> cases hs : p.storeDres <;>
    cases hy : p.recompute <;> cases hb : p.hasBias
  all_goals
    simp [backBefore,backwardKernel,backwardSource,ComputeKernel.surfaceBody,List.take,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,FP.Structural.numeric,FP.Structural.bop,FP.Structural.numericLt,FP.Structural.natLt,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,Region.cast,
      hd,hs,hy,hb,State.setReg] at ran
    subst t
    simp [State.setReg]
    rfl

private theorem back_index_run (M : Algebra α) (p : BackwardParameters) (s t : State α)
    (ran : run M (backBefore p) s=some t) :
    run M (backIndex Bool.false p) t=run M (backIndex Bool.true p) t := by
  obtain ⟨pid,start⟩ := back_registers M p s t ran
  simp [backIndex,backwardKernel,backwardSource,backwardRowStartReusedKernel,
    backwardRowStartReusedSource,ComputeKernel.surfaceBody,List.drop,List.take,
    run,step,evalExpr,evalOp_unfold,Option.bind,FP.Structural.numeric,FP.Structural.bop,FP.Structural.numericLt,FP.Structural.natLt,
    Broadcast.leftIndex,Broadcast.rightIndex,pid,start,Nat.add_mul]

private theorem back_index_safe (M : Algebra α) (bounds : RegionName → Nat)
    (reuse : Bool) (p : BackwardParameters) (s : State α) :
    Safety.traceSafeList M bounds (backIndex reuse p) s := by
  cases reuse
  all_goals
    simp [backIndex,backwardKernel,backwardSource,backwardRowStartReusedKernel,
      backwardRowStartReusedSource,ComputeKernel.surfaceBody,List.drop,List.take,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeAlg,
      Safety.safeOp,Safety.safeOp_unfold]
    split <;> trivial

theorem backward_run (M : Algebra α) (p : BackwardParameters) (s : State α) :
    exec M (backwardKernel p) s=exec M (backwardRowStartReusedKernel p) s := by
  change run M (if Bool.false then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody s =
    run M (if Bool.true then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody s
  conv_lhs => rw [back_split]
  conv_rhs => rw [back_split]
  simp only [run_append]
  cases h : run M (backBefore p) s with
  | none => rfl
  | some t => simp only [Option.bind_some,back_index_run M p s t h]

theorem backward_safe (M : Algebra α) (bounds : RegionName → Nat)
    (p : BackwardParameters) (s : State α) :
    TraceSafe M bounds (backwardKernel p) s ↔
      TraceSafe M bounds (backwardRowStartReusedKernel p) s := by
  have tail : ∀ t, run M (backBefore p) s=some t →
      (Safety.traceSafeList M bounds (backIndex Bool.false p ++ backAfter p) t ↔
       Safety.traceSafeList M bounds (backIndex Bool.true p ++ backAfter p) t) := by
    intro t ran
    simp only [Safety.traceSafeList_append,back_index_safe,true_and,
      back_index_run M p s t ran]
  change Safety.traceSafeList M bounds
    (if Bool.false then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody s ↔
    Safety.traceSafeList M bounds
    (if Bool.true then backwardRowStartReusedKernel p else backwardKernel p).surfaceBody s
  conv_lhs => rw [back_split]
  conv_rhs => rw [back_split]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  constructor
  · rintro ⟨head,next⟩
    exact ⟨head,fun t h => (tail t h).mp (next t h)⟩
  · rintro ⟨head,next⟩
    exact ⟨head,fun t h => (tail t h).mpr (next t h)⟩

theorem forward_memory (M : Algebra α) (p : ForwardParameters) (s : State α) :
    (FP.Structural.exec M (forwardKernel p) s).map State.mem=
      (FP.Structural.exec M (forwardMaskReusedKernel p) s).map State.mem := by
  cases hr : p.rms <;> cases ha : p.hasResidual <;>
    cases hs : p.storeResidual <;> cases hb : p.hasBias
  all_goals
    simp [FP.Structural.exec,forwardKernel,forwardSource,forwardMaskReusedKernel,forwardMaskReusedSource,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
      Option.bind,ComputeDType.eraseDType,FloatDType.toTileDType,
      FloatDType.numericDType,ofFloat,toFloat,FP.Structural.numeric,FP.Structural.bop,FP.Structural.numericLt,
      FP.Structural.natLt,Region.cast,Forward.vector_bop,hr,ha,hs,hb]
  all_goals simp [VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits.fold_writes_mem,
    State.setReg_mem,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem forward_candidate_safe (M : Algebra α) (p : ForwardParameters) (s : State α)
    (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hres : p.hasResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualStride+i.val<bounds p.residual)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y)
    (hresout : p.storeResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualOutStride+i.val<bounds p.residualOut)
    (hm : p.rms=Bool.false → s.pids 0<bounds p.mean)
    (hr : s.pids 0<bounds p.rstd) :
    TraceSafe M bounds (forwardMaskReusedKernel p) s := by
  cases hR : p.rms <;> cases hA : p.hasResidual <;>
    cases hS : p.storeResidual <;> cases hB : p.hasBias
  all_goals
    simp [TraceSafe,forwardMaskReusedKernel,forwardMaskReusedSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      numeric,bop,numericLt,natLt,Region.cast,hR,hA,hS,hB,hx,hw,hb,hres,hy,hresout,hm,hr]
  all_goals repeat' first
    | apply And.intro
    | exact hx
    | exact hw
    | exact hy
    | exact hr
    | exact hb hB
    | exact hres hA
    | exact hresout hS
    | exact hm hR


#axiomsClean forward_memory
#axiomsClean forward_candidate_safe
#axiomsClean backward_run
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.LayerNormOps.SourcePair
