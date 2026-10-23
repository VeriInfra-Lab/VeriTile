import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightMemory
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightSafety
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeWeightSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
open FPWeightProgram FPWeightStep FPWeightLoop
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem iteration_safe (M : Algebra α) (h : Call) (K pid row : Nat) (initial s : State α)
    (bounds : RegionName → Nat) (inside : RealWeightSafety.Bounds h K pid bounds)
    (inv : Holds M h K pid initial row s) (active : row<h.rows) :
    traceSafeList (engine M) bounds (loopBody h K) (s.setReg "row" .nat [] (fun _=>row)) := by
  cases mode : h.gemma
  all_goals
    simp (config := {maxSteps:=500000}) [loopBody,Kernels.weightBackward,mode,
      ComputeKernel.surfaceBody,traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,
      maskActive_unfold,run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,
      ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,Region.cast,
      State.setReg,ofFloat,toFloat,inv.col,inv.mask,inv.acc,columns,masks]
  all_goals exact ⟨inside.upstream row active,inside.input row active,inside.saved row active⟩

theorem loop_safe (M : Algebra α) (h : Call) (K pid row : Nat) (initial s : State α)
    (bounds : RegionName → Nat) (inside : RealWeightSafety.Bounds h K pid bounds)
    (inv : Holds M h K pid initial row s) :
    rangeSafe (engine M) bounds "row" row h.rows 1 (loopBody h K) s := by
  rw [rangeSafe_unfold]
  by_cases active : row<h.rows
  · simp only [Nat.one_ne_zero,if_false,active,if_true]
    refine ⟨iteration_safe M h K pid row initial s bounds inside inv active,?_⟩
    obtain ⟨next,ran,mem,_,col,mask,acc⟩ := iteration M h K pid row s
      (carry M h K pid initial row) inv.col inv.mask inv.acc
    rw [ran]
    apply loop_safe M h K pid (row+1) initial next bounds inside
    refine ⟨by omega,col,mask,?_,mem.trans inv.mem⟩
    rw [acc]
    congr 1
    funext i
    simp [nextAcc,carry,FPWeightReference.carry,raw,saved,inv.mem]
  · simp [active]
termination_by h.rows-row

theorem complete (M : Algebra α) (h : Call) (K : Nat) (s : State α)
    (bounds : RegionName → Nat) (inside : RealWeightSafety.Bounds h K (s.pids 0) bounds) :
    TraceSafe (engine M) bounds (Kernels.weightBackward h K) s := by
  unfold TraceSafe
  rw [source_body,List.append_assoc,traceSafeList_append]
  constructor
  · simp [prefixCode,Kernels.weightBackward,ComputeKernel.surfaceBody,traceSafeList_unfold,
      stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,natLt,numericLt,bop,Region.cast,State.setReg]
  · intro after head
    rw [prefix_executes] at head
    cases head
    apply traceSafeList_cons_intro
    · simp only [stepSafe_unfold,safeExpr_unfold,safeOp_unfold,evalOp_unfold]
      exact ⟨trivial,trivial,trivial,loop_safe M h K (s.pids 0) 0 s _ bounds inside (initial_holds M h K s)⟩
    · intro after ran
      obtain ⟨final,executed,inv⟩ := loop_executes M h K (s.pids 0) s 0 _ (initial_holds M h K s)
      simp only [step,evalOp_unfold,Option.bind] at ran
      change range (engine M) "row" 0 h.rows 1 (loopBody h K) (initial M h K s)=some after at ran
      rw [executed] at ran
      cases ran
      simp [tailCode,Kernels.weightBackward,ComputeKernel.surfaceBody,traceSafeList_unfold,
        stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,
        safeMask_unfold,addressSafe_unfold,maskActive_unfold,run,step,evalExpr,evalComputeOp,
        evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,
        ComputeDType.eraseDType,FP.Structural.store,State.setReg,inv.col,inv.mask,inv.acc,columns,masks,Region.cast]
      exact inside.output

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeWeightSafety
