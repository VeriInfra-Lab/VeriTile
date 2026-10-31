import bench.tritonbench_g.iv_dependent_matmul.Proofs.SourceBridge
import bench.tritonbench_g.iv_dependent_matmul.Proofs.RealSchedules
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.SourceSafety
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Kernels
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

private theorem list_safe (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (each : ∀ st∈code, ∀ s : State α, Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st tail ih =>
    apply Safety.traceSafeList_cons_intro (each st List.mem_cons_self s)
    intro t _
    exact ih (fun st h => each st (List.mem_cons_of_mem _ h)) t

/-- All pointer initialization is memory-free; this does not assume that a
formed pointer is in bounds before its actual masked access. -/
theorem prologue_safe (M : Algebra α) (bounds : RegionName → Nat)
    (schedule : Schedule) (p : Parameters) (s : State α) :
    Safety.traceSafeList M bounds (prologueWithPointers schedule p) s := by
  apply list_safe
  intro st member before
  cases schedule
  all_goals
    simp [prologueWithPointers,initialPointers,prologue,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,
      ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero,
      List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  all_goals repeat' first | (rcases member with rfl | member) | subst st
  all_goals simp [copyPointer,aAt,bAt,pointer,strideOffset,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Schedule registers contain exactly the current/staged source addresses.
The two base-offset functions are geometry, independent of all payloads. -/
def Pointers (schedule : Schedule) (p : Parameters) (t : State α)
    (aBase : TileIndex [p.blockRows,p.blockInner] → Nat)
    (bBase : TileIndex [p.blockInner,p.blockColumns] → Nat) (cur : Nat) : Prop :=
  let pair (an bn : RegName) (k : Nat) :=
    t.regs .ptr [p.blockRows,p.blockInner] an=
      some (fun i => (p.input,aBase i+k*p.blockInner*p.inputInnerStride)) ∧
    t.regs .ptr [p.blockInner,p.blockColumns] bn=
      some (fun i => (p.weight,bBase i+k*p.blockInner*p.weightInnerStride))
  match schedule with
  | .preLoad => True
  | .postLoad => pair "a_ptrs" "b_ptrs" cur
  | .mixed => t.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs"=
      some (fun i => (p.weight,bBase i+cur*p.blockInner*p.weightInnerStride))
  | .two => pair "a_ptrs" "b_ptrs" cur ∧ pair "a_ptrs_next" "b_ptrs_next" (cur+1)
  | .three => pair "a_ptrs" "b_ptrs" cur ∧ pair "a_ptrs_next" "b_ptrs_next" (cur+1) ∧
      pair "a_ptrs_next_next" "b_ptrs_next_next" (cur+2)

theorem after_loads_safe (M : Algebra α) (bounds : RegionName → Nat)
    (schedule : Schedule) (g : Graph) (p : Parameters) (s : State α) :
    Safety.traceSafeList M bounds ((FPValue.body g p).drop 4++afterDot schedule p) s := by
  apply list_safe
  intro st member before
  cases schedule <;> cases g
  all_goals
    simp [FPValue.body,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,fusedTyped,ComputeKernel.surfaceBody,
      afterDot,List.drop_succ_cons,List.drop_zero,List.mem_append,List.mem_cons,
      List.not_mem_nil,or_false] at member
  all_goals repeat' first | (rcases member with rfl | member) | subst st
  all_goals simp [copyPointer,aAt,bAt,pointer,strideOffset,iteration,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- Both loads are checked under the actual source masks. Arithmetic and
prefetch-pointer updates after the loads introduce no memory access. -/
theorem body_safe (M : Algebra α) (bounds : RegionName → Nat)
    (schedule : Schedule) (g : Graph) (p : Parameters) (t : State α) (cur : Nat)
    (aBase : TileIndex [p.blockRows,p.blockInner] → Nat)
    (bBase : TileIndex [p.blockInner,p.blockColumns] → Nat)
    (ha : t.regs .ptr [p.blockRows,p.blockInner] "a_ptr"=some (fun i => (p.input,aBase i)))
    (hb : t.regs .ptr [p.blockInner,p.blockColumns] "b_ptr"=some (fun i => (p.weight,bBase i)))
    (hk : t.regs .nat [p.blockInner] "offs_k"=some (fun i => i.1.val))
    (hp : Pointers schedule p t aBase bBase cur)
    (ba : ∀ i, i.2.1.val<p.inner-cur*p.blockInner →
      aBase i+cur*p.blockInner*p.inputInnerStride<bounds p.input)
    (bb : ∀ i, i.1.val<p.inner-cur*p.blockInner →
      bBase i+cur*p.blockInner*p.weightInnerStride<bounds p.weight) :
    Safety.traceSafeList M bounds (Kernels.body schedule g p)
      (t.setReg "k" .nat [] (fun _ => cur)) := by
  have split : Kernels.body schedule g p=
      (beforeLoads schedule p++((FPValue.body g p).drop 2).take 2)++
      ((FPValue.body g p).drop 4++afterDot schedule p) := by
    cases schedule <;> cases g <;> rfl
  rw [split,Safety.traceSafeList_append]
  refine ⟨?_,fun s _ => after_loads_safe M bounds schedule g p s⟩
  cases schedule <;> cases g <;> simp only [Pointers] at hp
  all_goals
    simp [beforeLoads,copyPointer,aAt,bAt,pointer,strideOffset,iteration,
      FPValue.body,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,fusedTyped,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      numeric,bop,natLt,numericLt,integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,
      State.setReg,ha,hb,hk,hp]
  all_goals exact ⟨fun a b c => ba (a,b,c),fun a b c => bb (a,b,c)⟩

/-- The output cast and index calculations preserve the single masked store. -/
theorem suffix_safe (M : Algebra α) (bounds : RegionName → Nat) (p : Parameters)
    (t : State α) (pm pn : Nat) (acc : Values α .real [p.blockRows,p.blockColumns])
    (hm : t.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : t.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : t.regs .real [p.blockRows,p.blockColumns] "accumulator"=some acc)
    (bo : ∀ (i : Fin p.blockRows) (j : Fin p.blockColumns),
      pm*p.blockRows+i.val<p.rows → pn*p.blockColumns+j.val<p.columns →
      p.outputRowStride*(pm*p.blockRows+i.val)+p.outputColumnStride*(pn*p.blockColumns+j.val)<bounds p.output) :
    Safety.traceSafeList M bounds (FPValue.suffix p) t := by
  simp [suffix,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,natLt,numericLt,Region.cast,ofFloat,toFloat,State.setReg,hm,hn,ha]
  refine ⟨bo, ?_⟩
  split <;> trivial

#axiomsClean body_safe
#axiomsClean suffix_safe
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.SourceSafety
