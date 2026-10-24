import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.ProjectedPairIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.ProjectedCandidateKernels
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativePublicIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativeDimensionIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Pair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000





theorem run_cons {α : Type} [Inhabited α] (A : Algebra α) (st : ComputeStmt)
    (rest : List ComputeStmt) (s : State α) :
    run A (st::rest) s=(step A st s).bind (run A rest) := by rw [run]; rfl

namespace First
def before (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) := (firstOriginal kind X V I size block base index64).surfaceBody.take 2
def after (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) := (firstOriginal kind X V I size block base index64).surfaceBody.drop 3
def originalSite (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) := ((firstOriginal kind X V I size block base index64).surfaceBody[2]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) := ((firstCandidate kind X V I size block base index64).surfaceBody[2]? : Option ComputeStmt).getD (.effectMarker "unreachable")
theorem original_code (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) :
    (firstOriginal kind X V I size block base index64).surfaceBody=before kind X V I size block base index64 ++ originalSite kind X V I size block base index64 :: after kind X V I size block base index64 := by cases kind <;> rfl
theorem shifted_code (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) :
    (firstCandidate kind X V I size block base index64).surfaceBody=before kind X V I size block base index64 ++ shiftedSite kind X V I size block base index64 :: after kind X V I size block base index64 := by cases kind <;> rfl

theorem original_site (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) :
    originalSite kind X V I size block base index64=originalSite .fp32 X V I size block base index64 := by cases kind <;> rfl
theorem shifted_site (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) :
    shiftedSite kind X V I size block base index64=shiftedSite .fp32 X V I size block base index64 := by cases kind <;> rfl

theorem site_run {α : Type} [Inhabited α] (A : Algebra α) (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (s : State α) :
    step A (originalSite kind X V I size block base index64) s=step A (shiftedSite kind X V I size block base index64) s := by
  rw [original_site,shifted_site]
  simp only [originalSite,shiftedSite,firstOriginal,firstCandidate,TypedSource.stage1FP32,CandidateSource.stage1FP32,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : block=1024
  all_goals simp only [hg,decide_true,decide_false]
  all_goals simp only [step,FP.Structural.run,evalExpr,evalOp_unfold]
  all_goals cases hr : s.regs .nat [] "pid"
  all_goals simp [FP.Structural.run,step,evalExpr,evalOp_unfold,hr,hg,numeric,FP.Structural.bop,
    Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.shiftLeft_eq]

theorem site_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat) (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (s : State α) :
    Safety.stepSafe A bounds (originalSite kind X V I size block base index64) s ∧ Safety.stepSafe A bounds (shiftedSite kind X V I size block base index64) s := by
  rw [original_site,shifted_site]
  simp only [originalSite,shiftedSite,firstOriginal,firstCandidate,TypedSource.stage1FP32,CandidateSource.stage1FP32,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : block=1024
  all_goals simp only [hg,decide_true,decide_false]
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
    Safety.traceSafeList_cons,Safety.traceSafeList_nil,evalOp_unfold,
    implies_true,and_true,true_and,Bool.false_eq_true,if_true,if_false]

theorem kernel_run {α : Type} [Inhabited α] (A : Algebra α) (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (s : State α) :
    exec A (firstOriginal kind X V I size block base index64) s=exec A (firstCandidate kind X V I size block base index64) s := by
  change run A (firstOriginal kind X V I size block base index64).surfaceBody s=run A (firstCandidate kind X V I size block base index64).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (run A (before kind X V I size block base index64) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat) (kind : Precision) (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (s : State α) :
    TraceSafe A bounds (firstOriginal kind X V I size block base index64) s ↔ TraceSafe A bounds (firstCandidate kind X V I size block base index64) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe A bounds kind X V I size block base index64 t).1,(site_safe A bounds kind X V I size block base index64 t).2,true_and,site_run]
end First

namespace Dimension
def before (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) := (TypedSource.dimension X O M N K BM BN base index64).surfaceBody.take 4
def after (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) := (TypedSource.dimension X O M N K BM BN base index64).surfaceBody.drop 5
def originalSite (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) := ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody[4]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) := ((CandidateSource.dimension X O M N K BM BN base index64).surfaceBody[4]? : Option ComputeStmt).getD (.effectMarker "unreachable")
theorem original_code (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) :
    (TypedSource.dimension X O M N K BM BN base index64).surfaceBody=before X O M N K BM BN base index64 ++ originalSite X O M N K BM BN base index64 :: after X O M N K BM BN base index64 := rfl
theorem shifted_code (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) :
    (CandidateSource.dimension X O M N K BM BN base index64).surfaceBody=before X O M N K BM BN base index64 ++ shiftedSite X O M N K BM BN base index64 :: after X O M N K BM BN base index64 := rfl

theorem site_run {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) (s : State α) :
    step A (originalSite X O M N K BM BN base index64) s=step A (shiftedSite X O M N K BM BN base index64) s := by
  simp only [originalSite,shiftedSite,TypedSource.dimension,CandidateSource.dimension,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : BM=128
  all_goals simp only [hg,decide_true,decide_false]
  all_goals simp only [step,FP.Structural.run,evalExpr,evalOp_unfold]
  all_goals cases hr : s.regs .nat [] "pid_m"
  all_goals simp [FP.Structural.run,step,evalExpr,evalOp_unfold,hr,hg,numeric,FP.Structural.bop,
    Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.shiftLeft_eq]

theorem site_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat) (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) (s : State α) :
    Safety.stepSafe A bounds (originalSite X O M N K BM BN base index64) s ∧ Safety.stepSafe A bounds (shiftedSite X O M N K BM BN base index64) s := by
  simp only [originalSite,shiftedSite,TypedSource.dimension,CandidateSource.dimension,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : BM=128
  all_goals simp only [hg,decide_true,decide_false]
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
    Safety.traceSafeList_cons,Safety.traceSafeList_nil,evalOp_unfold,
    implies_true,and_true,true_and,Bool.false_eq_true,if_true,if_false]

theorem kernel_run {α : Type} [Inhabited α] (A : Algebra α) (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) (s : State α) :
    exec A (TypedSource.dimension X O M N K BM BN base index64) s=exec A (CandidateSource.dimension X O M N K BM BN base index64) s := by
  change run A (TypedSource.dimension X O M N K BM BN base index64).surfaceBody s=run A (CandidateSource.dimension X O M N K BM BN base index64).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (run A (before X O M N K BM BN base index64) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat) (X : RegionName) (O : Region .int) (M N K BM BN base : Nat) (index64 : Bool) (s : State α) :
    TraceSafe A bounds (TypedSource.dimension X O M N K BM BN base index64) s ↔ TraceSafe A bounds (CandidateSource.dimension X O M N K BM BN base index64) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe A bounds X O M N K BM BN base index64 t).1,(site_safe A bounds X O M N K BM BN base index64 t).2,true_and,site_run]
end Dimension



theorem first_run {casts kind} (binding : BoundSource.Binding casts kind) (A : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    exec A (binding.first c index64) s=exec A (candidateFirst binding c index64) s := by
  cases binding with
  | floating k round codec =>
    cases k with
    | fp32 => exact First.kernel_run A .fp32 _ _ _ _ _ _ _ _
    | fp16 => exact First.kernel_run A .fp16 _ _ _ _ _ _ _ _
    | bf16 => exact First.kernel_run A .bf16 _ _ _ _ _ _ _ _
    | fp64 => exact First.kernel_run A .fp64 _ _ _ _ _ _ _ _
  | integer k zero => exact First.kernel_run A .integer _ _ _ _ _ _ _ _

theorem first_safe {casts kind} (binding : BoundSource.Binding casts kind) (A : Algebra Word)
    (bounds : RegionName → Nat) (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    TraceSafe A bounds (binding.first c index64) s ↔ TraceSafe A bounds (candidateFirst binding c index64) s := by
  cases binding with
  | floating k round codec =>
    cases k with
    | fp32 => exact First.kernel_safe A bounds .fp32 _ _ _ _ _ _ _ _
    | fp16 => exact First.kernel_safe A bounds .fp16 _ _ _ _ _ _ _ _
    | bf16 => exact First.kernel_safe A bounds .bf16 _ _ _ _ _ _ _ _
    | fp64 => exact First.kernel_safe A bounds .fp64 _ _ _ _ _ _ _ _
  | integer k zero => exact First.kernel_safe A bounds .integer _ _ _ _ _ _ _ _


theorem first_invocation_run {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) (s : State Word) :
    (IO.atProgram (binding.ops rest) (binding.first c index64) pid).run s=
      (firstInvocation binding rest c index64 pid).run s := by
  simp only [firstInvocation,Invocation.run,IO.atProgram_initial]
  exact first_run _ _ _ _ _

theorem first_invocation_safe {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) (bounds : RegionName → Nat) (s : State Word) :
    (IO.atProgram (binding.ops rest) (binding.first c index64) pid).Safe bounds s ↔
      (firstInvocation binding rest c index64 pid).Safe bounds s := by
  simp only [firstInvocation,Invocation.Safe,IO.atProgram_initial]
  exact first_safe _ _ _ _ _ _

theorem first_list_run {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (work : List Nat) (s : State Word) :
    runSchedule (work.map (IO.atProgram (binding.ops rest) (binding.first c index64))) s=
      runSchedule (work.map (firstInvocation binding rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,runSchedule,←first_invocation_run]
    cases he : (IO.atProgram (binding.ops rest) (binding.first c index64) pid).run s with
    | none => rfl
    | some next => exact ih next

theorem first_list_safe {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (work : List Nat) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (IO.atProgram (binding.ops rest) (binding.first c index64))) s ↔
      ScheduleSafe bounds (work.map (firstInvocation binding rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,ScheduleSafe,←first_invocation_run,←first_invocation_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)


theorem global_run (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    runSchedule (IO.nativeSchedule casts kind binding rest c index64) s=
      runSchedule (globalSchedule binding rest c index64) s := by
  simp only [IO.nativeSchedule,globalSchedule,runSchedule_append,first_list_run]

theorem global_safe (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (IO.nativeSchedule casts kind binding rest c index64) s ↔
      ScheduleSafe bounds (globalSchedule binding rest c index64) s := by
  simp only [IO.nativeSchedule,globalSchedule,scheduleSafe_append,first_list_safe,first_list_run]


theorem global_equivalent (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.block)
    (midPositive : 0<c.midBlock) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (index64 : Bool) :
    (IO.nativeGlobal casts kind binding rest c index64).EquivalentOn (globalVariant casts kind binding rest c index64)
      (NativeIO.initialized casts kind binding rest c index64) := by
  intro xs admitted bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    NativeIO.implements casts kind binding rest c g positive midPositive active index64 xs admitted bounds backed initial pinned
  refine ⟨safe,(global_safe casts kind binding rest c index64 bounds initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (globalSchedule binding rest c index64) initial=some final
    rw [←global_run]
    exact ran
  · intro o j _
    rfl


theorem dimension_invocation_run (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (job : Nat × Nat) (s : State Word) :
    (NativeDimensionIO.launch rest c index64 job).run s=(dimensionInvocation rest c index64 job).run s := by
  simp only [NativeDimensionIO.launch,dimensionInvocation,Invocation.run,IO.atProgram_initial]
  exact Dimension.kernel_run _ _ _ _ _ _ _ _ _ _ _

theorem dimension_invocation_safe (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (job : Nat × Nat) (bounds : RegionName → Nat) (s : State Word) :
    (NativeDimensionIO.launch rest c index64 job).Safe bounds s ↔
      (dimensionInvocation rest c index64 job).Safe bounds s := by
  simp only [NativeDimensionIO.launch,dimensionInvocation,Invocation.Safe,IO.atProgram_initial]
  exact Dimension.kernel_safe _ _ _ _ _ _ _ _ _ _ _ _

theorem dimension_list_run (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (work : List (Nat × Nat)) (s : State Word) :
    runSchedule (work.map (NativeDimensionIO.launch rest c index64)) s=
      runSchedule (work.map (dimensionInvocation rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job work ih =>
    simp only [List.map_cons,runSchedule,←dimension_invocation_run]
    cases he : (NativeDimensionIO.launch rest c index64 job).run s with
    | none => rfl
    | some next => exact ih next

theorem dimension_list_safe (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (work : List (Nat × Nat)) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (NativeDimensionIO.launch rest c index64)) s ↔
      ScheduleSafe bounds (work.map (dimensionInvocation rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job work ih =>
    simp only [List.map_cons,ScheduleSafe,←dimension_invocation_run,←dimension_invocation_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)


theorem dimension_equivalent (rest : Algebra Word) (c : DimConfig) (nonempty : 0<c.extent) (index64 : Bool) :
    (IO.nativeDimension rest c index64).Equivalent (dimensionVariant rest c index64) := by
  intro xs _ bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    NativeDimensionIO.implements rest c nonempty index64 xs trivial bounds backed initial pinned
  have safe' : ScheduleSafe bounds ((BoundSource.groups c).map (NativeDimensionIO.launch rest c index64)) initial := safe
  refine ⟨safe,(dimension_list_safe rest c index64 _ bounds initial).mp safe',final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule ((BoundSource.groups c).map (dimensionInvocation rest c index64)) initial=some final
    rw [←dimension_list_run]
    exact ran
  · intro o j _
    rfl

#axiomsClean global_equivalent
#axiomsClean dimension_equivalent
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Pair
