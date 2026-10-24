import bench.repairs.tritonbench_g.triton_argmax_nan_first.PairIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITPair
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITNativePublicIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITNativeDimensionIO

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITProgramPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
theorem first_run {casts kind} (binding : BoundSource.Binding casts kind) (A : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    exec A (JITBinding.first binding c index64) s=exec A (candidateFirst binding c index64) s := by
  exact JITPair.First.kernel_run A s BoundSource.X BoundSource.V BoundSource.I
    (JITBinding.dtype binding) c.block index64

theorem first_safe {casts kind} (binding : BoundSource.Binding casts kind) (A : Algebra Word)
    (bounds : RegionName → Nat) (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    TraceSafe A bounds (JITBinding.first binding c index64) s ↔ TraceSafe A bounds (candidateFirst binding c index64) s := by
  exact JITPair.First.kernel_safe A bounds BoundSource.X BoundSource.V BoundSource.I
    (JITBinding.dtype binding) c.block index64 s


theorem first_invocation_run {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) (s : State Word) :
    (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0 pid).run s=
      (firstInvocation binding rest c index64 pid).run s := by
  simp only [firstInvocation,Invocation.run,JITIO.firstLaunch_initial,JITIO.atProgram_initial]
  exact first_run _ _ _ _ _

theorem first_invocation_safe {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) (bounds : RegionName → Nat) (s : State Word) :
    (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0 pid).Safe bounds s ↔
      (firstInvocation binding rest c index64 pid).Safe bounds s := by
  simp only [firstInvocation,Invocation.Safe,JITIO.firstLaunch_initial,JITIO.atProgram_initial]
  exact first_safe _ _ _ _ _ _

theorem first_list_run {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (work : List Nat) (s : State Word) :
    runSchedule (work.map (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0)) s=
      runSchedule (work.map (firstInvocation binding rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,runSchedule,←first_invocation_run]
    cases he : (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0 pid).run s with
    | none => rfl
    | some next => exact ih next

theorem first_list_safe {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (work : List Nat) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (JITIO.firstLaunch (JITBinding.ops binding rest) (JITBinding.first binding c index64) c.size 0)) s ↔
      ScheduleSafe bounds (work.map (firstInvocation binding rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid tail ih =>
    simp only [List.map_cons,ScheduleSafe,←first_invocation_run,←first_invocation_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)


theorem global_run (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) (s : State Word) :
    runSchedule (JITIO.nativeSchedule casts kind binding rest c index64) s=
      runSchedule (globalSchedule binding rest c index64) s := by
  simp only [JITIO.nativeSchedule,globalSchedule,runSchedule_append,first_list_run]

theorem global_safe (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (JITIO.nativeSchedule casts kind binding rest c index64) s ↔
      ScheduleSafe bounds (globalSchedule binding rest c index64) s := by
  simp only [JITIO.nativeSchedule,globalSchedule,scheduleSafe_append,first_list_safe,first_list_run]


theorem global_equivalent (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (g : Native.GlobalPrograms c) (positive : 0<c.block)
    (midPositive : 0<c.midBlock) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (index64 : Bool) :
    (JITIO.nativeGlobal casts kind binding rest c index64).EquivalentOn (globalVariant casts kind binding rest c index64)
      (JITNativeIO.initialized casts kind binding rest c index64) := by
  intro xs admitted bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    JITNativeIO.implements casts kind binding rest c g positive midPositive active index64 xs admitted bounds backed initial pinned
  refine ⟨safe,(global_safe casts kind binding rest c index64 bounds initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (globalSchedule binding rest c index64) initial=some final
    rw [←global_run]
    exact ran
  · intro o j _
    rfl


theorem dimension_invocation_run (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (job : Nat × Nat) (s : State Word) :
    (JITNativeDimensionIO.launch rest c index64 job).run s=(dimensionInvocation rest c index64 job).run s := by
  simp only [JITNativeDimensionIO.launch,dimensionInvocation,Invocation.run,JITIO.dimensionLaunch_initial,JITIO.atProgram_initial]
  exact JITPair.Dimension.kernel_run _ _ _ _ _ _ _

theorem dimension_invocation_safe (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (job : Nat × Nat) (bounds : RegionName → Nat) (s : State Word) :
    (JITNativeDimensionIO.launch rest c index64 job).Safe bounds s ↔
      (dimensionInvocation rest c index64 job).Safe bounds s := by
  simp only [JITNativeDimensionIO.launch,dimensionInvocation,Invocation.Safe,JITIO.dimensionLaunch_initial,JITIO.atProgram_initial]
  exact JITPair.Dimension.kernel_safe _ _ _ _ _ _ _ _

theorem dimension_list_run (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (work : List (Nat × Nat)) (s : State Word) :
    runSchedule (work.map (JITNativeDimensionIO.launch rest c index64)) s=
      runSchedule (work.map (dimensionInvocation rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job work ih =>
    simp only [List.map_cons,runSchedule,←dimension_invocation_run]
    cases he : (JITNativeDimensionIO.launch rest c index64 job).run s with
    | none => rfl
    | some next => exact ih next

theorem dimension_list_safe (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (work : List (Nat × Nat)) (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (JITNativeDimensionIO.launch rest c index64)) s ↔
      ScheduleSafe bounds (work.map (dimensionInvocation rest c index64)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job work ih =>
    simp only [List.map_cons,ScheduleSafe,←dimension_invocation_run,←dimension_invocation_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)


theorem dimension_equivalent (rest : Algebra Word) (c : DimConfig) (nonempty : 0<c.extent) (index64 : Bool) :
    (JITIO.nativeDimension rest c index64).Equivalent (dimensionVariant rest c index64) := by
  intro xs _ bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    JITNativeDimensionIO.implements rest c nonempty index64 xs trivial bounds backed initial pinned
  have safe' : ScheduleSafe bounds ((BoundSource.groups c).map (JITNativeDimensionIO.launch rest c index64)) initial := safe
  refine ⟨safe,(dimension_list_safe rest c index64 _ bounds initial).mp safe',final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule ((BoundSource.groups c).map (dimensionInvocation rest c index64)) initial=some final
    rw [←dimension_list_run]
    exact ran
  · intro o j _
    rfl

#axiomsClean global_equivalent
#axiomsClean dimension_equivalent
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITProgramPair
