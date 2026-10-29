import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.PairIO
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.CandidateKernels
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.PublicIO
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.SourcePair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 32000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev kernel_run := @SourcePair.kernel_run
abbrev kernel_safe := @SourcePair.kernel_safe

theorem invocation_run {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (pid : Nat) (s : State α) :
    (PublicIO.invocation g M D pid).run s = (invocation g M D pid).run s := by
  change exec (backend M D) (Source.typed g.request.parameters g.storage) (instanceState g.request pid s) =
    exec (backend M D) (CandidateSource.typed g.request.parameters g.storage) (instanceState g.request pid s)
  exact kernel_run _ _ _ _

theorem invocation_safe {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat) (pid : Nat) (s : State α) :
    (PublicIO.invocation g M D pid).Safe bounds s ↔ (invocation g M D pid).Safe bounds s := by
  change TraceSafe (backend M D) bounds (Source.typed g.request.parameters g.storage) (instanceState g.request pid s) ↔
    TraceSafe (backend M D) bounds (CandidateSource.typed g.request.parameters g.storage) (instanceState g.request pid s)
  exact kernel_safe _ _ _ _ _

theorem schedule_run {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (ids : List Nat) (s : State α) :
    runSchedule (ids.map (PublicIO.invocation g M D)) s = runSchedule (ids.map (invocation g M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [List.map_cons,runSchedule,←invocation_run]
    cases he : (PublicIO.invocation g M D pid).run s with
    | none => rfl
    | some next => exact ih next

theorem schedule_safe {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat) (ids : List Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (PublicIO.invocation g M D)) s ↔
      ScheduleSafe bounds (ids.map (invocation g M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [List.map_cons,ScheduleSafe,←invocation_safe,←invocation_run]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)

theorem equivalent (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    (io rest I g ids).Equivalent (variant rest I g ids) := by
  intro vals _ bounds backed initial pinned
  have proved := PublicIO.implements g (Native.algebra rest I) (Native.dotAcc I)
    (Native.numerics I g.storage) (Native.primitives rest I g.storage) ids order sepA sepB
    (logicalReference (Native.numerics I g.storage) g.request)
    (reference_logical (Native.numerics I g.storage) g.request)
  obtain ⟨safe,final,ran,outputs,frame⟩ := proved vals trivial bounds backed initial pinned
  refine ⟨safe,(schedule_safe g _ _ bounds ids initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (ids.map (invocation g (Native.algebra rest I) (Native.dotAcc I))) initial=some final
    rw [←schedule_run]
    exact ran
  · intro k j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair
