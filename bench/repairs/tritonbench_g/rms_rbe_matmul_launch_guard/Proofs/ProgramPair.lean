import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.IO
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicSourceIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmul
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

theorem invocation_run (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (q : KernelGrid.Program) (s : State α) :
    (candidateInvocation a fp8 p q).run s = (PublicIO.invocation a fp8 p q).run s :=
  (SourcePair.execution (KernelSource.engine a) fp8 p (KernelGrid.atProgram s q)).symm

theorem invocation_safe (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (q : KernelGrid.Program) (bounds : RegionName → Nat) (s : State α) :
    (PublicIO.invocation a fp8 p q).Safe bounds s ↔ (candidateInvocation a fp8 p q).Safe bounds s :=
  SourcePair.trace_safe (KernelSource.engine a) fp8 p bounds (KernelGrid.atProgram s q)

theorem schedule_run (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (qs : List KernelGrid.Program) (s : State α) :
    runSchedule (qs.map (candidateInvocation a fp8 p)) s =
      runSchedule (qs.map (PublicIO.invocation a fp8 p)) s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    exact congrArg (fun f => (PublicIO.invocation a fp8 p q).run s >>= f) (funext ih)

theorem schedule_safe (a : KernelSource.Operations α) (fp8 : Bool)
    (p : KernelSource.Parameters) (qs : List KernelGrid.Program) (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (qs.map (PublicIO.invocation a fp8 p)) s ↔
      ScheduleSafe bounds (qs.map (candidateInvocation a fp8 p)) s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_run]
    exact and_congr (invocation_safe a fp8 p q bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Successful safe execution of the complete candidate grid, with the same
ordered scalar graph at every output and the original exterior-cell frame. -/
theorem candidate_implements (a : KernelSource.Operations α) (fp8 : Bool) (B : Nat)
    (p : KernelSource.Parameters) :
    (candidate a fp8 B p).toProgramIO.ImplementsOn (PublicIO.domain fp8 B p)
      (PublicIO.formula a fp8 B p) := by
  intro xs admitted bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := PublicIO.implements a fp8 B p xs admitted bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (schedule_safe a fp8 p (KernelGrid.programs B p) bounds initial).mp safe
  · exact (schedule_run a fp8 p (KernelGrid.programs B p) initial).trans ran

/-- All original logical inputs and signed-byte storage are retained. No
finite-value or successful-execution premise is added to the public domain. -/
theorem equivalent (a : KernelSource.Operations α) (fp8 : Bool) (B : Nat)
    (p : KernelSource.Parameters) :
    (PublicIO.io a fp8 B p).EquivalentOn (candidate a fp8 B p) (PublicIO.domain fp8 B p) :=
  ProgramIO.EquivalentOn.of_implementsOn (PublicIO.implements a fp8 B p)
    (candidate_implements a fp8 B p)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PairIO
