import bench.tritonbench_g.rotary_emb.Proofs.IOResult
import bench.tritonbench_g.rotary_emb.Proofs.SourceSafety
import bench.tritonbench_g.rotary_emb.Proofs.TypedNative

namespace VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- Public active windows cover each evaluated source access. In particular,
the head-broadcast coefficient lanes are checked even for inactive Q/K heads. -/
theorem invocation_safe {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (pids : Nat → Nat) (values : Inputs A shape)
    (bounds : RegionName → Nat) (backed : (kernelIO rest ops shape pids).InBounds bounds values)
    (s : State A) : (invocation rest ops shape pids).Safe bounds s := by
  unfold Invocation.Safe
  rw [invocation_initial]
  apply Typed.source_trace_safe
  · intro key odd j live
    rw [offset_eq]
    exact payload_bounds rest ops shape pids values bounds backed key odd j live
  · intro sine odd j live
    have bound := coefficient_bounds rest ops shape pids values bounds backed sine odd
      (j.1,0,j.2.2) live
    cases sine <;> cases odd <;>
      simpa [Typed.coefficientOffset,Typed.layout,coefficientOffset,Nat.add_assoc] using bound

theorem implements {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    (kernelIO rest ops shape pids).Implements (rotation ops shape) := by
  apply ProgramIO.Implements.intro
  · intro values bounds backed s _
    exact (scheduleSafe_singleton _ _ _).mpr
      (invocation_safe rest ops shape pids values bounds backed s)
  · intro values s bound
    obtain ⟨t,executed,outputs,frame⟩ := invocation_result rest ops shape valid pids values s bound
    exact ⟨t,(runSchedule_singleton _ _).trans executed,outputs,frame⟩

theorem candidate_implements {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    (candidate rest ops shape pids).toProgramIO.Implements (rotation ops shape) := by
  intro values _ bounds backed s bound
  obtain ⟨t,executed,outputs,frame⟩ := invocation_result rest ops shape valid pids values s bound
  have safe := invocation_safe rest ops shape pids values bounds backed s
  have candidateSafe := Typed.candidate_trace_safe rest ops (Typed.layout shape)
    ((invocation rest ops shape pids).initial s) bounds safe
  refine ⟨?_,t,?_,outputs,frame⟩
  · exact (scheduleSafe_singleton _ _ _).mpr candidateSafe
  · change runSchedule [{invocation rest ops shape pids with
        kernel := Candidate.optimized (Typed.layout shape)}] s=some t
    rw [runSchedule_singleton]
    change exec (Typed.algebra rest ops) (Candidate.optimized (Typed.layout shape))
      ((invocation rest ops shape pids).initial s)=some t
    exact (Candidate.execution_equiv _ _ _).symm.trans executed

theorem equivalent {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    (kernelIO rest ops shape pids).Equivalent (candidate rest ops shape pids) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (implements rest ops shape valid pids) (candidate_implements rest ops shape valid pids)

theorem native_result (rest : Algebra Native.Node) (p : Native.Instructions)
    (shape : Shape) (valid : shape.Valid) (pids : Nat → Nat) :
    (kernelIO rest (Native.ops p) shape pids).Implements (nativeRotation p shape) ∧
      (kernelIO rest (Native.ops p) shape pids).Equivalent (candidate rest (Native.ops p) shape pids) := by
  exact ⟨by simpa only [nativeRotation_eq] using implements rest (Native.ops p) shape valid pids,
    equivalent rest (Native.ops p) shape valid pids⟩

#axiomsClean implements
#axiomsClean equivalent
#axiomsClean native_result
end
end VeriTile.Bench.Repairs.RotaryEmb.RotaryIO
