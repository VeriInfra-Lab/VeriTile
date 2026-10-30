import bench.tritonbench_g.layer_norm_ops.Proofs.MathValues
import bench.tritonbench_g.layer_norm_ops.Proofs.IOResults

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section

theorem forward_real (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p) (covered : p.columns≤p.block) :
    (forwardIO realAlgebra p pid).Implements (forwardMath p) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact forward_safe realAlgebra p pid vals bounds hb initial
  · intro vals initial hin
    let u := (invocation realAlgebra (forwardKernel p) pid).initial initial
    have hp : u.pids 0=pid := rfl
    have hm : u.mem=initial.mem := Invocation.initial_mem _ _
    have bound : ForwardBinding p vals u := by
      apply forward_binding realAlgebra p pid vals u hp
      intro port lane live
      simpa only [hm] using hin port lane live
    obtain ⟨final,ran,outputs,frame⟩ := forward_result realAlgebra p pid admissible vals initial hin
    refine ⟨final,ran,?_,frame⟩
    intro o j live
    have result := outputs o j live
    rw [forward_graph_math p vals u bound covered o j live] at result
    exact result

theorem backward_real (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p) (covered : p.columns≤p.block) :
    (backwardIO realAlgebra p pid).Implements (backwardMath p pid) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact backward_safe realAlgebra p pid vals bounds hb initial
  · intro vals initial hin
    let u := (invocation realAlgebra (backwardKernel p) pid).initial initial
    have hp : u.pids 0=pid := rfl
    have hm : u.mem=initial.mem := Invocation.initial_mem _ _
    have bound : BackwardBinding p pid vals u := by
      apply backward_binding realAlgebra p pid vals u
      intro port lane live
      simpa only [hm] using hin port lane live
    obtain ⟨final,ran,outputs,frame⟩ := backward_result realAlgebra p pid admissible vals initial hin
    refine ⟨final,ran,?_,frame⟩
    intro o j live
    have result := outputs o j live
    rw [backward_graph_math p pid vals u bound hp admissible covered o j live] at result
    exact result

#axiomsClean forward_real
#axiomsClean backward_real
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
