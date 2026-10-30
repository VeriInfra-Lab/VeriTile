import bench.tritonbench_g.rms_rbe_matmul.IO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

theorem invocation_initial (a : Operations α) (fp8 : Bool) (p : Parameters)
    (q : KernelGrid.Program) (s : State α) :
    (invocation a fp8 p q).initial s=KernelGrid.atProgram s q := rfl

theorem schedule_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (qs : List KernelGrid.Program) (s : State α) :
    runSchedule (qs.map (invocation a fp8 p)) s=KernelGrid.runPrograms a fp8 p qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,runSchedule,KernelGrid.runPrograms]
    change (FP.Structural.exec (engine a) (kernel fp8 p) (KernelGrid.atProgram s q)).bind
      (runSchedule (qs.map (invocation a fp8 p))) = _
    simp only [ih]
    rfl

#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
