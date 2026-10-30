import VeriTile.Triton.Memory.RegionalScheduledKernelIO

/-! Logical coordinates for a complete scheduled Real kernel contract.

This view reuses one existing regional IO record and its actual schedule.
Both coordinate maps are equivalences: no input values or output cells are
discarded. `Implements` is exactly the underlying scheduled judgment after
reindexing, retaining execution, access safety and the exact memory frame.
-/
namespace VeriTile.Triton
noncomputable section

/-- A logical view of a scheduled regional IO with Real-valued outputs. -/
structure LogicalScheduledKernelIO (Inputs Index : Type) where
  core : UKernelIO
  schedule : core.Ctx → List AlgorithmLaunch
  inputs : core.Ctx ≃ Inputs
  indices : ((o : Fin core.nOut) × Fin (core.oarity o)) ≃ Index
  realOutput : ∀ o, core.oty o = .float

namespace LogicalScheduledKernelIO

/-- Reindex only the mathematical formula; the source, schedule, input
binding, bounds, successful execution and frame remain those of `core`. -/
def Implements {Inputs Index : Type} (io : LogicalScheduledKernelIO Inputs Index)
    (f : Inputs → Index → ℝ) : Prop :=
  io.core.RegionalScheduledImplements io.schedule (fun vals o j =>
    (io.realOutput o).symm ▸ f (io.inputs vals) (io.indices ⟨o, j⟩))

scoped infix:25 " ⊨ " => LogicalScheduledKernelIO.Implements

/-- Restrict logical inputs while retaining the complete scheduled judgment.
In particular the restriction applies before safety and successful execution;
it does not weaken the output or exterior-frame obligations. -/
def ImplementsOn {Inputs Index : Type} (io : LogicalScheduledKernelIO Inputs Index)
    (allowed : Inputs → Prop) (f : Inputs → Index → ℝ) : Prop :=
  ∀ (s : BlockState) (vals : io.core.Ctx), allowed (io.inputs vals) →
    io.core.RegionalInputs s vals →
    (∀ bounds, io.core.RegionalBounds bounds s vals →
      AlgorithmScheduleSafe io.core.kernel.toAlgKernel bounds (io.schedule vals) s) ∧
    ∃ final,
      runAlgorithmSchedule io.core.kernel.toAlgKernel (io.schedule vals) s = some final ∧
      (∀ (o : Fin io.core.nOut) (j : Fin (io.core.oarity o)),
        io.core.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        (io.core.oty o).read final (io.core.obuf o)
          (io.core.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
          (io.realOutput o).symm ▸ f (io.inputs vals) (io.indices ⟨o,j⟩)) ∧
      io.core.RegionalFrame s final vals

theorem implementsOn_true {Inputs Index : Type}
    (io : LogicalScheduledKernelIO Inputs Index) (f : Inputs → Index → ℝ) :
    io.ImplementsOn (fun _ => True) f ↔ io.Implements f := by
  simp only [ImplementsOn, Implements, UKernelIO.RegionalScheduledImplements,
    UKernelIO.RegionalScheduledImplements3D, true_implies]

end LogicalScheduledKernelIO
end
end VeriTile.Triton
