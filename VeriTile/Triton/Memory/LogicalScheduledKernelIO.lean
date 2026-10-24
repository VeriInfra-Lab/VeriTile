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

end LogicalScheduledKernelIO
end
end VeriTile.Triton
