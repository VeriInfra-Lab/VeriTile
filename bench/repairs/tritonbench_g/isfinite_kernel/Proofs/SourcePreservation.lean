import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.LegacySurface
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.IsfiniteKernel.SourcePreservation
open VeriTile.Triton FP.Structural

/-- The full executable body is unchanged. The call signature also repairs the
legacy collector's omitted input port: the actual source reads region x. -/
theorem source (precision : Precision) (input output : View) (n B : Nat) :
    IsfiniteKernel.source precision input output n B =
      .mk ["x"] ["y"] (LegacySurface.source precision input output n B).surfaceBody := by
  cases precision <;> rfl

theorem indexCommuted (precision : Precision) (input output : View) (n B : Nat) :
    IsfiniteKernel.indexCommuted precision input output n B =
      .mk ["x"] ["y"] (LegacySurface.indexCommuted precision input output n B).surfaceBody := by
  cases precision <;> rfl

/-- Whole-state execution, including failures, is identical for every
primitive algebra. This does not assume a floating-point cast identity. -/
theorem source_execution {A : Type} [Inhabited A] (M : Algebra A)
    (precision : Precision) (input output : View) (n B : Nat) (s : State A) :
    exec M (IsfiniteKernel.source precision input output n B) s =
      exec M (LegacySurface.source precision input output n B) s := by
  cases precision <;> rfl

theorem candidate_execution {A : Type} [Inhabited A] (M : Algebra A)
    (precision : Precision) (input output : View) (n B : Nat) (s : State A) :
    exec M (IsfiniteKernel.indexCommuted precision input output n B) s =
      exec M (LegacySurface.indexCommuted precision input output n B) s := by
  cases precision <;> rfl

theorem source_safety {A : Type} [Inhabited A] (M : Algebra A)
    (precision : Precision) (input output : View) (n B : Nat) (s : State A) (bounds : RegionName → Nat) :
    TraceSafe M bounds (IsfiniteKernel.source precision input output n B) s ↔
      TraceSafe M bounds (LegacySurface.source precision input output n B) s := by
  cases precision <;> rfl

theorem candidate_safety {A : Type} [Inhabited A] (M : Algebra A)
    (precision : Precision) (input output : View) (n B : Nat) (s : State A) (bounds : RegionName → Nat) :
    TraceSafe M bounds (IsfiniteKernel.indexCommuted precision input output n B) s ↔
      TraceSafe M bounds (LegacySurface.indexCommuted precision input output n B) s := by
  cases precision <;> rfl

#axiomsClean source
#axiomsClean indexCommuted
#axiomsClean source_execution
#axiomsClean candidate_execution
#axiomsClean source_safety
#axiomsClean candidate_safety
end VeriTile.Bench.Repairs.IsfiniteKernel.SourcePreservation
