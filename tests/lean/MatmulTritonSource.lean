import bench.tritonbench_g.matmul_triton2.Proofs.SourceParts
import bench.tritonbench_g.matmul_triton2.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.MatmulTritonSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton2 Revision001
noncomputable section

/-- Nine ABI arguments precede the unchanged 28 Python statements; the
candidate adds exactly its two cached masks. -/
theorem complete_sources (p : Core.Parameters) :
    (KernelOnly.source p).surfaceBody.length=37 ∧
      (KernelOnly.maskHoisted p).surfaceBody.length=39 := by
  constructor <;> rfl

theorem scalar_wrap {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (s : State (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)) :
    (SourceRuntime.preludeExecution rest P {p with rows := 2^64+3} s).val.regs
      .nat [] "M"=some (fun _ => 3) := by
  simpa using SourceRuntime.prelude_rows rest P {p with rows := 2^64+3} s

/-- The default TF32 source profile stays distinct from the explicit
allow_tf32=False profile used by matmul_triton1. -/
theorem reject_different_dot_profile {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    {m k n : Nat} (xs : TileIndex [m,k] → VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α)
    (ys : TileIndex [k,n] → VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.Value α) :
    (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P {}).dotWithOptions (some .fp32) {allowTF32 := some Bool.false}
      (batch:=[]) (m:=m) (k:=k) (n:=n) xs ys=none := by
  simp [VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra]

#axiomsClean complete_sources
#axiomsClean scalar_wrap
#axiomsClean reject_different_dot_profile
end
end VeriTile.Tests.MatmulTritonSource
