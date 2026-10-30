import bench.tritonbench_g.chunked_cumsum_fwd.Kernels

/-! Execution helpers for the complete ordinary DSL source in Kernels.lean.
The historical quotation is independently checked in the campaign source
preservation check; no source generator is part of this proof dependency. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton

def sourceSHA256 : String := "5f834ef4712a5f138681eb9c97ec332c43119e716e4d3c82fa8537fc0ec02d2a"

open FP.Structural

def engine (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32
/-- The launch binds each scalar once, before executing the complete source. -/
def bindBounds (s : State α) (lower upper : α) : State α :=
  (s.setReg "dt_min" .real [] (fun _ => lower)).setReg "dt_max" .real [] (fun _ => upper)

theorem bound_registers (s : State α) (lower upper : α) :
    (bindBounds s lower upper).regs .real [] "dt_min" = some (fun _ => lower) ∧
    (bindBounds s lower upper).regs .real [] "dt_max" = some (fun _ => upper) := by
  simp [bindBounds,State.setReg]

end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
