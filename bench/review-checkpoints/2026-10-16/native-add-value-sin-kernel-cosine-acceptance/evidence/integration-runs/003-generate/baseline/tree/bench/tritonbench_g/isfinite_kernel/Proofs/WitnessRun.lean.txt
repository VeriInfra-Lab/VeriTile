import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessBody
import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessComposition

/-! Complete source execution composed from actual header and iteration
proofs. The second body's incoming state is opaque during its proof. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

theorem grid_initial32_ready : GridReady (gridEntered initial32) :=
  grid_entered_ready initial32 rfl rfl

theorem grid_initial64_ready : GridReady (gridEntered initial64) :=
  grid_entered_ready initial64 rfl rfl

def grid32_first (rest : Algebra Float32Bits) :=
  grid32_body_run rest 0 (gridEntered initial32) grid_initial32_ready

def grid64_first (rest : Algebra Float64Bits) :=
  grid64_body_run rest 0 (gridEntered initial64) grid_initial64_ready

def grid32_second (rest : Algebra Float32Bits) :=
  grid32_body_run rest 1 (grid32_first rest).val
    (grid32_body_ready rest 0 (gridEntered initial32) grid_initial32_ready)

def grid64_second (rest : Algebra Float64Bits) :=
  grid64_body_run rest 1 (grid64_first rest).val
    (grid64_body_ready rest 0 (gridEntered initial64) grid_initial64_ready)

def grid32_run (rest : Algebra Float32Bits) :
    {s : State Float32Bits //
      FP.Structural.exec (engine (backend32 rest)) gridKernel32 initial32 = some s} :=
  ⟨(grid32_second rest).val,
    grid_exec_of_steps (engine (backend32 rest)) gridKernel32 gridBefore
      (gridArms gridKernel32).1 gridNumCtas gridBody32 initial32 (gridEntered initial32)
      (grid32_first rest).val (grid32_second rest).val grid32_split
      (grid_header_eval _ initial32) (grid32_first rest).property
      (grid32_second rest).property⟩

def grid64_run (rest : Algebra Float64Bits) :
    {s : State Float64Bits //
      FP.Structural.exec (engine (backend64 rest)) gridKernel64 initial64 = some s} :=
  ⟨(grid64_second rest).val,
    grid_exec_of_steps (engine (backend64 rest)) gridKernel64 gridBefore
      (gridArms gridKernel64).1 gridNumCtas gridBody64 initial64 (gridEntered initial64)
      (grid64_first rest).val (grid64_second rest).val grid64_split
      (grid_header_eval _ initial64) (grid64_first rest).property
      (grid64_second rest).property⟩

#axiomsClean grid_initial32_ready
#axiomsClean grid_initial64_ready
#axiomsClean grid32_run
#axiomsClean grid64_run
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
