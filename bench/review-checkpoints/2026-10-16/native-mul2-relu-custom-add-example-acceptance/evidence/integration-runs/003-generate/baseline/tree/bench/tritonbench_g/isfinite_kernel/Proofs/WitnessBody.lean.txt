import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessHeader

/-! A single actual loop iteration is interpreted over an abstract incoming
state. Its store remains opaque; readiness is recovered solely by projecting
registers. The second iteration therefore never expands the first one's state. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def grid32_body_run (rest : Algebra Float32Bits) (j : Nat)
    (s : State Float32Bits) (h : GridReady s) :
    {t : State Float32Bits // FP.Structural.run (engine (backend32 rest)) gridBody32
      (s.setReg "j" .nat [] (fun _ => j)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [gridBody32, gridBody, gridArms, gridKernel32, source32,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend32, h.1, h.2, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

def grid64_body_run (rest : Algebra Float64Bits) (j : Nat)
    (s : State Float64Bits) (h : GridReady s) :
    {t : State Float64Bits // FP.Structural.run (engine (backend64 rest)) gridBody64
      (s.setReg "j" .nat [] (fun _ => j)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [gridBody64, gridBody, gridArms, gridKernel64, source64,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets, FP.Structural.store, FP.Structural.numeric,
    FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend64, h.1, h.2, ComputeDType.eraseDType,
    FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
  try rfl

theorem grid32_body_ready (rest : Algebra Float32Bits) (j : Nat)
    (s : State Float32Bits) (h : GridReady s) :
    GridReady (grid32_body_run rest j s h).val := by
  simp [GridReady, grid32_body_run, h.1, h.2]

theorem grid64_body_ready (rest : Algebra Float64Bits) (j : Nat)
    (s : State Float64Bits) (h : GridReady s) :
    GridReady (grid64_body_run rest j s h).val := by
  simp [GridReady, grid64_body_run, h.1, h.2]

#axiomsClean grid32_body_run
#axiomsClean grid64_body_run
#axiomsClean grid32_body_ready
#axiomsClean grid64_body_ready
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
