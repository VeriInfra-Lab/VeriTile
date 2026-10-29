import bench.tritonbench_g.isfinite_kernel.Proofs.BitBackend
import bench.tritonbench_g.isfinite_kernel.Kernels

/-! Small first target for complete, actual source execution. It invokes the
unchanged writer with empty registers, an infinity input, and a Bool output.
The arbitrary remaining algebra operations cannot determine the result. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource BitClassification
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def one32 : ComputeKernel :=
  source32 "input" "output" 1 0 2 0 1 1 1 1 Bool.true rfl rfl

def one64 : ComputeKernel :=
  source64 "input" "output" 1 0 2 0 1 1 1 1 Bool.true rfl rfl

theorem whole32_infinity (rest : Algebra Float32Bits) :
    (FP.Structural.exec (engine (backend32 rest)) one32
      (initial32 (fun _ => bits32 0xff800000))).map
        (fun s => (boolCell (s.mem "output" 0), natCell (s.mem "output" 1))) =
      some (some Bool.false, some 1729) := by
  simp [one32, source32, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend32, initial32, State.setReg, State.write, Cell.read,
    boolCell, natCell, ComputeDType.eraseDType, FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast,
    TileShape.allIndices, TileShape.indexToList, BlockPtr.inBounds, BlockPtr.address,
    finite32, exponent32, bits32, Option.bind, List.finRange_succ] <;> decide

theorem whole64_infinity (rest : Algebra Float64Bits) :
    (FP.Structural.exec (engine (backend64 rest)) one64
      (initial64 (fun _ => bits64 0xfff0000000000000))).map
        (fun s => (boolCell (s.mem "output" 0), natCell (s.mem "output" 1))) =
      some (some Bool.false, some 1729) := by
  simp [one64, source64, FP.Structural.exec, ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.evalOffsets, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.bop, engine, Algebra.withDefaultPrecision, resolvePrecision,
    backend64, initial64, State.setReg, State.write, Cell.read,
    boolCell, natCell, ComputeDType.eraseDType, FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast,
    TileShape.allIndices, TileShape.indexToList, BlockPtr.inBounds, BlockPtr.address,
    finite64, exponent64, bits64, Option.bind, List.finRange_succ] <;> decide

#axiomsClean whole32_infinity
#axiomsClean whole64_infinity
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
