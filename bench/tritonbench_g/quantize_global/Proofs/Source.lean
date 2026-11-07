import bench.tritonbench_g.quantize_global.Proofs.Model
import VeriTile.Triton.Float.SignedCast
namespace QuantizeGlobalRepair
open VeriTile.Triton FP.Structural QuantizeIntegerStage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem signed8 (value : Int) :
    (if 2 * ((value % 256).toNat % 256) < 256
     then max (value % 256) 0 % 256 else max (value % 256) 0 % 256 - 256) =
      value.bmod 256 := by
  simpa [IntegerExpr.signedBits, BitVec.toInt_ofInt] using signedBits_int_mod 8 value

def offset (s : State α) (base B : Nat) (i : TileIndex [B]) := s.pids 0 * B + i.1.val + base
def word [Inhabited α] (a : Scalar α) (s : State α)
    (X Maximum : RegionName) (base B : Nat) (i : TileIndex [B]) : BitVec 8 :=
  scalar a ((s.mem X (offset s base B i)).read .real) ((s.mem Maximum 0).read .real)
def written [Inhabited α] (a : Scalar α) (s : State α)
    (X Maximum Out : RegionName) (N base B : Nat) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if offset s base B i < N then
      t.write Out (offset s base B i) (.mk .int (word a s X Maximum base B i).toInt)
    else t) s

/-- The checked AST executes every assignment, comparison, safe masked load,
div, multiplication, clamp, signed64 llrint and modular int8 store. -/
theorem source_run [Inhabited α] (rest : Algebra α) (a : Scalar α) (s : State α)
    (X Maximum Out : RegionName) (N base B : Nat) :
    ∃ t, QuantizeIntegerStage.exec (backend rest a) (some a.round64)
      (kernel X Maximum Out N base B) s = some t ∧
      t.mem = (written a s X Maximum Out N base B).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [kernel, sourceKernel, ComputeKernel.surfaceBody,
    QuantizeIntegerStage.exec, QuantizeIntegerStage.run, QuantizeIntegerStage.step,
    QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store, storedInteger, byteOutput, outputRegister,
    FP.Structural.step, stepAlg, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    FP.Structural.store, FP.Structural.eval_intToSigned, backend, State.setReg, TileShape.axisDim, written, word, scalar,
    bounded, scaled, denominator, offset, ofFloat, toFloat, Region.cast,
    ComputeDType.eraseDType, bop, numeric, numericLt, numericEq, natLt, Option.bind,
    -Nat.cast_mul, -Nat.cast_add, ←Int.natCast_mul, ←Int.natCast_add, Int.toNat_natCast,
    Int.ofNat_lt, integralMod, signed8]
  simp only [fold_memory]
  simp +contextual [show (127.0 : ℝ) = 127 by norm_num]
  congr 1
  funext mem i r o
  by_cases hi : s.pids 0 * B + i.1.val + base < N <;>
    simp [-Nat.cast_mul, -Nat.cast_add, ←Int.natCast_mul, ←Int.natCast_add,
      Int.toNat_natCast, signed8, hi, show (0.0 : ℝ) = 0 by norm_num, show (1.0 : ℝ) = 1 by norm_num]

#axiomsClean source_run
end
end QuantizeGlobalRepair
