import bench.tritonbench_g.quant_transpose_kernel.Kernels
import bench.tritonbench_g.quantize_global.Proofs.IntegerSemantics
local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace QuantizeIntegerStage.Transpose
open VeriTile.Triton FP.Structural QuantizeIntegerStage.FP32
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def kernel (A AbsmaxInv B : RegionName)
    (sam san sbn sbm M N BM BN : Nat) : TypedKernel :=
  ⟨source A AbsmaxInv B sam san sbn sbm M N BM BN, byteOutput B, outputRegister⟩

def row (s : State Word) (BM : Nat) (i : Fin BM) := s.pids 0 * BM + i.val
def col (s : State Word) (BN : Nat) (j : Fin BN) := s.pids 1 * BN + j.val

def active (s : State Word) (M N BM BN : Nat) (i : TileIndex [BM, BN]) : Prop :=
  row s BM i.1 < M ∧ col s BN i.2.1 < N
instance (s : State Word) (M N BM BN : Nat) (i : TileIndex [BM, BN]) :
    Decidable (active s M N BM BN i) := inferInstanceAs (Decidable (_ ∧ _))

def inputOffset (s : State Word) (sam san BM BN : Nat) (i : TileIndex [BM, BN]) :=
  row s BM i.1 * sam + col s BN i.2.1 * san

def outputOffset (s : State Word) (sbn sbm BM BN : Nat) (i : TileIndex [BM, BN]) :=
  row s BM i.1 * sbm + col s BN i.2.1 * sbn

/-- Independent finite-word target: source-ordered fp32 multiplications,
llrint's signed 64-bit result, then signed int8 narrowing. -/
def word (round : Llrint64 Word) (cfg : FP.Config) (s : State Word) (A AbsmaxInv : RegionName)
    (sam san BM BN : Nat) (i : TileIndex [BM, BN]) : BitVec 8 :=
  outputWord round cfg ((s.mem A (inputOffset s sam san BM BN i)).read .real)
    ((s.mem AbsmaxInv 0).read .real)

def written (round : Llrint64 Word) (cfg : FP.Config) (s : State Word) (A AbsmaxInv B : RegionName)
    (sam san sbn sbm M N BM BN : Nat) : State Word :=
  (TileShape.allIndices [BM, BN]).foldl (fun t i =>
    if active s M N BM BN i then
      t.write B (outputOffset s sbn sbm BM BN i) (.mk .int (word round cfg s A AbsmaxInv sam san BM BN i).toInt)
    else t) s

theorem source_run (rest : Algebra Word) (round : Llrint64 Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat) :
    ∃ t, QuantizeIntegerStage.exec (backend rest cfg) (some round) (kernel A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      t.mem = (written round cfg s A AbsmaxInv B sam san sbn sbm M N BM BN).mem := by
  simp +contextual (config := { maxSteps := 1000000 }) [kernel, source, ComputeKernel.surfaceBody,
    QuantizeIntegerStage.exec, QuantizeIntegerStage.run, QuantizeIntegerStage.step,
    QuantizeIntegerStage.evalExpr, QuantizeIntegerStage.store, storedInteger, byteOutput, outputRegister,
    FP.Structural.step, stepAlg, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    FP.Structural.store, backend, FP.Software.scalarFP32, FP.Software.fp32Context,
    State.setReg, TileShape.axisDim, written, word, outputWord, scaled, active, inputOffset, outputOffset, row, col,
    ofFloat, toFloat, Region.cast, ComputeDType.eraseDType, bop, numeric,
    numericLt, natLt, Option.bind]
  simp only [fold_memory]
  simp +contextual [show (127.0 : ℝ) = 127 by norm_num]

#axiomsClean source_run

end
end QuantizeIntegerStage.Transpose
