import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Contract
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Mathematical
noncomputable section

/-- Finite logical input arrays. INT8 operands remain bytes; scales and bias
use the selected FP32 carrier. The bias channel is read only when enabled. -/
structure Inputs (s : Shape) (F : Type) where
  a : Fin s.M → Fin s.K → BitVec 8
  b : Fin s.K → Fin s.N → BitVec 8
  rowScale : Fin s.M → F
  columnScale : Fin s.N → F
  bias : Fin s.N → F

/-- Independent dense dequantized matrix multiplication. -/
def real (s : Shape) (hasBias : Bool) (input : Inputs s ℝ) (cell : MatrixCell s) : ℝ :=
  input.columnScale cell.2 * (input.rowScale cell.1 *
    ((List.ofFn (fun k : Fin s.K => ((input.a cell.1 k).toInt : ℝ) *
      ((input.b k cell.2).toInt : ℝ))).sum / 16129)) +
    if hasBias then input.bias cell.2 else 0

def left (s : Shape) (c : Config) (input : Inputs s F) (r : Fin s.M) (t z : Nat)
    (lane : Fin c.BK) : BitVec 8 :=
  if active : sourceK c t z lane.val < s.K then input.a r ⟨sourceK c t z lane.val,active⟩ else 0

def right (s : Shape) (c : Config) (input : Inputs s F) (n : Fin s.N) (t z : Nat)
    (lane : Fin c.BK) : BitVec 8 :=
  if active : sourceK c t z lane.val < s.K then input.b ⟨sourceK c t z lane.val,active⟩ n else 0

/-- Same fixed instruction trees and live accumulator as the retained native
reference; no overflow, exceptional-word or split-order cases are excluded. -/
def accumulator (ops : Ops I F H) (s : Shape) (c : Config) (selected : Selection s c)
    (input : Inputs s F) (cell : MatrixCell s) (split : Fin c.S) : Nat → I
  | 0 => ops.izero
  | t+1 => (selected.plan cell.1.val cell.2.val split.val t).eval ops
      (left s c input cell.1 t split.val) (right s c input cell.2 t split.val)
      (accumulator ops s c selected input cell split t)

def splitValue (ops : Ops I F H) (s : Shape) (c : Config) (selected : Selection s c)
    (divisor : F) (hasBias : Bool) (input : Inputs s F) (cell : MatrixCell s) (split : Fin c.S) : H :=
  let acc := accumulator ops s c selected input cell split (steps s c)
  let value := ops.narrow (ops.mul (input.columnScale cell.2)
    (ops.mul (input.rowScale cell.1) (ops.mul (ops.toFloat acc) divisor)))
  if hasBias && split.val==0 then ops.hadd value (ops.narrow (input.bias cell.2)) else value

/-- Exact INT8 / INT32 / FP32 / FP16 computation under the original fixed
instruction profile and each cell's own complete split linearization. -/
def native (profile : FixedISAProfile) (s : Shape) (c : Config) (selected : Selection s c)
    (hasBias : Bool) (order : Schedule s c) (input : Inputs s Word32) (cell : MatrixCell s) : Word16 :=
  cellFold (fpOps profile) c
    (splitValue (fpOps profile) s c selected Source.runtimeDivisor hasBias input cell)
    (projection cell order.events) (fpOps profile).hzero

#specNonCircular real avoiding [gridRun,partialValue,accumulation]
#specNonCircular native avoiding [gridRun,partialValue,accumulation]
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Mathematical
