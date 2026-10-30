import bench.tritonbench_g.iv_dependent_matmul.Proofs.HostSource
import Mathlib.Tactic

/-! Original-source signed i32 counterexample. The compiler receipt retains
`mul i32` then `sext i32 to i64` for source `stride_cm * offs_cm`, and PTX
retains `mul.lo.s32`. This theorem checks that concrete source arithmetic;
it does not claim a GPU was executed. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.IndexOverflow
open VeriTile.Triton

/-- Scalar ABI values all fit signed i32; the intermediate product does not. -/
def rows : Nat := 65537
def columns : Nat := 32768
def inner : Nat := 32
def blockWidth : Nat := 32
def grid : Nat := ((rows+31)/32)*((columns+31)/32)
def pid : Nat := 2097152
def outputRow : Nat := (pid/((columns+31)/32))*32
def outputColumn : Nat := (pid%((columns+31)/32))*32

def sourceRowOffset : Int :=
  (BitVec.ofNat 32 columns * BitVec.ofNat 32 outputRow).toInt

def logicalOffset : Nat := outputRow*columns+outputColumn

/-- A valid launch and active lane address an element before the allocated C
base, while the mathematical matrix coordinate lies inside that allocation. -/
theorem active_store_underflows :
    pid<grid ∧ outputRow<rows ∧ outputColumn<columns ∧
    rows<2^31 ∧ columns<2^31 ∧ logicalOffset<rows*columns ∧
    logicalOffset=2147483648 ∧ sourceRowOffset = -2147483648 ∧
    4*sourceRowOffset = -8589934592 ∧ sourceRowOffset<0 := by decide

theorem no_unrestricted_signed_address_refinement :
    ¬ (∀ i<rows, ((BitVec.ofNat 32 columns)*(BitVec.ofNat 32 i)).toInt=(i*columns : Nat)) := by
  intro h
  have hh := h outputRow active_store_underflows.2.1
  have hbad : ((BitVec.ofNat 32 columns)*(BitVec.ofNat 32 outputRow)).toInt ≠ ((outputRow*columns : Nat) : Int) := by decide
  exact hbad hh

#axiomsClean active_store_underflows
#axiomsClean no_unrestricted_signed_address_refinement
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.IndexOverflow
