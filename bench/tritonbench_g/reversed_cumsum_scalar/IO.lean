import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Geometry
import bench.tritonbench_g.reversed_cumsum_scalar.Contract
import VeriTile.Triton.Memory.AlgorithmProgramIO
import VeriTile.Triton.Memory.LogicalIO

/-! Public source-connected row IO. The original strided input/output windows
and source precision are retained; only the mathematical execution projects
floating arithmetic to Real. The launch preserves all initial local registers
and unspecified-value oracles. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Interface
open VeriTile.Triton
noncomputable section

def launch {h : Call} (row : Fin (h.B*h.H)) : AlgorithmLaunch where
  pids := fun _ => row.val
  setPids := fun ax => decide (ax=0)

def io (h : Call) (c : Config) (row : Fin (h.B*h.H)) : AlgorithmProgramIO where
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [h.input,h.output]
  ity := fun _ => .float
  iarity := fun _ => h.T
  ibuf := fun _ => h.input
  oty := fun _ => .float
  oarity := fun _ => h.T
  obuf := fun _ => h.output
  obuf_mem := fun _ => by simp
  sarity := fun k => k.elim0
  sbuf := fun k => k.elim0
  iwin := fun _ _ _ _ _ j => h.inputStrides.address h.H row.val j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => h.outputStrides.address h.H row.val j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun k => k.elim0
  smask := fun k => k.elim0
  initialUndef := .preserve
  stages := fun _ => [.single
    {kernel := (Kernels.source h c).eraseDType
     projection := by
       simp only [ComputeKernel.eraseDType,Kernels.source_projects]
       simp [ComputeKernel.toAlgKernel]}
    (launch row)]

abbrev Inputs (h : Call) := Fin 1 → Fin h.T → ℝ

/-- Independent finite suffix sum, with no interpreter or execution state. -/
def reference (h : Call) (vals : Inputs h) (_ : Fin 1) (t : Fin h.T) : ℝ :=
  ∑ k : Fin h.T, if t.val≤k.val then vals 0 k else 0


/-- The sole complete row, with an inverse retaining every original input. -/
def rowValues (h : Call) : (Fin 1 → Fin h.T → ℝ) ≃ (Fin h.T → ℝ) where
  toFun := fun values => values 0
  invFun := fun values _ => values
  left_inv := by
    intro values
    funext channel
    have zero : channel=0 := Fin.eq_zero channel
    subst channel
    rfl
  right_inv := fun _ => rfl

/-- Same source row, original strided windows, stages, safety and frame. -/
def logicalIO (h : Call) (c : Config) (row : Fin (h.B*h.H)) :
    LogicalIO (Inputs h) (Fin 1 → Fin h.T → ℝ) (Fin h.T → ℝ) (Fin h.T → ℝ)
      (io h c row).Implements where
  inputs := rowValues h
  outputs := rowValues h

end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Interface
