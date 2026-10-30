import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.reversed_cumsum.Proofs.Geometry
import VeriTile.Triton.Memory.AlgorithmProgramIO

/-! Public IO for one selected row and feature tile. Time is the outer lane
coordinate and the 32-feature tile is the inner coordinate. The feature mask
retains partial tiles; there are no hidden memory or execution premises.
FP32 load conversion, dot and storage precision remain in the shared source;
only this ideal mathematical interpretation erases its floating dtypes. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Interface
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Index a time/feature cell in the program's finite logical channel. -/
def lane (h : Call) (t : Fin h.T) (j : Fin BS) : Fin (h.T*BS) :=
  ⟨t.val*BS+j.val,by have ht:=t.isLt; have hj:=j.isLt; nlinarith⟩

def launch {h : Call} (program : Geometry.Program h) : AlgorithmLaunch where
  pids := fun ax => if ax=0 then program.2.val else program.1.val
  setPids := fun ax => decide (ax=0 ∨ ax=1)

def io (h : Call) (c : Config) (program : Geometry.Program h) : AlgorithmProgramIO where
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [h.input,h.output]
  ity := fun _ => .float
  iarity := fun _ => h.T*BS
  ibuf := fun _ => h.input
  oty := fun _ => .float
  oarity := fun _ => h.T*BS
  obuf := fun _ => h.output
  obuf_mem := fun _ => by simp
  sarity := fun k => k.elim0
  sbuf := fun k => k.elim0
  iwin := fun _ _ _ _ _ j =>
    Geometry.address h program.1.val (j.val/BS) (program.2.val*BS+j.val%BS)
  imask := fun _ _ _ _ _ j => program.2.val*BS+j.val%BS<h.S
  owin := fun _ _ _ _ _ j =>
    Geometry.address h program.1.val (j.val/BS) (program.2.val*BS+j.val%BS)
  omask := fun _ _ _ _ _ j => program.2.val*BS+j.val%BS<h.S
  swin := fun k => k.elim0
  smask := fun k => k.elim0
  initialUndef := .preserve
  stages := fun _ => [.single
    {kernel := (Kernels.source h c).eraseDType
     projection := by
       simp only [ComputeKernel.eraseDType,Kernels.source_projects]
       simp [ComputeKernel.toAlgKernel]}
    (launch program)]

abbrev Inputs (h : Call) := Fin 1 → Fin (h.T*BS) → ℝ

/-- Independent ordinary suffix sum for each feature of the selected row. -/
def reference (h : Call) (vals : Inputs h) (_ : Fin 1) (j : Fin (h.T*BS)) : ℝ :=
  ∑ t : Fin h.T, if j.val/BS≤t.val then
    vals 0 (lane h t ⟨j.val%BS,Nat.mod_lt _ (by decide)⟩) else 0

end
end VeriTile.Bench.Repairs.ReversedCumsum.Interface

/-! Logical time/feature coordinates for the complete existing single-program
IO. The bijections retain every tile value, including inactive feature lanes;
the original IO still supplies its mask, execution, safety and exterior frame. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Interface
open VeriTile.Triton
noncomputable section

abbrev Matrix (h : Call) := Fin h.T → Fin BS → ℝ

/-- Decode the one finite channel into its time and feature axes. -/
def matrixEquiv (h : Call) : Inputs h ≃ Matrix h where
  toFun := fun xs t j => xs 0 (lane h t j)
  invFun := fun xs _ j => xs
    ⟨j.val / BS, (Nat.div_lt_iff_lt_mul (by decide : 0 < BS)).mpr j.isLt⟩
    ⟨j.val % BS, Nat.mod_lt _ (by decide)⟩
  left_inv := by
    intro xs
    funext port j
    have hp : port = 0 := by omega
    subst port
    dsimp only
    congr 1
    apply Fin.ext
    simp only [lane]
    simpa only [Nat.mul_comm] using Nat.div_add_mod j.val BS
  right_inv := by
    intro xs
    funext t j
    dsimp only
    congr 1 <;> apply Fin.ext <;> simp only [lane]
    · have ht := t.isLt; have hj := j.isLt
      dsimp [BS] at *
      omega
    · simp [Nat.add_mod, Nat.mod_eq_of_lt j.isLt]

/-- Source, launch and full contract are the exact existing IO judgment. -/
def suffixIO (h : Call) (c : Config) (program : Geometry.Program h) :
    LogicalIO (Inputs h) (Inputs h) (Matrix h) (Matrix h)
      (io h c program).Implements where
  inputs := matrixEquiv h
  outputs := matrixEquiv h

end
end VeriTile.Bench.Repairs.ReversedCumsum.Interface
