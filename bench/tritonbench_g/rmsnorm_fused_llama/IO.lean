import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.RealGrid
import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import Mathlib.Logic.Equiv.Fin.Basic

/-! Real interpretation of both complete source variants. The finite channel
windows describe the entire row grid; all actual two-pass iterations are kept
for arbitrary column counts. Source precision is fixed by Kernels.lean. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.IO
open VeriTile Triton
noncomputable section
set_option maxRecDepth 30000

def source (candidate : Bool) (X Y W : RegionName) (N B : Nat) (epsilon : ℝ) : ComputeKernel :=
  (if candidate then indexCommutedKernel X Y W N B epsilon else kernel X Y W N B epsilon).eraseDType

def launch (rows row : Nat) : AlgorithmLaunch where
  pids := fun ax => if ax=0 then row else 0
  numPids := some (fun ax => if ax=0 then rows else 1)

def schedule (rows count : Nat) : List AlgorithmLaunch :=
  (List.range count).map (launch rows)

def grid (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ) : UKernelIO where
  kernel := source candidate X Y W N B epsilon
  projection := by
    have h : (if candidate then indexCommutedKernel X Y W N B epsilon else kernel X Y W N B epsilon).toAlgorithm? =
        .ok (if candidate then indexCommutedKernel X Y W N B epsilon else kernel X Y W N B epsilon).toAlgKernel := by
      cases candidate <;> rfl
    simp only [source,ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [X,W,Y]
  ity := fun _ => .float
  iarity := fun i => if i.val=0 then rows*N else N
  ibuf := fun i => if i.val=0 then X else W
  oty := fun _ => .float
  oarity := fun _ => rows*N
  obuf := fun _ => Y
  obuf_mem := fun _ => by simp
  sarity := fun i => Fin.elim0 i
  sbuf := fun i => Fin.elim0 i
  iwin := fun _ _ _ _ _ j => j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ _ _ _ j => j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

def vector (xs : Fin n → ℝ) (i : Nat) : ℝ := if h : i<n then xs ⟨i,h⟩ else 0


/-- Lossless row-major view of the complete matrix input. -/
def matrixValues (rows N : Nat) : (Fin (rows*N) → ℝ) ≃ (Fin rows → Fin N → ℝ) where
  toFun := fun xs row col => xs (finProdFinEquiv (row,col))
  invFun := fun xs i => xs (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
  left_inv := by intro xs; funext i; exact congrArg xs (finProdFinEquiv.apply_symm_apply i)
  right_inv := by intro xs; funext row col; simp

/-- Matrix X and vector W, retaining both complete original channels. -/
def logicalInputs (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ) :
    (grid candidate X Y W rows N B epsilon).Ctx ≃
      ((Fin rows → Fin N → ℝ) × (Fin N → ℝ)) := by
  exact (piFinTwoEquiv _).trans (Equiv.prodCongr (matrixValues rows N) (Equiv.refl _))

/-- The sole flat result channel is exactly the row/column matrix. -/
def logicalIndices (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ) :
    ((o : Fin (grid candidate X Y W rows N B epsilon).nOut) ×
      Fin ((grid candidate X Y W rows N B epsilon).oarity o)) ≃ (Fin rows × Fin N) := by
  change ((o : Fin 1) × Fin (rows*N)) ≃ (Fin rows × Fin N)
  exact ({ toFun := fun cell => cell.2
           invFun := fun i => ⟨0,i⟩
           left_inv := by rintro ⟨o,i⟩; have h : o=0 := Fin.eq_zero o; subst o; rfl
           right_inv := fun _ => rfl } : _ ≃ Fin (rows*N)).trans finProdFinEquiv.symm

/-- Unchanged complete canonical grid, including empty dimensions. -/
def logicalIO (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ) :
    LogicalScheduledKernelIO ((Fin rows → Fin N → ℝ) × (Fin N → ℝ)) (Fin rows × Fin N) where
  core := grid candidate X Y W rows N B epsilon
  schedule := fun _ => schedule rows rows
  inputs := logicalInputs candidate X Y W rows N B epsilon
  indices := logicalIndices candidate X Y W rows N B epsilon
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.IO

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Reference
open VeriTile Triton FP.Structural
noncomputable section

/-- Independent RMS normalization over all N columns of each logical row. -/
def real (candidate : Bool) (X Y W : RegionName) (rows N B : Nat) (epsilon : ℝ)
    (a : (IO.grid candidate X Y W rows N B epsilon).Ctx)
    (_ : Fin 1) (j : Fin (rows*N)) : ℝ :=
  RealGrid.value (IO.vector (a (0 : Fin 2))) (IO.vector (a (1 : Fin 2)))
    (j.val/N) N epsilon (j.val%N)


#specNonCircular RealGrid.value avoiding [kernel,RealGrid.run]

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Reference
