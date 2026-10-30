import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.softmax_optimize.Proofs.Source
import VeriTile.Triton.Memory.RegionalKernelIO

/-! The selected complete four-loop row kernel. Logical IO covers N elements,
even when the source's selected tile is shorter. Real uses the exact typed
source's mathematical dtype projection; source FP32 precision is unchanged. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.SoftmaxIO
open VeriTile.Triton
noncomputable section

def io (Y X : RegionName) (M N : Nat) : UKernelIO where
  kernel := (kernel Y X M N (Config.tile N)).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType, Source.projects,
      ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [X,Y]
  ity := fun _ => .float
  iarity := fun _ => N
  ibuf := fun _ => X
  oty := fun _ => .float
  oarity := fun _ => N
  obuf := fun _ => Y
  obuf_mem := fun _ => by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun _ _ row _ _ j => row*N+j.val
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ row _ _ j => row*N+j.val
  omask := fun _ _ _ _ _ _ => True
  swin := by intro t; exact Fin.elim0 t
  smask := by intro t; exact Fin.elim0 t

/-- Independent unpadded mathematical row softmax. -/
def rowSoftmax {N : Nat} (xs : Fin N → ℝ) (j : Fin N) : ℝ :=
  Real.exp (xs j) / ∑ i : Fin N, Real.exp (xs i)

def target (Y X : RegionName) (M N : Nat) :
    Nat → Nat → (io Y X M N).Ctx → (o : Fin (io Y X M N).nOut) →
      Fin ((io Y X M N).oarity o) → ((io Y X M N).oty o).carrier :=
  fun _ _ vals _ j => rowSoftmax (vals (0 : Fin 1)) j

/-- The complete N-element logical row, retaining all original PID-dependent
windows in the core IO and covering every full/tail tile of both passes. -/
def rowInputs (Y X : RegionName) (M N : Nat) :
    (io Y X M N).Ctx ≃ (Fin N → ℝ) where
  toFun := fun xs => xs (0 : Fin 1)
  invFun := fun xs _ => xs
  left_inv := by intro xs; funext channel; have h : channel=(0 : Fin 1) := Fin.eq_zero channel; subst channel; rfl
  right_inv := fun _ => rfl

def rowOutputs (Y X : RegionName) (M N : Nat) :
    ((o : Fin (io Y X M N).nOut) → Fin ((io Y X M N).oarity o) →
      ((io Y X M N).oty o).carrier) ≃ (Fin N → ℝ) :=
  rowInputs Y X M N


/-- Reindex the exact regional single-invocation contract. Arbitrary initial
program IDs, regional windows, source safety, execution and frame remain. -/
def logicalIO (Y X : RegionName) (M N : Nat) :
    LogicalIO (io Y X M N).Ctx
      ((o : Fin (io Y X M N).nOut) → Fin ((io Y X M N).oarity o) →
        ((io Y X M N).oty o).carrier)
      (Fin N → ℝ) (Fin N → ℝ)
      (fun f => (io Y X M N).RegionalImplements (fun _ _ values => f values)) where
  inputs := rowInputs Y X M N
  outputs := rowOutputs Y X M N

end
end VeriTile.Bench.Repairs.SoftmaxOptimize.SoftmaxIO
