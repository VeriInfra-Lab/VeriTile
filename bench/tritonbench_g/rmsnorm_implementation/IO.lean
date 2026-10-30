import bench.tritonbench_g.rmsnorm_implementation.Contract
import bench.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Triton.Memory.LogicalScheduledKernelIO
import Mathlib.Logic.Equiv.Fin.Basic

/-! Public finite row IO for the canonical FP32 source and its discrete-index
candidate. The mathematical projection erases FP32 only in Correct. Input and
weight channels may represent coherent read-only aliases. There is no scratch.
A singleton identity launch retains all initial program IDs and undefined values. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.Interface
open VeriTile Triton
noncomputable section

def source (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) : ComputeKernel :=
  if candidate then indexCommutedKernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps
  else kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n blockSize eps

def rowIO (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) : UKernelIO where
  kernel := (source candidate sxb sxm sxk sw sob som sok n eps).eraseDType
  projection := by
    have h : (source candidate sxb sxm sxk sw sob som sok n eps).toAlgorithm? =
        .ok (source candidate sxb sxm sxk sw sob som sok n eps).toAlgKernel := by
      cases candidate <;> rfl
    simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := ["x","weights","out"]
  ity := fun _ => .float
  iarity := fun _ => n
  ibuf := fun i => if i.val=0 then "x" else "weights"
  oty := fun _ => .float
  oarity := fun _ => n
  obuf := fun _ => "out"
  obuf_mem := fun _ => by simp
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ p₀ p₁ _ j => if i.val=0 then p₀*sxb+p₁*sxm+j.val*sxk else j.val*sw
  imask := fun _ _ _ _ _ _ => True
  owin := fun _ _ p₀ p₁ _ j => p₀*sob+p₁*som+j.val*sok
  omask := fun _ _ _ _ _ _ => True
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i

/-- One invocation using the initial launch coordinates, with no memory or
register installation. Both complete source loops execute in that invocation. -/
def oneProgram {candidate : Bool} {sxb sxm sxk sw sob som sok n : Nat} {eps : ℝ}
    (_ : (rowIO candidate sxb sxm sxk sw sob som sok n eps).Ctx) : List AlgorithmLaunch :=
  [{ pids := fun _ => 0, setPids := fun _ => Bool.false }]

/-- Only active feature addresses must be unique; zero output stride remains
valid for empty and singleton feature domains. Leading strides are unrestricted. -/
def OutputUnique (n sok : Nat) : Prop := Function.Injective (fun k : Fin n => k.val*sok)

/-- Independent finite-array RMS normalization. No execution, memory, reduction
implementation or source reference occurs in this mathematical formula. -/
def formula {n : Nat} (eps : ℝ) (x w : Fin n → ℝ) (k : Fin n) : ℝ :=
  (x k / Real.sqrt ((∑ j : Fin n, (x j)^2)/(n : ℝ)+eps))*w k


/-- Both complete input vectors, with no channel selector in the formula. -/
def logicalInputs (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    (rowIO candidate sxb sxm sxk sw sob som sok n eps).Ctx ≃
      ((Fin n → ℝ) × (Fin n → ℝ)) := by
  change (Fin 2 → Fin n → ℝ) ≃ _
  exact finTwoArrowEquiv _

/-- The only output channel is the logical feature vector. -/
def logicalIndices (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    ((o : Fin (rowIO candidate sxb sxm sxk sw sob som sok n eps).nOut) ×
      Fin ((rowIO candidate sxb sxm sxk sw sob som sok n eps).oarity o)) ≃ Fin n := by
  change ((o : Fin 1) × Fin n) ≃ Fin n
  exact {
  toFun := fun i => i.2
  invFun := fun k => ⟨0,k⟩
  left_inv := by rintro ⟨o,k⟩; have h : o=0 := Fin.eq_zero o; subst o; rfl
  right_inv := fun _ => rfl }

/-- The exact original singleton launch and regional memory contract. -/
def logicalIO (candidate : Bool) (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ) :
    LogicalScheduledKernelIO ((Fin n → ℝ) × (Fin n → ℝ)) (Fin n) where
  core := rowIO candidate sxb sxm sxk sw sob som sok n eps
  schedule := oneProgram
  inputs := logicalInputs candidate sxb sxm sxk sw sob som sok n eps
  indices := logicalIndices candidate sxb sxm sxk sw sob som sok n eps
  realOutput := fun _ => rfl

end
end VeriTile.Bench.Repairs.RmsnormImplementation.Interface
