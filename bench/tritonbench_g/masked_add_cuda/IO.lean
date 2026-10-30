import bench.tritonbench_g.masked_add_cuda.Proofs.Graph
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Triton.Memory.LogicalIO


/-! Public native graph IO. The source fixes FP32 arithmetic and int32/int64
mask storage. Its selected backend folds either actual addition order into the
same FMA, retaining the runtime scalar's FP32 ABI word. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.NativeIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Typed interpreter channels; FP32 source regions are interpreted as the
`.real` opaque graph carrier, and the signed mask remains an integer. -/
def inputs : List IOChannel :=
  [⟨.real, [blockSize], "grad"⟩, ⟨.real, [blockSize], "data"⟩,
   ⟨.int, [blockSize], "mask"⟩]
def outputs : List IOChannel := [⟨.real, [blockSize], "grad"⟩]
abbrev Inputs := ChannelValues Graph.Expr inputs

def inputLane : (i : Fin inputs.length) → TileIndex inputs[i].shape → Fin blockSize
  | ⟨0, _⟩ => fun j => j.1
  | ⟨1, _⟩ => fun j => j.1
  | ⟨2, _⟩ => fun j => j.1
  | ⟨i + 3, h⟩ => False.elim (by simp [inputs] at h)

def outputLane : (o : Fin outputs.length) → TileIndex outputs[o].shape → Fin blockSize
  | ⟨0, _⟩ => fun j => j.1
  | ⟨i + 1, h⟩ => False.elim (by simp [outputs] at h)

def gradient (values : Inputs) (j : Fin blockSize) : Graph.Expr :=
  values ⟨0, by decide⟩ (j, PUnit.unit)
def data (values : Inputs) (j : Fin blockSize) : Graph.Expr :=
  values ⟨1, by decide⟩ (j, PUnit.unit)
def veto (values : Inputs) (j : Fin blockSize) : Int :=
  values ⟨2, by decide⟩ (j, PUnit.unit)

/-- Only source-active gradient values are constrained to be input word leaves.
Inactive/vetoed grad values and all data graph values remain unrestricted. -/
def LeafGradient (n pid : Nat) (values : Inputs) : Prop :=
  ∀ j : Fin blockSize, pid * blockSize + j.val < n → veto values j = 0 →
    ∃ word, gradient values j = Graph.Expr.leaf word

def invocation (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) (candidate : Bool) :
    Invocation Graph.Expr where
  kernel := if candidate then commutedKernel storage "grad" "data" "mask" n token
    else kernel storage "grad" "data" "mask" n token
  backend := Graph.nativeAlgebra rest alphaWord
  pids := fun ax => if ax = 0 then pid else 0

/-- Grad/data/mask read windows are explicit. Only the zero-veto part of grad
is writable; every other raw cell, including vetoed grad, remains in the frame. -/
def originalIO (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) :
    ProgramIO Graph.Expr Graph.Expr where
  inputs := inputs
  outputs := outputs
  scratch := []
  buffers := ["grad", "data", "mask"]
  read := fun i _ j => pid * blockSize + (inputLane i j).val
  inputMask := fun i _ j => pid * blockSize + (inputLane i j).val < n
  write := fun o _ j => pid * blockSize + (outputLane o j).val
  outputMask := fun o values j =>
    pid * blockSize + (outputLane o j).val < n ∧ veto values (outputLane o j) = 0
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [invocation rest alphaWord storage n pid token Bool.false]
  encode := id
  observe := fun _ _ => id

def candidate (rest : Algebra Graph.Expr) (alphaWord : Graph.Word)
    (storage : MaskStorage) (n pid : Nat) (token : ℝ) :
    ProgramIO.Variant (originalIO rest alphaWord storage n pid token) where
  schedule := fun _ => [invocation rest alphaWord storage n pid token Bool.true]
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.MaskedAddCuda.NativeIO

namespace VeriTile.Bench.Repairs.MaskedAddCuda.RealIO
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.MaskedAddCuda
noncomputable section

/-- The FP32 in-place source, projected to Real. The signed integer veto is a
separate logical channel; it is read under the bounds mask, while only zero
mask values enable a store. Read-only data/mask regions need no separation. -/
def io (storage : MaskStorage) (n : Nat) (alpha : ℝ) : UKernelIO where
  kernel := (kernel storage "grad" "data" "mask" n alpha).eraseDType
  projection := by
    rw [kernel_eq_original]
    exact integer_supported storage Bool.false "grad" "data" "mask" n alpha blockSize
  nIn := 3
  nOut := 1
  nScr := 0
  bufs := ["grad", "data", "mask"]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | _ => .int
  iarity := fun _ => blockSize
  ibuf := fun i => match i with
    | ⟨0, _⟩ => "grad"
    | ⟨1, _⟩ => "data"
    | _ => "mask"
  oty := fun _ => .float
  oarity := fun _ => blockSize
  obuf := fun _ => "grad"
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun _ _ pid _ _ j => pid * blockSize + j.val
  imask := fun _ _ pid _ _ j => pid * blockSize + j.val < n
  owin := fun _ _ pid _ _ j => pid * blockSize + j.val
  omask := fun _ vals pid _ _ j =>
    pid * blockSize + j.val < n ∧ vals (⟨2, by decide⟩ : Fin 3) j = 0
  swin := fun t => t.elim0
  smask := fun t => t.elim0

end
end VeriTile.Bench.Repairs.MaskedAddCuda.RealIO

namespace VeriTile.Bench.Repairs.MaskedAddCuda.ReaderIO
open VeriTile Triton
noncomputable section

/-- Every declared source lane, including inactive lanes, is retained. -/
structure Inputs where
  gradient : Fin blockSize → ℝ
  data : Fin blockSize → ℝ
  mask : Fin blockSize → Int

def inputs (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    (RealIO.io storage n alpha).Ctx ≃ Inputs where
  toFun := fun xs => ⟨xs (0 : Fin 3), xs (1 : Fin 3), xs (2 : Fin 3)⟩
  invFun := fun xs => Fin.cases xs.gradient (Fin.cases xs.data (Fin.cases xs.mask (fun i => i.elim0)))
  left_inv := by intro xs; funext i j; fin_cases i <;> rfl
  right_inv := by intro xs; cases xs; rfl

abbrev RawOutputs (storage : MaskStorage) (n : Nat) (alpha : ℝ) :=
  Nat → Nat → (o : Fin (RealIO.io storage n alpha).nOut) →
    Fin ((RealIO.io storage n alpha).oarity o) → ((RealIO.io storage n alpha).oty o).carrier

def outputs (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    RawOutputs storage n alpha ≃ (Nat → Nat → Fin blockSize → ℝ) where
  toFun := fun ys p₀ p₁ j => ys p₀ p₁ (0 : Fin 1) j
  invFun := fun ys p₀ p₁ => Fin.cases (ys p₀ p₁) (fun i => i.elim0)
  left_inv := by intro ys; funext p₀ p₁ o j; fin_cases o; rfl
  right_inv := fun _ => rfl

/-- A regional contract over all original launch coordinates. This is the
original complete RegionalImplements judgment, including data-dependent stores. -/
def view (storage : MaskStorage) (n : Nat) (alpha : ℝ) :
    LogicalIO (RealIO.io storage n alpha).Ctx (RawOutputs storage n alpha)
      Inputs (Nat → Nat → Fin blockSize → ℝ)
      (fun f => (RealIO.io storage n alpha).RegionalImplements
        (fun p₀ p₁ xs => f xs p₀ p₁)) where
  inputs := inputs storage n alpha
  outputs := outputs storage n alpha
end
end VeriTile.Bench.Repairs.MaskedAddCuda.ReaderIO
