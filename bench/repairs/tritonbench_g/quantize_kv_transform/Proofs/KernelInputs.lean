import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelExecution
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.Contract
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.FP

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelInputs
open VeriTile.Triton FP.Structural
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

def parameters (m : Inputs) (words : Nat → MetadataWord) : KernelSource.Parameters :=
  ⟨(request m words).heads, (request m words).columns, headBlock (request m words), columnBlock (request m words)⟩

def inputCell (p : Request) (token head column : Nat) : Nat :=
  token * (p.heads * p.columns) + head * p.columns + column

theorem width_positive (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words) :
    0 < (parameters m words).width := by unfold parameters columnBlock; positivity

theorem heads_covered (m : Inputs) (words : Nat → MetadataWord) :
    (parameters m words).heads ≤ (parameters m words).height := headBlock_covers _

theorem columns_covered (m : Inputs) (words : Nat → MetadataWord) :
    (request m words).columns ≤ (parameters m words).width := columnBlock_covers _

/-- The mathematical input is an arbitrary logical Tensor. No finite codec
restricts its Real values. Metadata retains the original signed-INT32 words. -/
structure RealReads (m : Inputs) (words : Nat → MetadataWord)
    (values : Nat → Nat → Nat → ℝ) (s : State ℝ) : Prop where
  input : ∀ token head column, token < (request m words).tokens →
    head < (request m words).heads → column < (request m words).columns →
    (s.mem "input" (inputCell (request m words) token head column)).read .real =
      values token head column
  destination : ∀ token, token < (request m words).tokens →
    (s.mem "destination" token).read .int = (words token).toInt

/-- Initial native read interpretation, using the same caller data pointers
and one physical byte heap as TensorBinding. These are input-only equations;
neither a successful execution nor a desired output is assumed. -/
structure NativeReads (m : Inputs) (words : Nat → MetadataWord)
    (heap : Heap) (s : State Word) : Prop where
  input : ∀ token head column, token < (request m words).tokens →
    head < (request m words).heads → column < (request m words).columns →
    (s.mem "input" (inputCell (request m words) token head column)).read .real =
      decodeWord (fun byte => heap (base m .input +
        4 * inputCell (request m words) token head column + byte.val))
  destination : ∀ token, token < (request m words).tokens →
    (s.mem "destination" token).read .int =
      (loadDest heap (base m .destination + 4 * token) : Int)

theorem real_destination (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (values : Nat → Nat → Nat → ℝ) (s : State ℝ)
    (reads : RealReads m words values s) (token : Nat) (ht : token < (request m words).tokens) :
    (s.mem "destination" token).read .int = ((request m words).destination token : Int) := by
  rw [reads.destination token ht]
  exact (Int.toNat_of_nonneg (h.bounds token ht).1).symm

theorem native_destination (m : Inputs) (words : Nat → MetadataWord)
    (heap : Heap) (s : State Word) (binding : TensorBinding m words heap)
    (reads : NativeReads m words heap s) (token : Nat)
    (ht : token < (request m words).tokens) :
    (s.mem "destination" token).read .int = ((request m words).destination token : Int) := by
  rw [reads.destination token ht, binding.metadata token ht]
  rfl

/-- The typed logical FP32 read is the existing physical Tensor input reader.
This reuses the prior byte layout and source precision rather than duplicating it. -/
theorem native_input (m : Inputs) (words : Nat → MetadataWord)
    (objects : Role → Nat) (heap : Heap) (s : State Word)
    (reads : NativeReads m words heap s) (token head column : Nat)
    (ht : token < (request m words).tokens) (hh : head < (request m words).heads)
    (hc : column < (request m words).columns) :
    (s.mem "input" (inputCell (request m words) token head column)).read .real =
      loadInput decodeWord (request m words) (directLayout m objects) heap token head column := by
  rw [reads.input token head column ht hh hc]
  simp only [loadInput,if_pos hc]
  congr 1
  funext byte
  congr 1
  simp [inputAddress, directLayout, Tensor.data, inputCell, Nat.mul_add, Nat.add_mul,
    Nat.mul_assoc, Nat.add_assoc]

#axiomsClean width_positive
#axiomsClean heads_covered
#axiomsClean real_destination
#axiomsClean native_destination
#axiomsClean native_input
#specNonCircular RealReads avoiding [KernelSource.kernel, FP.Structural.exec]
#specNonCircular NativeReads avoiding [KernelSource.kernel, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelInputs
