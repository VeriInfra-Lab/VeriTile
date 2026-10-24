import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.InputBinding
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelIndex
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelFP

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelBinding
open VeriTile.Triton FP.Structural
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNative (Payload)
noncomputable section

/-- Ordinary already-supplied output buffer for one invocation. Its span and
input separation are the only output premises; no allocation, returned Tensor,
or computed output value is assumed. Input buffers need not be disjoint. -/
structure OutputMemory (N M matrixBase vectorBase : Nat) where
  base : Nat
  span : 0 < N → 4 * base + 4 * N ≤ 2^64
  matrixSeparate : ∀ i r j, i < N → r < N → j < M →
    base + i ≠ matrixBase + r * M + j
  vectorSeparate : ∀ i j, i < N → j < M → base + i ≠ vectorBase + j

variable (x y : FP32.Tensor) (lx : FP32.TensorLaws x) (ly : FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)

abbrev Output := OutputMemory (rows x.metadata) (columns x.metadata)
  (TensorBinding.matrixBase x y lx h) (TensorBinding.vectorBase x y ly h)

/-- Input bounds and aligned word bases are derived from checked metadata and
shared Tensor storage laws. Only the output buffer is supplied externally. -/
def memory (out : Output x y lx ly h) : KernelMemory (rows x.metadata) (columns x.metadata) where
  matrixBase := TensorBinding.matrixBase x y lx h
  vectorBase := TensorBinding.vectorBase x y ly h
  outputBase := out.base
  matrixBound := by
    intro i j hi hj
    have idx := TensorBinding.matrix_index x y h i j hi hj
    have span := (TensorBinding.matrixView x y lx h).spanFits
    change (TensorBinding.matrixView x y lx h).base + 4 * x.metadata.numel ≤ 2^64 at span
    rw [←TensorBinding.matrix_base_bytes x y lx h] at span
    omega
  vectorBound := by
    intro j hj
    have idx : j < y.metadata.numel := by
      rw [vector_numel x.metadata y.metadata h]
      exact hj
    have span := (TensorBinding.vectorView x y ly h).spanFits
    change (TensorBinding.vectorView x y ly h).base + 4 * y.metadata.numel ≤ 2^64 at span
    rw [←TensorBinding.vector_base_bytes x y ly h] at span
    omega
  outputBound := by
    intro i hi
    have span := out.span (by omega)
    omega
  matrixSeparate := out.matrixSeparate
  vectorSeparate := out.vectorSeparate

/-- The checked actual matrix pointer, as a natural byte address. Empty data
pointers remain unused and have no required binding. -/
theorem matrix_pointer (nonempty : 0 < x.metadata.numel) :
    x.metadata.dataPtr.toNat = 4 * TensorBinding.matrixBase x y lx h := by
  have ptr := TensorBinding.matrix_pointer x y lx h nonempty
  have span := (TensorBinding.matrixView x y lx h).spanFits
  change (TensorBinding.matrixView x y lx h).base + 4 * x.metadata.numel ≤ 2^64 at span
  rw [←TensorBinding.matrix_base_bytes x y lx h] at span
  have bound : 4 * TensorBinding.matrixBase x y lx h < 2^64 := by omega
  simpa only [FP32.Tensor.dataPtr, BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound] using
    (congrArg BitVec.toNat ptr).symm

theorem vector_pointer (nonempty : 0 < y.metadata.numel) :
    y.metadata.dataPtr.toNat = 4 * TensorBinding.vectorBase x y ly h := by
  have ptr := TensorBinding.vector_pointer x y ly h nonempty
  have span := (TensorBinding.vectorView x y ly h).spanFits
  change (TensorBinding.vectorView x y ly h).base + 4 * y.metadata.numel ≤ 2^64 at span
  rw [←TensorBinding.vector_base_bytes x y ly h] at span
  have bound : 4 * TensorBinding.vectorBase x y ly h < 2^64 := by omega
  simpa only [FP32.Tensor.dataPtr, BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound] using
    (congrArg BitVec.toNat ptr).symm

/-- Metadata admission and ordinary legal invocation facts yield the exact
KernelContract for the memory derived from the actual input tensors. -/
theorem contract (out : Output x y lx ly h) (c : KernelConfig) (pid : Nat)
    (program : pid < (rows x.metadata + c.blockRows - 1) / c.blockRows)
    (programABI : pid < 2^31) :
    KernelContract x.metadata y.metadata c pid (memory x y lx ly h out) :=
  ⟨h, matrix_pointer x y lx h, vector_pointer x y ly h, program, programABI⟩

/-- One physical byte-addressed FP32 memory supplies all three named buffers.
A and B aliases read the same physical word. Other typed cells and all private
registers are taken from the arbitrary surrounding state. -/
def state {α : Type} (out : Output x y lx ly h) (decode : FP32.Word → α)
    (read : Nat → FP32.Word) (pid : Nat) (surrounding : State α) : State α :=
  { surrounding with
    mem := fun r i =>
      if r = "A" then .mk .fp32 (decode (read (4 * (TensorBinding.matrixBase x y lx h + i))))
      else if r = "B" then .mk .fp32 (decode (read (4 * (TensorBinding.vectorBase x y ly h + i))))
      else if r = "C" then .mk .fp32 (decode (read (4 * (out.base + i))))
      else surrounding.mem r i
    pids := fun ax => if ax = 0 then pid else surrounding.pids ax }

@[simp] theorem state_pid {α : Type} (out : Output x y lx ly h) (decode : FP32.Word → α)
    (read : Nat → FP32.Word) (pid : Nat) (surrounding : State α) :
    (state x y lx ly h out decode read pid surrounding).pids 0 = pid := by simp [state]

/-- The value at each actual matrix load is the original Tensor's logical
FP32 value; this derives the previously abstract mathematical input premise. -/
theorem matrix_read {α : Type} [Inhabited α] (out : Output x y lx ly h)
    (decode : FP32.Word → α) (read : Nat → FP32.Word) (pid : Nat) (surrounding : State α)
    (i j : Nat) (hi : i < rows x.metadata) (hj : j < columns x.metadata) :
    ((state x y lx ly h out decode read pid surrounding).mem "A"
      (i * columns x.metadata + j)).read .fp32 = TensorBinding.matrix x lx decode read i j := by
  simp only [state, if_pos rfl, Cell.read_same]
  have idx := TensorBinding.matrix_index x y h i j hi hj
  have value := congrArg decode (FP32.direct_read_eq_logical read x lx
    h.contiguous.1 h.noNeg.1 (i * columns x.metadata + j) idx)
  change decode (read (4 * (TensorBinding.matrixBase x y lx h + (i * columns x.metadata + j)))) = _
  rw [show 4 * (TensorBinding.matrixBase x y lx h + (i * columns x.metadata + j)) =
      (TensorBinding.matrixView x y lx h).address (i * columns x.metadata + j) by
    simp only [FP32.DataView.address]
    rw [←TensorBinding.matrix_base_bytes x y lx h]
    omega]
  exact value

theorem vector_read {α : Type} [Inhabited α] (out : Output x y lx ly h)
    (decode : FP32.Word → α) (read : Nat → FP32.Word) (pid : Nat) (surrounding : State α)
    (j : Nat) (hj : j < columns x.metadata) :
    ((state x y lx ly h out decode read pid surrounding).mem "B" j).read .fp32 =
      TensorBinding.vector y ly decode read j := by
  simp only [state, show ("B" : RegionName) ≠ "A" by decide, if_neg, if_pos rfl, Cell.read_same]
  have idx : j < y.metadata.numel := by rw [vector_numel x.metadata y.metadata h]; exact hj
  have value := congrArg decode (FP32.direct_read_eq_logical read y ly
    h.contiguous.2 h.noNeg.2 j idx)
  change decode (read (4 * (TensorBinding.vectorBase x y ly h + j))) = _
  rw [show 4 * (TensorBinding.vectorBase x y ly h + j) =
      (TensorBinding.vectorView x y ly h).address j by
    simp only [FP32.DataView.address]
    rw [←TensorBinding.vector_base_bytes x y ly h]
    omega]
  exact value

/-- Physical input aliases are coherent by construction; no A/B disjointness
premise is introduced by the State bridge. -/
theorem aliases {α : Type} (out : Output x y lx ly h) (decode : FP32.Word → α)
    (read : Nat → FP32.Word) (pid : Nat) (surrounding : State α) (a b : Nat)
    (sameAddress : TensorBinding.matrixBase x y lx h + a = TensorBinding.vectorBase x y ly h + b) :
    (state x y lx ly h out decode read pid surrounding).mem "A" a =
      (state x y lx ly h out decode read pid surrounding).mem "B" b := by
  simp [state, sameAddress, show ("B" : RegionName) ≠ "A" by decide]

abbrev bits (w : FP32.Word) : BitVec 32 := w.bits

def fpState (out : Output x y lx ly h) (read : Nat → FP32.Word) (pid : Nat)
    (surrounding : State Payload) : State Payload :=
  state x y lx ly h out (fun w => Payload.word w.bits) read pid surrounding

/-- The actual Tensor laws discharge InputWords, including every source mask,
without assuming an accumulator, reduction, or output-value equation. -/
theorem input_words (out : Output x y lx ly h) (read : Nat → FP32.Word) (pid : Nat)
    (surrounding : State Payload) :
    KernelFP.InputWords (rows x.metadata) (columns x.metadata)
      (fpState x y lx ly h out read pid surrounding)
      (TensorBinding.matrix x lx bits read) (TensorBinding.vector y ly bits read) := by
  constructor
  · intro i j hi hj
    exact matrix_read x y lx ly h out (fun w => Payload.word w.bits) read pid surrounding i j hi hj
  · intro j hj
    exact vector_read x y lx ly h out (fun w => Payload.word w.bits) read pid surrounding j hj

#axiomsClean memory
#axiomsClean contract
#axiomsClean matrix_read
#axiomsClean vector_read
#axiomsClean aliases
#axiomsClean input_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelBinding
