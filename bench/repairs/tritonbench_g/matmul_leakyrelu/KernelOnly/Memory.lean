import bench.repairs.tritonbench_g.matmul_leakyrelu.KernelOnly.Kernels

/-! Task-local realization of the proved DSL output cells at actual FP16
Tensor addresses. Each cell occupies exactly two bytes. This adapter reads
the executed source state, never the mathematical/native reference. -/
namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section
set_option maxHeartbeats 3000000

def OutputByte (c : OutputTensor) (p : Nat) : Prop :=
  outputBase c ≤ p ∧ p < outputBase c+2*c.metadata.numel

instance (c : OutputTensor) (p : Nat) : Decidable (OutputByte c p) := by
  unfold OutputByte
  infer_instance

/-- Physical interpretation of the typed C view in an executed state. Input
and complement memory are inherited from the same initial physical heap. -/
def writeback {α β : Type} [Inhabited α] (c : OutputTensor)
    (encode : α → Fin 2 → β) (before : Nat → β) (final : State α) (p : Nat) : β :=
  if OutputByte c p then
    encode ((final.mem "c" ((p-outputBase c)/2)).read .fp16)
      ⟨(p-outputBase c)%2, Nat.mod_lt _ (by decide)⟩
  else before p

theorem writeback_at {α β : Type} [Inhabited α] (a b : Matrix)
    (c : OutputTensor) (cl : OutputLaws c a b) (encode : α → Fin 2 → β)
    (before : Nat → β) (final : State α) (i : Nat) (hi : i < c.metadata.numel) (byte : Fin 2) :
    writeback c encode before final (outputAddress c i+byte.val) =
      encode ((final.mem "c" i).read .fp16) byte := by
  rw [output_address a b c cl i hi]
  have hs : outputBase c+2*i+byte.val-outputBase c = 2*i+byte.val := by omega
  have hd : (2*i+byte.val)/2 = i := by omega
  have hm : (2*i+byte.val)%2 = byte.val := by omega
  have inside : OutputByte c (outputBase c+2*i+byte.val) := by
    unfold OutputByte
    have hb := byte.isLt
    omega
  simp [writeback, inside, hs, hd, hm]

theorem writeback_frame {α β : Type} [Inhabited α] (c : OutputTensor)
    (encode : α → Fin 2 → β) (before : Nat → β) (final : State α)
    (p : Nat) (outside : ¬OutputByte c p) : writeback c encode before final p = before p := by
  simp [writeback, outside]

theorem outside_of_disjoint (c : OutputTensor) (base size p : Nat)
    (sep : Whole.byteSpansDisjoint (outputBase c) (2*c.metadata.numel) base size)
    (inside : base ≤ p ∧ p < base+size) : ¬OutputByte c p := by
  unfold Whole.byteSpansDisjoint at sep
  unfold OutputByte
  rcases sep with h | h | h | h <;> omega

theorem backing_frame {α β : Type} [Inhabited α] (a b : Matrix)
    (c : OutputTensor) (fresh : Fresh a b c) (encode : α → Fin 2 → β)
    (before : Nat → β) (final : State α) :
    (∀ p, a.tensor.storage.base ≤ p → p < a.tensor.storage.base+a.storageBytes →
      writeback c encode before final p = before p) ∧
    (∀ p, b.tensor.storage.base ≤ p → p < b.tensor.storage.base+b.storageBytes →
      writeback c encode before final p = before p) := by
  constructor
  · intro p hp hq
    exact writeback_frame c encode before final p (outside_of_disjoint c _ _ p fresh.a ⟨hp,hq⟩)
  · intro p hp hq
    exact writeback_frame c encode before final p (outside_of_disjoint c _ _ p fresh.b ⟨hp,hq⟩)

theorem output_offset (a b : Matrix) (activation : String) (i j : Nat)
    (hj : j < b.columns) : Whole.outputOffset (request a b activation) i j = i*b.columns+j := by
  have hn : 1 ≤ b.columns := by omega
  simp [Whole.outputOffset, request, max_eq_left hn, Nat.mul_comm]

/-- Every active source store reaches the corresponding actual logical FP16
Tensor element; the physical realization is tied to the source pointer. -/
theorem source_store_address {α : Type} (a b : Matrix) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (s : State α)
    (i : FPFused.CIndex (request a b activation).contractedParameters)
    (hi : FPFused.active (request a b activation).contractedParameters s i) :
    let row := FPFused.row (request a b activation).contractedParameters s i.1
    let col := FPFused.col (request a b activation).contractedParameters s i.2.1
    outputAddress c (row*b.columns+col) = outputBase c+
      2*FPFused.cOffset (request a b activation).contractedParameters s i := by
  dsimp only
  rw [Whole.Contracted.source_offset, output_offset a b activation _ _ hi.2]
  apply output_address a b c cl
  rw [output_numel a b c cl]
  exact flat_bound _ _ _ _ hi.1 hi.2

def TensorResult {α β : Type} [Inhabited α] (a b : Matrix) (c : OutputTensor)
    (encode : α → Fin 2 → β) (target : Fin a.rows → Fin b.columns → α)
    (before : Nat → β) (final : State α) : Prop :=
  (∀ i : Fin a.rows, ∀ j : Fin b.columns, ∀ byte : Fin 2,
    writeback c encode before final (outputAddress c (i.val*b.columns+j.val)+byte.val) =
      encode (target i j) byte) ∧
  (∀ p, ¬OutputByte c p → writeback c encode before final p = before p) ∧
  (∀ p, a.tensor.storage.base ≤ p → p < a.tensor.storage.base+a.storageBytes →
    writeback c encode before final p = before p) ∧
  (∀ p, b.tensor.storage.base ≤ p → p < b.tensor.storage.base+b.storageBytes →
    writeback c encode before final p = before p)

theorem tensor_result {α β : Type} [Inhabited α] (a b : Matrix) (activation : String)
    (c : OutputTensor) (cl : OutputLaws c a b) (fresh : Fresh a b c)
    (encode : α → Fin 2 → β) (target : Fin a.rows → Fin b.columns → α)
    (before : Nat → β) (final : State α)
    (value : ∀ i : Fin a.rows, ∀ j : Fin b.columns,
      final.mem "c" (Whole.outputOffset (request a b activation) i.val j.val) =
        Cell.mk .fp16 (target i j)) : TensorResult a b c encode target before final := by
  refine ⟨?_, writeback_frame c encode before final,
    (backing_frame a b c fresh encode before final).1,
    (backing_frame a b c fresh encode before final).2⟩
  intro i j byte
  rw [writeback_at a b c cl encode before final _
    (by rw [output_numel a b c cl]; exact flat_bound _ _ _ _ i.isLt j.isLt) byte]
  have hv := value i j
  rw [output_offset a b activation _ _ j.isLt] at hv
  rw [hv, Cell.read_same]

#specNonCircular writeback avoiding [Whole.Contracted.reference, Whole.Contracted.logicalReference, Whole.mathematical]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly
