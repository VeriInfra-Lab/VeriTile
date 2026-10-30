import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.EventResult

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
open VeriTile.Triton
open MatmulLeakyrelu.KernelOnly MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 3000000

def OutputByte (c : OutputTensor) (p : Nat) : Prop :=
  outputBase c ≤ p ∧ p < outputBase c+2*c.metadata.numel

/-- Realization of the executed output cells at actual FP16 Tensor addresses.
The complement retains the same caller heap. No reference appears here. -/
def writeback {α β : Type} (c : OutputTensor) (encode : α → Fin 2 → β)
    (before : Nat → β) (final : Memory α) (p : Nat) : β :=
  if OutputByte c p then
    encode (final (2,c.storageOffset+(p-outputBase c)/2))
      ⟨(p-outputBase c)%2, Nat.mod_lt _ (by decide)⟩
  else before p

theorem writeback_at {α β : Type} (r : Request) (c : OutputTensor) (h : Contract r c)
    (encode : α → Fin 2 → β) (before : Nat → β) (final : Memory α)
    (i : Nat) (hi : i < c.metadata.numel) (byte : Fin 2) :
    writeback c encode before final (outputAddress c i+byte.val) =
      encode (final (2,c.storageOffset+i)) byte := by
  rw [output_address r.A r.B c h.outputAPI i hi]
  have hs : outputBase c+2*i+byte.val-outputBase c = 2*i+byte.val := by omega
  have hd : (2*i+byte.val)/2 = i := by omega
  have hm : (2*i+byte.val)%2 = byte.val := by omega
  have inside : OutputByte c (outputBase c+2*i+byte.val) := by
    unfold OutputByte
    have hb := byte.isLt
    omega
  simp [writeback, inside, hs, hd, hm]

theorem writeback_frame {α β : Type} (c : OutputTensor) (encode : α → Fin 2 → β)
    (before : Nat → β) (final : Memory α) (p : Nat) (outside : ¬OutputByte c p) :
    writeback c encode before final p = before p := by simp [writeback,outside]

theorem outside_of_disjoint (c : OutputTensor) (base size p : Nat)
    (sep : VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.byteSpansDisjoint
      (outputBase c) (2*c.metadata.numel) base size)
    (inside : base ≤ p ∧ p < base+size) : ¬OutputByte c p := by
  unfold VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole.byteSpansDisjoint at sep
  unfold OutputByte
  rcases sep with hh | hh | hh | hh <;> omega

def TensorResult {α β : Type} (r : Request) (c : OutputTensor)
    (encode : α → Fin 2 → β) (target : Fin r.A.rows → Fin r.B.columns → α)
    (before : Nat → β) (final : Memory α) : Prop :=
  (∀ i : Fin r.A.rows, ∀ j : Fin r.B.columns, ∀ byte : Fin 2,
    writeback c encode before final (outputAddress c (i.val*r.B.columns+j.val)+byte.val) =
      encode (target i j) byte) ∧
  (∀ p, ¬OutputByte c p → writeback c encode before final p = before p) ∧
  (∀ p, r.A.tensor.storage.base ≤ p → p < r.A.tensor.storage.base+r.A.storageBytes →
    writeback c encode before final p = before p) ∧
  (∀ p, r.B.tensor.storage.base ≤ p → p < r.B.tensor.storage.base+r.B.storageBytes →
    writeback c encode before final p = before p)

theorem tensor_result {α β : Type} (r : Request) (c : OutputTensor) (h : Contract r c)
    (encode : α → Fin 2 → β) (target : Fin r.A.rows → Fin r.B.columns → α)
    (before : Nat → β) (final : Memory α)
    (value : ∀ i : Fin r.A.rows, ∀ j : Fin r.B.columns,
      final ((outputDescriptor r c).address i.val j.val) = target i j) :
    TensorResult r c encode target before final := by
  refine ⟨?_,writeback_frame c encode before final,?_,?_⟩
  · intro i j byte
    rw [writeback_at r c h encode before final _
      (by rw [output_numel r.A r.B c h.outputAPI]; exact flat_bound _ _ _ _ i.isLt j.isLt) byte]
    have hv := value i j
    rw [(output_coordinate r c h i.val j.val i.isLt j.isLt).1] at hv
    simpa [Nat.add_assoc] using congrArg (fun x => encode x byte) hv
  · intro p hp hq
    exact writeback_frame c encode before final p (outside_of_disjoint c _ _ p h.fresh.a ⟨hp,hq⟩)
  · intro p hp hq
    exact writeback_frame c encode before final p (outside_of_disjoint c _ _ p h.fresh.b ⟨hp,hq⟩)

#specNonCircular writeback avoiding [eventRun,logicalReference,reference,resultMemory]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast
