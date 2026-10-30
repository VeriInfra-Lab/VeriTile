import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelWhole
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelNativeProfile

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelNativeResult
open VeriTile.Triton FP.Structural KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- All FP32 words are admitted. Only the storage interpretations of the int
and fp16 tagged cells are used to read the two observable output Tensors. -/
def Result (a : Arithmetic) (p : Launch) (values : Nat → Nat → Word)
    (initial final : State Word) : Prop :=
  (∀ r c, r < p.rows → c < p.columns →
    narrow8 ((final.mem "q" (r*p.columns+c)).read .int) = quantized a p (values r) c) ∧
  (∀ r, r < p.rows → KernelEncoding.unpackHalf ((final.mem "m" r).read .fp16) =
    storedMaximum a p (values r)) ∧
  (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
    (region ≠ "m" ∨ p.rows ≤ offset) → final.mem region offset = initial.mem region offset)

theorem kernel_correct (a : Arithmetic) (rest : Algebra Word) (p : Launch) (schedule : Schedule p)
    (values : Nat → Nat → Word) (initial : State Word) (reads : KernelWhole.Reads p values initial) :
    ∃ final, KernelGrid.run (algebra a p rest) (llrint a) (equal a)
      p schedule.order initial = some final ∧ Result a p values initial final := by
  obtain ⟨final, run, hq, hm, hf⟩ := KernelWhole.source_correct (operations a)
    (fun x => KernelEncoding.packHalf (a.half x)) (llrint a) rest p schedule values initial reads
  refine ⟨final, run, ?_, ?_, hf⟩
  · intro r c hr hc
    rw [hq r c hr hc]
    change narrow8 (KernelExecution.signedByte _) = _
    rw [KernelExecution.narrow_signedByte]
    rfl
  · intro r hr
    rw [hm r hr]
    exact KernelEncoding.half_roundtrip _

#axiomsClean kernel_correct
#specNonCircular Result avoiding [RowwiseQuantizationTriton.source, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.KernelNativeResult
