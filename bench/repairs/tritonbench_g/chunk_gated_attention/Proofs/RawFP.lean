import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.RawNative
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ScalarObservation

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawFP
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section

/-- Complete selected FP32 source execution on raw graph memory, every native
word observation against the independent ordered recurrence, and exterior frame.
The config supplies geometry and source branches; the source fixes FP32 storage
and the literal region roles k/v/g/h/h0/ht, as in NativeGrid.innerSource. -/
theorem inner_fp_equiv (c : HostFP.Inner) (before : NativeGrid.State)
    (actual reference : ScalarNative.Primitives) (laws : ScalarLaws actual reference) :
    ∃ after,
      HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.innerSource c))
        (NativeGrid.innerOwns c) c.plans before = some after ∧
      after.mem = (NativeGrid.innerResult c before).mem ∧
      (∀ region offset (dtype : FloatDType),
        ScalarNative.eval actual (toFloat dtype ((after.mem region offset).read dtype.toTileDType)) =
          ScalarNative.eval reference (toFloat dtype (((NativeGrid.innerResult c before).mem region offset).read dtype.toTileDType))) ∧
      (∀ region offset, ¬ RawNative.Writes (NativeGrid.innerOwns c) c.plans region offset →
        after.mem region offset = before.mem region offset) := by
  refine ⟨NativeGrid.innerResult c before, NativeGrid.inner_complete c before, rfl, ?_, ?_⟩
  · intro region offset dtype
    exact scalar_eval_eq actual reference laws _
  · intro region offset outside
    exact RawNative.frame _ _ _ before region offset outside

/-- Complete selected FP32 prefix source, independently defined triangular-dot
reference words and every exterior cell, using the original scalar laws. -/
theorem prefix_fp_equiv (c : HostFP.Cumulative) (before : NativeGrid.State)
    (actual reference : ScalarNative.Primitives) (laws : ScalarLaws actual reference) :
    ∃ after,
      HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.prefixSource c))
        (NativeGrid.prefixOwns c) c.plans before = some after ∧
      after.mem = (NativeGrid.prefixResult c before).mem ∧
      (∀ region offset (dtype : FloatDType),
        ScalarNative.eval actual (toFloat dtype ((after.mem region offset).read dtype.toTileDType)) =
          ScalarNative.eval reference (toFloat dtype (((NativeGrid.prefixResult c before).mem region offset).read dtype.toTileDType))) ∧
      (∀ region offset, ¬ RawNative.Writes (NativeGrid.prefixOwns c) c.plans region offset →
        after.mem region offset = before.mem region offset) := by
  refine ⟨NativeGrid.prefixResult c before, NativeGrid.prefix_complete c before, rfl, ?_, ?_⟩
  · intro region offset dtype
    exact scalar_eval_eq actual reference laws _
  · intro region offset outside
    exact RawNative.frame _ _ _ before region offset outside

#specNonCircular NativeGrid.innerResult avoiding [FP.Structural.exec,HostFP.batches]
#specNonCircular NativeGrid.prefixResult avoiding [FP.Structural.exec,HostFP.batches]
#axiomsClean scalar_eval_eq
#axiomsClean inner_fp_equiv
#axiomsClean prefix_fp_equiv
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.RawFP
