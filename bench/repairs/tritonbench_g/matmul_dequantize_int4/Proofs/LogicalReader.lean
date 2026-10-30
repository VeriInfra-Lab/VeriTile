import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IOReal
import bench.repairs.tritonbench_g.matmul_dequantize_int4.IO
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.ReaderProof
open VeriTile Triton FP.Structural ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

theorem target_eq (c : RawKernel.Layout) (i : Fin 17) :
    (view c i).lift (fun xs row column =>
      ∑ k ∈ Finset.range c.K, xs.arrays.A row.val k *
        ((Real.nibble (xs.arrays.Q (k / 8) column.val) k -
          Real.nibble (xs.arrays.Z (k / c.group) (column.val / 8)) column.val) *
            xs.arrays.S (k / c.group) column.val)) = (fun vals => IOData.expected c (fun r n => ∑ k ∈ Finset.range c.K, (IOMath.input c vals).A r k * ((Real.nibble ((IOMath.input c vals).Q (k / 8) n) k - Real.nibble ((IOMath.input c vals).Z (k / c.group) (n / 8)) n) * (IOMath.input c vals).S (k / c.group) n))) := by
  funext xs o j
  fin_cases o
  rfl

/-- Exact complete old/new judgment equivalence, including the signed-int32
packed domain, all source casts/stores, execution safety and exterior frame. -/
theorem judgment_iff (c : RawKernel.Layout) (i : Fin 17) :
    (view c i ⊨{packedInt32 c} fun xs row column =>
      ∑ k ∈ Finset.range c.K, xs.arrays.A row.val k *
        ((Real.nibble (xs.arrays.Q (k / 8) column.val) k -
          Real.nibble (xs.arrays.Z (k / c.group) (column.val / 8)) column.val) *
            xs.arrays.S (k / c.group) column.val)) ↔
      (IOData.program ℝ c (IOReal.jobs c i)).ImplementsOn (IOData.packedInt32 c) (fun vals => IOData.expected c (fun r n => ∑ k ∈ Finset.range c.K, (IOMath.input c vals).A r k * ((Real.nibble ((IOMath.input c vals).Q (k / 8) n) k - Real.nibble ((IOMath.input c vals).Z (k / c.group) (n / 8)) n) * (IOMath.input c vals).S (k / c.group) n))) := by
  rw [LogicalProgramIO.implementsOn_iff,target_eq]
  rfl

theorem correct (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    (view c i ⊨{packedInt32 c} fun xs row column =>
      ∑ k ∈ Finset.range c.K, xs.arrays.A row.val k *
        ((Real.nibble (xs.arrays.Q (k / 8) column.val) k -
          Real.nibble (xs.arrays.Z (k / c.group) (column.val / 8)) column.val) *
            xs.arrays.S (k / c.group) column.val)) :=
  (judgment_iff c i).mpr (IOReal.implements c valid i)
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.ReaderProof
