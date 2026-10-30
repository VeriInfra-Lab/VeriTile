import bench.tritonbench_g.matmul_dequantize_int4.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.Correct
open VeriTile Triton FP.Structural
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Full logical packed matrix product for every selected source tuning.
All typed backing cells and the original signed-int32 domain are retained. -/
specification correct (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    Spec.Real (ReaderIO.view c i ⊨{ReaderIO.packedInt32 c} fun xs row column =>
      ∑ k ∈ Finset.range c.K, xs.arrays.A row.val k *
        ((Real.nibble (xs.arrays.Q (k / 8) column.val) k -
          Real.nibble (xs.arrays.Z (k / c.group) (column.val / 8)) column.val) *
            xs.arrays.S (k / c.group) column.val)) :=
  ReaderProof.correct c valid i

#axiomsClean correct
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.Correct
