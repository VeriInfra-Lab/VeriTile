import bench.tritonbench_g.matrix_transpose.Proofs.FPBits
import bench.tritonbench_g.matrix_transpose.Proofs.Source
import bench.tritonbench_g.matrix_transpose.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
open VeriTile.Triton FP.Structural

theorem exact_existing_source (v : Bool) (p : Parameters) :
    source v p = typedKernel v p.input p.output p.sx p.sy p.ox p.oy p.m p.d := rfl

theorem old_output_address (p : Parameters) (i : TileIndex [p.d,p.m]) :
    outputOffset p i = outAddr p.ox p.oy i := rfl

theorem old_input_address (p : Parameters) (i : TileIndex [p.d,p.m]) :
    inputOffset p i = matrixAddr p.sx p.sy i := rfl

/-- The complete source has six statements and the real fp16 load followed
by the fp16 transpose store, without inserting arithmetic or casts. -/
theorem full_source_storage (v : Bool) (p : Parameters) :
    (source v p).surfaceBody.length = 6 ∧
    (source v p).surfaceBody[4]? =
      some (.assign .fp16 [p.m,p.d] "matrix" (.alg
        (.load .fp16 (.ptr (.ref .ptr [p.m,p.d] "matrix_ptr")) .none))) ∧
    (source v p).surfaceBody[5]? =
      some (.store .fp16 [p.d,p.m] (.ptr (.ref .ptr [p.d,p.m] "out_ptr"))
        (.alg (Op.transpose (batch := []) (.ref .fp16 [p.m,p.d] "matrix"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

#axiomsClean exact_existing_source
#axiomsClean old_output_address
#axiomsClean old_input_address
#axiomsClean full_source_storage
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FPBits
