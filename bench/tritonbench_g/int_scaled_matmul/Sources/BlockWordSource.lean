import bench.tritonbench_g.int_scaled_matmul.Sources.Kernels
import bench.tritonbench_g.int_scaled_matmul.Sources.Mathematical_IntScaledMatmul
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.SignedCast
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
open VeriTile.Bench.TritonBenchG.IntScaledMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
def wrap (width : Nat) (x : Int) : Int := (BitVec.ofInt width x).toInt
def backend {α : Type} (rest : Algebra α) : Algebra α :=
  {rest with integerDot := fun _ batch {m} {k} {n} a b => some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt)}

theorem original_decomposition (p : Parameters) :
    (original p).surfaceBody = prelude p ++
      [.forRange "k" 0 p.inner p.blockInner (originalBody p)] ++ epilogue p := by rfl

theorem source_boundaries (p : Parameters) :
    (body p)[0]? = some (.assign .int [p.blockRows,p.blockInner] "a"
      (.alg (.intToSigned 8 (.load .int
        (.blockPtr (.ref .blockPtr [p.blockRows,p.blockInner] "a_block_ptr") [0,1]) .none .zero)))) ∧
    (body p)[1]? = some (.assign .int [p.blockInner,p.blockColumns] "b"
      (.alg (.intToSigned 8 (.load .int
        (.blockPtr (.ref .blockPtr [p.blockInner,p.blockColumns] "b_block_ptr") [0,1]) .none .zero)))) ∧
    (body p).length = 5 := by exact ⟨rfl,rfl,rfl⟩
#axiomsClean original_decomposition
#axiomsClean source_boundaries
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.BlockWord
