import bench.tritonbench_g.int_scaled_matmul.IntScaledMatmul
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.SignedCast
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner groupRows : Nat)
def original (p : Parameters) : ComputeKernel := int_scaled_matmul_matmul_surface
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

def widths : ComputeStmt → ComputeStmt
  | .assign .int sh name (.alg e) =>
    if name = "a" ∨ name = "b" then .assign .int sh name (.alg (.intToSigned 8 e))
    else if name = "accumulator" then .assign .int sh name (.alg (.intToSigned 32 e))
    else .assign .int sh name (.alg e)
  | stmt => stmt

def prelude (p : Parameters) := (original p).surfaceBody.take 12
def originalBody (p : Parameters) : List ComputeStmt := match (original p).surfaceBody[12]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def body (p : Parameters) := (originalBody p).map widths
def epilogue (p : Parameters) := (original p).surfaceBody.drop 13
/-- Raw source range(0,K,BK) and both checked axes are retained. No omitted
padding is replaced by zero; each outside lane consumes its typed byte oracle. -/
def source (p : Parameters) : ComputeKernel := .mk [p.input,p.weight] [p.output]
  (prelude p ++ [.forRange "k" 0 p.inner p.blockInner (body p)] ++ epilogue p)

def wrap (width : Nat) (x : Int) : Int := (BitVec.ofInt width x).toInt
def backend {α : Type} (rest : Algebra α) : Algebra α :=
  {rest with integerDot := fun _ batch {m} {k} {n} a b => some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt)}

theorem original_decomposition (p : Parameters) :
    (original p).surfaceBody = prelude p ++
      [.forRange "k" 0 p.inner p.blockInner (originalBody p)] ++ epilogue p := by rfl

theorem source_boundaries (p : Parameters) :
    (body p)[0]? = some (.assign .int [p.blockRows,p.blockInner] "a"
      (.alg (.intToSigned 8 (.load .int
        (.blockPtr (.ref .blockPtr [p.blockRows,p.blockInner] "a_block_ptr") [0,1]) .none .unspecified)))) ∧
    (body p)[1]? = some (.assign .int [p.blockInner,p.blockColumns] "b"
      (.alg (.intToSigned 8 (.load .int
        (.blockPtr (.ref .blockPtr [p.blockInner,p.blockColumns] "b_block_ptr") [0,1]) .none .unspecified)))) ∧
    (body p).length = 5 := by exact ⟨rfl,rfl,rfl⟩
#axiomsClean original_decomposition
#axiomsClean source_boundaries
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
