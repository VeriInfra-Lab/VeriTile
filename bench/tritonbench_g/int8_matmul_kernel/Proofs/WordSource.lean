import bench.tritonbench_g.int8_matmul_kernel.Int8MatmulKernel
import VeriTile.Triton.Float.IntegerDot
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner groupRows steps : Nat)
def original (p : Parameters) : ComputeKernel := int8_matmul_kernel_surface
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows p.steps

def originalPrelude (p : Parameters) := (original p).surfaceBody.take 16
/-- The compile-time static assertion remains a no-op statement; its actual
configuration predicate is retained on the public source contract. -/
def prelude (p : Parameters) : List ComputeStmt :=
  [.ifThen (.constBool Bool.false) []] ++ (originalPrelude p).drop 1
def originalOuter (p : Parameters) : List ComputeStmt := match (original p).surfaceBody[16]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def originalInner (p : Parameters) : List ComputeStmt := match (originalOuter p)[1]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
/-- Restore exactly the three signed-width boundaries erased by the legacy
port: its two explicit int8 casts and the int32 accumulator update. Indexing,
loads, masks, dot options and loop order remain those of the source. -/
def widths : ComputeStmt → ComputeStmt
  | .assign .int sh name (.alg e) =>
      if name = "a" ∨ name = "b" then .assign .int sh name (.alg (.intToSigned 8 e))
      else if name = "accumulator" then .assign .int sh name (.alg (.intToSigned 32 e))
      else .assign .int sh name (.alg e)
  | stmt => stmt

def innerBody (p : Parameters) := (originalInner p).map widths
def outerPrelude (p : Parameters) := (originalOuter p).take 1
def outerBody (p : Parameters) := outerPrelude p ++ [.forRange "j" 0 p.steps 1 (innerBody p)]
def epilogue (p : Parameters) := (original p).surfaceBody.drop 17
def source (p : Parameters) : ComputeKernel := .mk [p.input,p.weight] [p.output]
  (prelude p ++ [.forRange "i" 0 4 1 (outerBody p)] ++ epilogue p)

/-- Both the source casts and accumulator narrowing decode two's-complement
words. This is a bit operation, not a finite-input/no-overflow assumption. -/
def wrap (w : Nat) (x : Int) : Int := (BitVec.ofInt w x).toInt

def backend {α : Type} (rest : Algebra α) : Algebra α := by
  refine { rest with integerDot := ?_ }
  intro options batch m k n a b
  exact some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt)

def wordDot {m k n : Nat} (a : TileIndex [m,k] → Int) (b : TileIndex [k,n] → Int) :
    TileIndex [m,n] → Int := fun i => (FP.IntegerDot.wordDot [] a b i).toInt

/-- No source statements disappear during the per-assignment annotation. -/
theorem annotated_inner_length (p : Parameters) : (innerBody p).length = 9 := by rfl

theorem original_decomposition (p : Parameters) : (original p).surfaceBody =
    originalPrelude p ++ [.forRange "i" 0 4 1
      (outerPrelude p ++ [.forRange "j" 0 p.steps 1 (originalInner p)])] ++ epilogue p := by rfl
#axiomsClean annotated_inner_length
#axiomsClean original_decomposition
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
