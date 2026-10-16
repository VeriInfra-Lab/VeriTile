import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPTypedSource
import VeriTile.Triton.Float.SignedCast
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
structure Parameters where
  (input weight output bias rowScale columnScale : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner groupRows steps : Nat)
  (hasBias : Bool)
def original (p : Parameters) (divisor : ℝ) : ComputeKernel := typed
  p.input p.weight p.output p.bias p.rowScale p.columnScale p.rows p.columns p.inner
  divisor p.hasBias p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows p.steps
/-- Explicit source storage boundaries: signed byte loads and the signed
32-bit accumulator assignment. No integer no-overflow premise is used. -/
def widths : ComputeStmt → ComputeStmt
  | .assign .int sh name (.alg e) =>
    if name = "a" ∨ name = "b" then .assign .int sh name (.alg (.intToSigned 8 e))
    else if name = "acc_int" then .assign .int sh name (.alg (.intToSigned 32 e))
    else .assign .int sh name (.alg e)
  | stmt => stmt
/-- The old port used Real loads for these two physical fp32 scale buffers.
The typed annotation retains their actual storage and the fp32 register cast. -/
def scales (p : Parameters) : List ComputeStmt := [
  .assign .real [1,p.blockColumns] "w_factor" (.compute (.alg .fp32
    (.castFloat .fp32 .real (.expandDim ⟨0, by simp⟩
      (.load .fp32 (.region (Region.cast p.columnScale) (.ref .nat [p.blockColumns] "rbn")) .none))))),
  .assign .real [p.blockRows,1] "x_factor" (.compute (.alg .fp32
    (.castFloat .fp32 .real (.expandDim ⟨1, by simp⟩
      (.load .fp32 (.region (Region.cast p.rowScale) (.ref .nat [p.blockRows] "ram")) .none)))))]
/-- Python passes divfactor as a runtime fp32 scalar. Its initial register
contains the actual supplied bits; it is not replaced by a Real arithmetic oracle. -/
def scaled (p : Parameters) : ComputeStmt :=
  .assign .real [p.blockRows,p.blockColumns] "scaled" (.compute (.alg .fp32
    (.mul .real (.consL (.consSame .nil)) (.ref .real [1,p.blockColumns] "w_factor")
      (.mul .real (.consSame (.consL .nil)) (.ref .real [p.blockRows,1] "x_factor")
        (.mul .real .scalarR (.intToReal (.ref .int [p.blockRows,p.blockColumns] "acc_int"))
          (.castFloat .fp32 .real (.ref .fp32 [] "divfactor")))))))
def prelude (p : Parameters) : List ComputeStmt :=
  (original p 0).surfaceBody.take 18 ++ scales p ++
    ((original p 0).surfaceBody.drop 20).take 1

def originalBody (p : Parameters) : List ComputeStmt := match (original p 0).surfaceBody[21]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def body (p : Parameters) : List ComputeStmt := (originalBody p).map widths
def epilogue (p : Parameters) : List ComputeStmt := scaled p :: (original p 0).surfaceBody.drop 23

def source (p : Parameters) : ComputeKernel := .mk
  [p.input,p.weight,p.bias,p.rowScale,p.columnScale] [p.output]
  (prelude p ++ [.forRange "k" 0 p.steps 1 (body p)] ++ epilogue p)

def wrap (width : Nat) (x : Int) : Int := (BitVec.ofInt width x).toInt
/-- Source inputs are narrowed to int8 before this callback. The result is
exact integer dot modulo 2^32; no floating dot callback or reduction law. -/
def backend {α : Type} (rest : Algebra α) : Algebra α :=
  {rest with integerDot := fun _ batch {m} {k} {n} a b => some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt)}

theorem original_decomposition (p : Parameters) (divisor : ℝ) :
  (original p divisor).surfaceBody =
    ((original p divisor).surfaceBody.take 21) ++
    [.forRange "k" 0 p.steps 1 (originalBody p)] ++
    ((original p divisor).surfaceBody.drop 22) := by rfl

theorem preserved_address_prefix (p : Parameters) (divisor : ℝ) :
  (prelude p).take 18 = (original p divisor).surfaceBody.take 18 := by rfl

theorem preserved_epilogue_tail (p : Parameters) (divisor : ℝ) :
  (epilogue p).drop 1 = (original p divisor).surfaceBody.drop 23 := by rfl

theorem width_annotation_length (p : Parameters) : (body p).length = 5 := by rfl
#axiomsClean original_decomposition
#axiomsClean preserved_address_prefix
#axiomsClean preserved_epilogue_tail
#axiomsClean width_annotation_length
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
