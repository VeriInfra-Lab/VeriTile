import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Kernels
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.NativeArithmetic
import VeriTile.Triton.Float.IntegerDot
import VeriTile.Triton.Float.StructuralLoopNormalization
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte truncateFloat DefinedCast)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

structure Parameters where
  (input weight output : RegionName)
  (rows columns inner inputRowStride inputInnerStride weightInnerStride weightColumnStride
    outputRowStride outputColumnStride blockRows blockColumns blockInner groupRows : Nat)

/-- The actual source ceil-div loop bound, not a freely chosen loop count. -/
def Parameters.steps (p : Parameters) : Nat := (p.inner/4+p.blockInner-1)/p.blockInner

def quotedSource (p : Parameters) : ComputeKernel := kernel
  (Region.cast p.input) (Region.cast p.weight) (Region.cast p.output)
  p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

/-- Historical compatibility for the former leading algorithm-only marker.
Current raw source emission uses an executable ComputeStmt no-op directly and
`source` below does not apply this projection. This helper still recognizes
only that exact old leading node, preserving its complete tail and ports. -/
def compileAssertionProjection : ComputeKernel → ComputeKernel
  | .mk inputs outputs (.alg (.ifThen (.constBool Bool.false) []) :: body) =>
      .mk inputs outputs (.ifThen (.constBool Bool.false) [] :: body)
  | quoted => quoted

/-- The current raw quotation executes directly. The historical projection
above remains available only for the old algorithm-only marker. -/
def source (p : Parameters) : ComputeKernel := quotedSource p

theorem source_eq_quoted (p : Parameters) : source p = quotedSource p := rfl

theorem quoted_assertion_node (p : Parameters) :
    (quotedSource p).surfaceBody.head? = some (.ifThen (.constBool Bool.false) []) := by rfl

theorem projection_exact (p : Parameters) : (source p).surfaceBody =
    .ifThen (.constBool Bool.false) [] :: (quotedSource p).surfaceBody.tail := by rfl

theorem projection_tail (p : Parameters) :
    (source p).surfaceBody.tail = (quotedSource p).surfaceBody.tail := by rfl

theorem projection_length (p : Parameters) :
    (source p).surfaceBody.length = (quotedSource p).surfaceBody.length := by rfl

def prelude (p : Parameters) := (source p).surfaceBody.take 16
def outerBody (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[16]? with
  | some (ComputeStmt.forLoop _ _ body) => body
  | _ => []
def outerPrelude (p : Parameters) := (outerBody p).take 1
def innerBody (p : Parameters) : List ComputeStmt := match (outerBody p)[1]? with
  | some (ComputeStmt.forRange _ _ _ _ body) => body
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 17

/-- The original constexpr bound is evaluated once at specialization. -/
def stepsExpr (p : Parameters) : Op .nat [] := .constNat p.steps

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [.forLoop "i" 4 (outerBody p)] ++ epilogue p := by rfl

theorem outer_decomposition (p : Parameters) : outerBody p =
    outerPrelude p ++ [.forRange "j" 0 p.steps 1 (innerBody p)] := by rfl

theorem inner_length (p : Parameters) : (innerBody p).length = 10 := by rfl

/-- Conversion fails outside the wrapper's exact finite/truncation domain.
The supplied native conversion is only used after that domain is established. -/
def convert8 (cast : Word32 → Byte) (word : Word32) : Option Int :=
  (truncateFloat word).bind fun value =>
    if -128 ≤ value ∧ value ≤ 127 then some (cast word).toInt else none

theorem convert8_defined (cast : Word32 → Byte) (word : Word32) (h : DefinedCast word) :
    convert8 cast word = some (cast word).toInt := by
  obtain ⟨value, hv, hlo, hhi⟩ := h
  simp [convert8, hv, hlo, hhi]

/-- Concrete source-profile backend. FP32 storage loads are bit copies and the
FP32 zero literal is the zero word. The only numeric conversion implemented here
is the source's defined FP32-to-signed8 cast. Integer dot validates signed8
operands and the retained INT32/default options before a finite word fold.
Its correspondence to bounded native K32 instructions is proved separately. -/
def backend (cast : Word32 → Byte) (rest : Algebra Word32) : Algebra Word32 := by
  classical
  exact { FP.IntegerDot.backend rest with
    literal := fun precision dtype value =>
      if precision = some .fp32 ∧ dtype = .real ∧ value = 0 then 0
      else rest.literal precision dtype value
    fp32Load := id
    floatToInt := fun precision dtype width =>
      if precision = some .fp32 ∧ dtype = .real ∧ width = 8 then some (convert8 cast)
      else rest.floatToInt precision dtype width }

def wordDot {m k n : Nat} (a : TileIndex [m,k] → Int) (b : TileIndex [k,n] → Int) :
    TileIndex [m,n] → Int := fun i => (FP.IntegerDot.wordDot [] a b i).toInt

def wrap (width : Nat) (value : Int) : Int := (BitVec.ofInt width value).toInt

#axiomsClean source_eq_quoted
#axiomsClean source_decomposition
#axiomsClean quoted_assertion_node
#axiomsClean projection_exact
#axiomsClean projection_tail
#axiomsClean projection_length
#axiomsClean outer_decomposition
#axiomsClean inner_length
#axiomsClean convert8_defined
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
