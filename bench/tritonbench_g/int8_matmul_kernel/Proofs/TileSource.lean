import bench.tritonbench_g.int8_matmul_kernel.Kernels
import bench.tritonbench_g.int8_matmul_kernel.Proofs.NativeArithmetic
import VeriTile.Triton.Float.IntegerDot
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

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

/-- The existing frontend records the one leading compile-time assertion as
an algorithm no-op, which the structural evaluator does not execute. This
finite projection recognizes only that exact leading metadata node. It keeps
all ports and every subsequent statement; it never ignores an arbitrary
conditional. The assertion's configuration predicate is a public premise. -/
def compileAssertionProjection : ComputeKernel → ComputeKernel
  | .mk inputs outputs (.alg (.ifThen (.constBool Bool.false) []) :: body) =>
      .mk inputs outputs (.ifThen (.constBool Bool.false) [] :: body)
  | quoted => quoted

def source (p : Parameters) : ComputeKernel := compileAssertionProjection (quotedSource p)

theorem quoted_assertion_node (p : Parameters) :
    (quotedSource p).surfaceBody.head? = some (.alg (.ifThen (.constBool Bool.false) [])) := by rfl

theorem projection_exact (p : Parameters) : (source p).surfaceBody =
    .ifThen (.constBool Bool.false) [] :: (quotedSource p).surfaceBody.tail := by rfl

theorem projection_tail (p : Parameters) :
    (source p).surfaceBody.tail = (quotedSource p).surfaceBody.tail := by rfl

theorem projection_length (p : Parameters) :
    (source p).surfaceBody.length = (quotedSource p).surfaceBody.length := by rfl

def prelude (p : Parameters) := (source p).surfaceBody.take 16
def outerBody (p : Parameters) : List ComputeStmt := match (source p).surfaceBody[16]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def outerPrelude (p : Parameters) := (outerBody p).take 1
def innerBody (p : Parameters) : List ComputeStmt := match (outerBody p)[1]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 17

/-- Exact expression retained by the source's dynamic range syntax. -/
def stepsExpr (p : Parameters) : Op .nat [] :=
  .div .nat .nil
    (.sub .nat .nil
      (.add .nat .nil (.floorDiv .nat .nil (.constNat p.inner) (.constNat 4))
        (.constNat p.blockInner)) (.constNat 1)) (.constNat p.blockInner)

theorem source_decomposition (p : Parameters) : (source p).surfaceBody =
    prelude p ++ [.forRangeDyn "i" (.constNat 0) (.constNat 4) (.constNat 1)
      (outerBody p)] ++ epilogue p := by rfl

theorem outer_decomposition (p : Parameters) : outerBody p =
    outerPrelude p ++ [.forRangeDyn "j" (.constNat 0) (stepsExpr p) (.constNat 1)
      (innerBody p)] := by rfl

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
