import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealProgram
import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealSource

/-! Exact whole-source bridge for the signed mirrored loop. Source controls
remain present, including the original post-store nonnegative-index branch. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def reverseStart (p : Layout) : Op .int [] :=
  .sub .int .nil (.constInt 0)
    (.sub .int .nil (.castNatToInt (stopOp p)) (.constInt 1))

def iterationBody (p : Layout) (s o : RegionName) : List Stmt :=
  .assign .int [] "i_t"
    (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")) :: loopBody p s o

def statement (p : Layout) (s o : RegionName) : Stmt :=
  .forRangeInt "__rev_i_t" (reverseStart p) (.constInt 1) (.constNat 1)
    (iterationBody p s o)

def body (p : Layout) (s o : RegionName) (swapped : Bool) : List Stmt :=
  initCode p swapped ++ [statement p s o]

theorem source_body (p : Layout) (s o : RegionName) (swapped : Bool)
    (dtype out_dtype : FloatDType) :
    (RealSource.source swapped dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType.toAlgKernel.body = body p s o swapped := by
  rw [RealSource.storage_erasure]
  cases swapped
  all_goals simp only [RealSource.source, Bool.false_eq_true, ↓reduceIte,
    TypedSource.original, TypedSource.compareRight, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#axiomsClean source_body
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealProgram
