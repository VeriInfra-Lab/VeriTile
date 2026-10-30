import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Config
import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

def indexPrefix (X : RegionName) (B : Nat) : List Stmt :=
  [.assign .int [B] "n_offsets"
      (.add .int .scalarL (.ref .int [] "start_n") (.castNatToInt (.arange B))),
   .assign .int [B] "offset"
      (.add .int .scalarL (.mul .int .nil (.ref .int [] "pid_m") (.ref .int [] "N"))
        (.ref .int [B] "n_offsets")),
   .assign .ptr [B] "input_ptrs"
      (.ptrAdd .scalarL (.ptrBase X) (.castIntToNat (.ref .int [B] "offset")))]

def maskStatement (B : Nat) : Stmt :=
  .assign .bool [B] "mask" (.lt .int .scalarR (.ref .int [B] "n_offsets") (.ref .int [] "N"))

def previousExpression (B : Nat) : Op .int [] :=
  .sub .int .nil
    (.mul .int .nil
      (FullSource.signedQuotient
        (.add .int .nil (.ref .int [] "N")
          (.castNatToInt (.sub .nat .nil (.constNat B) (.constNat 1))))
        (.castNatToInt (.constNat B))) (.castNatToInt (.constNat B)))
    (.castNatToInt (.constNat B))

def update (B : Nat) : List Stmt :=
  [SignedTransport.sourceMaximum B,
   .assign .real [B] "old_mass"
     ((Op.eq .real .scalarR (.ref .real [B] "new_m") .negInf).where
       (.ref .real [B] "z")
       (.mul .real (.consSame .nil)
         (.exp (.sub .real (.consSame .nil) (.ref .real [B] "m") (.ref .real [B] "new_m")))
         (.ref .real [B] "z"))),
   .assign .real [B] "new_mass"
     ((Op.eq .real .scalarR (.ref .real [B] "inp") .negInf).where
       ((Op.const 0.0).broadcast [B])
       (.exp (.sub .real (.consSame .nil) (.ref .real [B] "inp") (.ref .real [B] "new_m")))),
   .assign .real [B] "new_z"
     (.add .real (.consSame .nil) (.ref .real [B] "old_mass") (.ref .real [B] "new_mass")),
   .assign .real [B] "m" (.ref .real [B] "new_m"),
   .assign .real [B] "z" (.ref .real [B] "new_z")]

def inputFull (X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefix X B ++ ((loopAbody X N B).drop 3).take 1 ++ update B

def inputTail (X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefix X B ++ [maskStatement B] ++ ((loopBbody X N B).drop 4).take 1 ++ update B

def outputFull (Y X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefix X B ++ ((loopApass2body Y X N B).drop 3).take 3 ++
    [SignedTransport.outputPointer Y B] ++ (loopApass2body Y X N B).drop 7

def outputTail (Y X : RegionName) (N B : Nat) : List Stmt :=
  indexPrefix X B ++ [maskStatement B] ++ ((loopBpass2body Y X N B).drop 4).take 3 ++
    [SignedTransport.outputPointer Y B] ++ (loopBpass2body Y X N B).drop 8

theorem projects (Y X : RegionName) (M N B : Nat) :
    (kernel Y X M N B).toAlgorithm? = .ok (kernel Y X M N B).toAlgKernel := by rfl

/-- Exact decomposition of the new typed source after mathematical dtype
projection. It retains widened signed registers and both actual mass guards. -/
theorem source_body (Y X : RegionName) (M N B : Nat) :
    (kernel Y X M N B).eraseDType.toAlgKernel.body =
      [.assign .int [] "N" (.constInt N),
       .assign .int [] "pid_m" (.castNatToInt (.programId 0)),
       .assign .real [B] "m" (.full [B] .negInf),
       .assign .real [B] "z" (.full [B] (.const 0)),
       .assign .int [] "prev_multiple" (previousExpression B),
       .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
         (inputFull X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "N") (.constNat B)
         (inputTail X N B)] ++ reductionStmts B ++
      [.assign .int [] "prev_multiple" (previousExpression B),
       .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
         (outputFull Y X N B),
       .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "N") (.constNat B)
         (outputTail Y X N B)] := by
  simp only [ComputeKernel.eraseDType, projects, ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [kernel, body, ComputeKernel.withScalarArguments, ScalarArgument.integer, ScalarArgument.statement,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, FloatDType.toTileDType,
    ComputeDType.eraseDType]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType, IntegralDType.eraseDType,
    eq_mpr_eq_cast, eq_mp_eq_cast, cast_eq]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Source
