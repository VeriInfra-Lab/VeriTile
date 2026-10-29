import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedProgram
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealValues
import VeriTile.Triton.Float.Erasure

/-! Complete real execution program obtained by the library's dtype erasure.
The source-body theorem checks every statement, signed loop bound and mask;
the following execution proofs use this exact checked program. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram
open VeriTile.Triton TypedSource
set_option maxHeartbeats 2400000
set_option maxRecDepth 24000

def numChunks (T BT : Nat) : Nat := (T + BT - 1) / BT

def pointerOp (swapped : Bool) (T BT : Nat) (r : RegionName) : Op .blockPtr [BT] :=
  .makeBlockPtrDynOffsets r
    (if swapped then .mul .nat .nil (.constNat T) (.ref .nat [] "i_bh")
      else .mul .nat .nil (.ref .nat [] "i_bh") (.constNat T))
    [T] [BT] [1] [TypedProgram.offsetOp BT]

def loopBody (swapped : Bool) (T BT : Nat) (s o : RegionName) : List Stmt :=
  [.assign .int [] "i_t" (.sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")),
   .assign .blockPtr [BT] "p_s" (pointerOp swapped T BT s),
   .assign .blockPtr [BT] "p_o" (pointerOp Bool.false T BT o),
   .assign .real [BT] "b_s"
     (.load .real (.blockPtr (.ref .blockPtr [BT] "p_s") [0]) .none .unspecified),
   .assign .real [] "b_zz" (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [BT] "b_s")),
   .assign .real [] "b_z" (.add .real .nil (.ref .real [] "b_z") (.ref .real [] "b_zz")),
   .assign .real [BT] "b_o" (RealValues.outputOp BT),
   .store .real [BT] (.blockPtr (.ref .blockPtr [BT] "p_o") [0])
     (.ref .real [BT] "b_o") .none]

def initCode : List Stmt :=
  [.assign .nat [] "i_bh" (.programId 0),
   .assign .real [] "b_z" (.full [] (.const 0))]

def statement (swapped : Bool) (T BT : Nat) (s o : RegionName) : Stmt :=
  .forRangeInt "__rev_i_t" (TypedProgram.startOp T BT) (.constInt 1) (.constNat 1)
    (loopBody swapped T BT s o)

def body (swapped : Bool) (T BT : Nat) (s o : RegionName) : List Stmt :=
  initCode ++ [statement swapped T BT s o]

theorem source_body (swapped : Bool) (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) :
    (source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel.body =
      body swapped T BT s o := by
  cases swapped
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, original, indexCommuted,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
    MaskOpt.eraseDType, NumericDType.eraseDType, Except.bind, bind, eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals cases dtype <;> cases out_dtype <;> rfl

theorem constInt_eval (n : Int) (u : BlockState) :
    evalOp (.constInt n) u = some (Tile.scalar n) := by
  simp only [evalOp]

theorem count_eval (T BT : Nat) (u : BlockState) :
    evalOp (TypedProgram.countOp T BT) u = some (Tile.scalar (numChunks T BT)) := by
  simp only [TypedProgram.countOp, evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
    Option.bind_some, Option.bind_eq_bind]
  rfl

theorem start_eval (T BT : Nat) (u : BlockState) :
    evalOp (TypedProgram.startOp T BT) u = some (Tile.scalar (1 - (numChunks T BT : Int))) := by
  have hc : evalOp (.castNatToInt (TypedProgram.countOp T BT)) u =
      some (Tile.scalar (numChunks T BT : Int)) := by
    simp only [evalOp, count_eval]
    rfl
  rw [TypedProgram.startOp, evalOp_sub, constInt_eval, evalOp_sub, hc, constInt_eval]
  apply congrArg some
  apply Tile.ext
  rintro ⟨⟩
  change (0 : Int) - ((numChunks T BT : Int) - 1) = 1 - (numChunks T BT : Int)
  omega

#axiomsClean source_body
#axiomsClean count_eval
#axiomsClean start_eval
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealProgram
