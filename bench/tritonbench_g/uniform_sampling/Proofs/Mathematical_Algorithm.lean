import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_Erasure
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_RandomDomain

/-! A normal form of the complete source projection. The source equality
below checks all twenty-three statements, including the simultaneous tuple's
five operand captures and four result writes. This is not an input/output
specification; the independent scalar/output formula lives in ScalarSpec and
Formula and is connected by the execution proof.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Algorithm
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def indexOp (swapped : Bool) (B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (if swapped then
    .mul .nat .nil (.ref .nat [] "BLOCK_arg") (.programId 0) else
    .mul .nat .nil (.programId 0) (.ref .nat [] "BLOCK_arg")) (.arange B)

def tupleBlock (B : Nat) : Stmt :=
  .ifThen (.constBool Bool.true) [
    .assign .int [] "__veritile_direct_random_0" (.ref .int [] "philox_seed"),
    .assign .int [B] "__veritile_direct_random_1" (.castNatToInt (.ref .nat [B] "c0")),
    .assign .int [] "__veritile_direct_random_2" (.castNatToInt (.ref .nat [] "c1")),
    .assign .int [B] "__veritile_direct_random_3" (.castNatToInt (.ref .nat [B] "_O")),
    .assign .int [B] "__veritile_direct_random_4" (.castNatToInt (.ref .nat [B] "_O")),
    .assign .nat [B] "r0" (.philox4x32 RandomDomain.profile (Random.Direct.uniformPlan [B] 9) .c0
      (.ref .int [] "__veritile_direct_random_0") (.ref .int [B] "__veritile_direct_random_1")
      (.ref .int [] "__veritile_direct_random_2") (.ref .int [B] "__veritile_direct_random_3")
      (.ref .int [B] "__veritile_direct_random_4")),
    .assign .nat [B] "r1" (.philox4x32 RandomDomain.profile (Random.Direct.uniformPlan [B] 9) .c1
      (.ref .int [] "__veritile_direct_random_0") (.ref .int [B] "__veritile_direct_random_1")
      (.ref .int [] "__veritile_direct_random_2") (.ref .int [B] "__veritile_direct_random_3")
      (.ref .int [B] "__veritile_direct_random_4")),
    .assign .nat [B] "r2" (.philox4x32 RandomDomain.profile (Random.Direct.uniformPlan [B] 9) .c2
      (.ref .int [] "__veritile_direct_random_0") (.ref .int [B] "__veritile_direct_random_1")
      (.ref .int [] "__veritile_direct_random_2") (.ref .int [B] "__veritile_direct_random_3")
      (.ref .int [B] "__veritile_direct_random_4")),
    .assign .nat [B] "r3" (.philox4x32 RandomDomain.profile (Random.Direct.uniformPlan [B] 9) .c3
      (.ref .int [] "__veritile_direct_random_0") (.ref .int [B] "__veritile_direct_random_1")
      (.ref .int [] "__veritile_direct_random_2") (.ref .int [B] "__veritile_direct_random_3")
      (.ref .int [B] "__veritile_direct_random_4"))]

def affineStmt (B : Nat) (name : RegName) (lower upper : ℝ) : Stmt :=
  .assign .real [B] name (.add .real .scalarR
    (.mul .real .scalarR (.uniform32 (.ref .nat [B] name))
      (.sub .real .nil (.const upper) (.const lower))) (.const lower))

def storeStmt (y : RegionName) (N B : Nat) (wordName offsetName : RegName) : Stmt :=
  .store .real [B] (.region y (.ref .nat [B] offsetName)) (.ref .real [B] wordName)
    (.mask (.lt .nat .scalarR (.ref .nat [B] offsetName) (.constNat N)))

def kernel (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) : Kernel :=
  { inputs := [], outputs := [y], body := [
    .assign .int [] "philox_seed_arg" (.constInt seed),
    .assign .int [] "philox_offset_arg" (.constInt offset),
    .assign .nat [] "BLOCK_arg" (.constNat B),
    .assign .int [] "philox_seed" (.ref .int [] "philox_seed_arg"),
    .assign .int [] "philox_offset" (.ref .int [] "philox_offset_arg"),
    .assign .nat [] "c0" (.intToUnsigned 32
      (.intBitAnd 64 .nil (.ref .int [] "philox_offset") (.constInt 0xffffffff))),
    .assign .nat [] "c1" (.intToUnsigned 32
      (.intBitAnd 64 .nil (.intShiftRight .nil (.ref .int [] "philox_offset") (.constNat 32))
        (.constInt 0xffffffff))),
    .assign .nat [B] "i4" (indexOp swapped B),
    .assign .nat [B] "c0" (.intToUnsigned 32 (.add .int .scalarL
      (.castNatToInt (.ref .nat [] "c0")) (.castNatToInt (.ref .nat [B] "i4")))),
    .assign .nat [B] "_O" (.intToUnsigned 32 (.mul .int .scalarR
      (.castNatToInt (.ref .nat [B] "c0")) (.constInt 0))),
    tupleBlock B,
    affineStmt B "r0" lower upper,
    affineStmt B "r1" lower upper,
    affineStmt B "r2" lower upper,
    affineStmt B "r3" lower upper,
    .assign .nat [B] "off_0" (.add .nat .scalarL
      (.mul .nat .nil (.mul .nat .nil (.programId 0) (.ref .nat [] "BLOCK_arg")) (.constNat 4))
      (.arange B)),
    .assign .nat [B] "off_1" (.add .nat .scalarR (.ref .nat [B] "off_0") (.ref .nat [] "BLOCK_arg")),
    .assign .nat [B] "off_2" (.add .nat .scalarR (.ref .nat [B] "off_1") (.ref .nat [] "BLOCK_arg")),
    .assign .nat [B] "off_3" (.add .nat .scalarR (.ref .nat [B] "off_2") (.ref .nat [] "BLOCK_arg")),
    storeStmt y N B "r0" "off_0", storeStmt y N B "r1" "off_1",
    storeStmt y N B "r2" "off_2", storeStmt y N B "r3" "off_3"] }

theorem source_eq (swapped : Bool) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Erasure.reference swapped y N seed offset lower upper B hB =
      kernel swapped y N seed offset lower upper B := by
  cases swapped
  all_goals simp only [Erasure.reference, source, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, Except.bind, bind]
  all_goals simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#axiomsClean source_eq

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Algorithm
