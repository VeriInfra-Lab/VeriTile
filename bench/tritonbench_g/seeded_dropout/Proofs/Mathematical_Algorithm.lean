import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Erasure
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Algorithm
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def seedStmt (kind : Random.IntegerType) (seed : SeedCarrier kind) : Stmt :=
  match kind, seed with
  | .i32, s | .i64, s => .assign .int [] "seed_arg" (.constInt s)
  | .u32, s | .u64, s => .assign .nat [] "seed_arg" (.constNat s)

def seedOp : Random.IntegerType → Op .int []
  | .i32 | .i64 => .ref .int [] "seed_arg"
  | .u32 | .u64 => .castNatToInt (.ref .nat [] "seed_arg")

/-- A checked normal form of the complete erased source; not a replacement
specification. It retains every source register, all eager operands and the
omitted padding on the input load. -/
def kernel (kind : Random.IntegerType) (swapped : Bool) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) : Kernel :=
  { inputs := [x], outputs := [y], body := [
    seedStmt kind seed,
    .assign .nat [] "BLOCK_SIZE_arg" (.constNat B),
    .assign .nat [] "pid" (.programId 0),
    .assign .nat [] "block_start" (if swapped then
      .mul .nat .nil (.ref .nat [] "BLOCK_SIZE_arg") (.ref .nat [] "pid") else
      .mul .nat .nil (.ref .nat [] "pid") (.ref .nat [] "BLOCK_SIZE_arg")),
    .assign .nat [B] "offsets" (.add .nat .scalarL (.ref .nat [] "block_start") (.arange B)),
    .assign .bool [B] "mask" (.lt .nat .scalarR (.ref .nat [B] "offsets") (.constNat N)),
    .assign .real [B] "x" (.load .real (.region x (.ref .nat [B] "offsets"))
      (.mask (.ref .bool [B] "mask"))),
    .assign .real [B] "random" (.rand (RandomDomain.profile kind) (seedOp kind)
      (.castNatToInt (.ref .nat [B] "offsets"))),
    .assign .bool [B] "x_keep" (.gt .real .scalarR (.ref .real [B] "random") (.const p)),
    .assign .real [B] "output" (.where (.ref .bool [B] "x_keep")
      (.div .real .scalarR (.ref .real [B] "x") (.sub .real .nil (.const 1) (.const p)))
      (.broadcast (.const 0.0) [B])),
    .store .real [B] (.region y (.ref .nat [B] "offsets")) (.ref .real [B] "output")
      (.mask (.ref .bool [B] "mask"))] }

theorem source_eq (kind : Random.IntegerType) (swapped : Bool) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Erasure.reference kind swapped x y N p seed B hB = kernel kind swapped x y N p seed B := by
  cases kind <;> cases swapped
  all_goals
      simp only [Erasure.reference, source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType, List.map_cons, List.map_nil, Kernel.eraseDType,
        Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
        MaskOpt.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
        Except.bind, bind]
      simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
      rfl
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Algorithm
