import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerKernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSource
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def offset (candidate : Bool) (B : Nat) : Op .nat [] :=
  if candidate then .mul .nat .nil (.constNat B) (.ref .nat [] "tile_id0")
  else .mul .nat .nil (.ref .nat [] "tile_id0") (.constNat B)

/-- The actual seven-statement source tail after resolving its tile id. -/
def perTile (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N B : Nat) : List Stmt :=
  [ .assign .nat [] "tile_id0" (.ref .nat [] "tile_id"),
    .assign .nat [] "offset0" (offset candidate B),
    .assign .blockPtr [B] "in0_bptr" (.makeBlockPtrDynOffsets X (.constNat 0)
      [N] [B] [sx] [.ref .nat [] "offset0"]),
    .assign .int [B] "in0" (.load .int
      (.blockPtr (.ref .blockPtr [B] "in0_bptr") [0]) .none .unspecified),
    .assign .real [B] "out0" (.libdevicePowi .scalarL (.const base) (.ref .int [B] "in0")),
    .assign .blockPtr [B] "out0_bptr" (.makeBlockPtrDynOffsets Y (.constNat 0)
      [N] [B] [sy] [.ref .nat [] "offset0"]),
    .store .real [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.ref .real [B] "out0") .none ]

def numTiles (N B : Nat) : Stmt :=
  .assign .nat [] "num_tiles0" (.div .nat .nil
    (.sub .nat .nil (.add .nat .nil (.constNat N) (.constNat B)) (.constNat 1))
    (.constNat B))

def loop (candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N B : Nat) : List Stmt :=
  .assign .nat [] "tile_id" (.add .nat .nil (.ref .nat [] "pid")
    (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas"))) ::
      perTile candidate base X Y sx sy N B

def body (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tiles B : Nat) : List Stmt :=
  [ .assign .nat [] "pid" (.programId 0), numTiles N B ] ++
    if oneTile then .assign .nat [] "tile_id" (.ref .nat [] "pid") ::
      perTile candidate base X Y sx sy N B
    else [ .assign .nat [] "num_ctas" (.numPrograms 0),
      .forRangeDyn "j" (.constNat 0) (.constNat tiles) (.constNat 1)
        (loop candidate base X Y sx sy N B) ]

/-- Explicit complete real source, checked against every typed source profile
by `projection`; it is not obtained by running the kernel or reading outputs. -/
def kernel (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) : Kernel :=
  ⟨[X], [Y], body oneTile candidate base X Y sx sy N tiles B⟩

/-- Every named stage is a consecutive part of the actual whole source. -/
theorem body_eq (oneTile candidate : Bool) (base : ℝ) (X Y : RegionName)
    (sx sy N tasks tiles B : Nat) :
    (kernel oneTile candidate base X Y sx sy N tasks tiles B).body =
      body oneTile candidate base X Y sx sy N tiles B := by
  rfl

/-- All six destination labels project to the same complete integer-exponent source. -/
theorem projection (output : TypedIntegerSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    (TypedIntegerSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgorithm? =
      Except.ok (kernel oneTile candidate base X Y sx sy N tasks tiles B) := by
  cases output <;> cases oneTile <;> cases candidate <;>
    simp only [TypedIntegerSource.kernel, TypedIntegerSource.fp32Kernel, TypedIntegerSource.fp64Kernel,
      TypedIntegerSource.fp16Kernel, TypedIntegerSource.bf16Kernel, TypedIntegerSource.f8e4Kernel,
      TypedIntegerSource.f8e5Kernel, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, List.map_cons, List.map_nil]
  all_goals
    simp [kernel, body, loop, perTile, offset, numTiles,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType]

theorem erase (output : TypedIntegerSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    (TypedIntegerSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B).eraseDType.toAlgKernel =
      kernel oneTile candidate base X Y sx sy N tasks tiles B := by
  simp only [ComputeKernel.toAlgKernel, projection]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSource
