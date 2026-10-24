import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerKernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Precision
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerPrecision
open VeriTile.Triton
open Precision
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩
/-- All 24 integer-exponent templates retain the int32 input carrier and
native integer-power AST, with the scalar fp32 cast and actual destination
conversion. Padding and every source branch remain independently checked. -/
theorem computation (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    let t := tail oneTile (TypedIntegerSource.kernel output oneTile candidate base X Y
      sx sy N tasks tiles B).surfaceBody
    t.length = 7 ∧
    t[3]! = .assign .int [B] "in0" (.alg (.load .int
      (.blockPtr (.ref .blockPtr [B] "in0_bptr") [0]) .none .unspecified)) ∧
    t[4]! = .assign .real [B] "out0" (.compute (.alg .fp32
      (.libdevicePowi .scalarL (.castFloat .real .real (.const base)) (.ref .int [B] "in0")))) ∧
    t[6]! = store output B := by
  cases output <;> cases oneTile <;> cases candidate <;>
    exact ⟨rfl, rfl, rfl, rfl⟩

/-- Both source branches retain the unused tile count. The grid-stride loop
keeps its runtime trip count and launch-grid-based tile-id computation. -/
theorem control (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    let body := (TypedIntegerSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B).surfaceBody
    body.length = 4 ∧
    body[1]! = .assign .nat [] "num_tiles0" (.alg (.div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat N) (.constNat B)) (.constNat 1)) (.constNat B))) ∧
    body[2]! = .assign .nat [] "num_ctas" (.alg (.numPrograms 0)) ∧
    body[3]! = .forRangeDyn "j" (.constNat 0) (.constNat tiles) (.constNat 1) (loopBody body[3]!) ∧
    (loopBody body[3]!)[0]! = .assign .nat [] "tile_id" (.alg (.add .nat .nil
      (.ref .nat [] "pid") (.mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas")))) := by
  cases output <;> cases candidate <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerPrecision
