import bench.tritonbench_g.pow_scalar_tensor.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Precision
open VeriTile.Triton
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRangeDyn _ _ _ _ body => body
  | _ => []

def tail (oneTile : Bool) (body : List ComputeStmt) : List ComputeStmt :=
  if oneTile then body.drop 3 else (loopBody body[3]!).drop 1

def store (output : TypedSource.FloatingOutput) (B : Nat) : ComputeStmt :=
  let address : MemAccess .real [B] := .blockPtr (.ref .blockPtr [B] "out0_bptr") [0]
  match output with
  | .fp32 => .store .real [B] address
      (.compute (.alg .fp32 (.castFloat .real .real (.ref .real [B] "out0")))) .none
  | .fp64 => .store .real [B] address
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] "out0"))))) .none
  | .fp16 => .store .fp16 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.alg (.castFloat .real .fp16 (.withPrecision .real .fp32 (.ref .real [B] "out0")))) .none
  | .bf16 => .store .bf16 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.alg (.castFloat .real .bf16 (.withPrecision .real .fp32 (.ref .real [B] "out0")))) .none
  | .f8e4 => .store .f8e4 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.alg (.castFloat .real .f8e4 (.withPrecision .real .fp32 (.ref .real [B] "out0")))) .none
  | .f8e5 => .store .f8e5 [B] (.blockPtr (.ref .blockPtr [B] "out0_bptr") [0])
      (.alg (.castFloat .real .f8e5 (.withPrecision .real .fp32 (.ref .real [B] "out0")))) .none

/-- The fp32-exponent path has no hidden fp16 promotion or integer dispatch:
load, explicit input cast and helper power are fp32; the scalar's explicit cast is present,
block padding stays unspecified, and the destination conversion remains. -/
theorem computation (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    let t := tail oneTile (TypedSource.kernel output oneTile candidate base X Y
      sx sy N tasks tiles B).surfaceBody
    t.length = 7 ∧
    t[3]! = .assign .real [B] "in0" (.compute (.alg .fp32
      (.castFloat .real .real (.load .real
        (.blockPtr (.ref .blockPtr [B] "in0_bptr") [0]) .none .unspecified)))) ∧
    t[4]! = .assign .real [B] "out0" (.compute (.alg .fp32
      (.pow .scalarL (.castFloat .real .real (.const base)) (.ref .real [B] "in0")))) ∧
    t[6]! = store output B := by
  cases output <;> cases oneTile <;> cases candidate <;>
    exact ⟨rfl, rfl, rfl, rfl⟩

/-- Both source branches retain the unused tile count. The grid-stride loop
keeps its runtime trip count and launch-grid-based tile-id computation. -/
theorem control (output : TypedSource.FloatingOutput) (candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    let body := (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B).surfaceBody
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
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Precision
