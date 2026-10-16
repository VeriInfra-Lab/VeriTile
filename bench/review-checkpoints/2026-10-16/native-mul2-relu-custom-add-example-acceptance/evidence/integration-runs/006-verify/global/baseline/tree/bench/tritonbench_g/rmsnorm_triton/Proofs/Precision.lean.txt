import bench.tritonbench_g.rmsnorm_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Precision
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option maxErrors 5
local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loopBody : ComputeStmt → List ComputeStmt
  | .forRange _ _ _ _ body => body
  | _ => []

def inputOperand (input : Storage) (B : Nat) : Op .real [B] :=
  match input with
  | .fp32 => .castFloat .real .real (.ref .real [B] "x")
  | .fp64 => .castFloat .real .real (.withPrecision .real .fp64 (.ref .real [B] "x"))
  | .fp16 => .castFloat .fp16 .real (.ref .fp16 [B] "x")
  | .bf16 => .castFloat .bf16 .real (.ref .bf16 [B] "x")
  | .f8e4 => .castFloat .f8e4 .real (.ref .f8e4 [B] "x")
  | .f8e5 => .castFloat .f8e5 .real (.ref .f8e5 [B] "x")

def weighted (weight : Storage) (B : Nat) : ComputeStmt :=
  let x : Op .real [B] := .ref .real [B] "x_hat"
  let w : Op .real [B] := match weight with
    | .fp32 | .fp64 => .ref .real [B] "rms_w"
    | .fp16 => .castFloat .fp16 .real (.ref .fp16 [B] "rms_w")
    | .bf16 => .castFloat .bf16 .real (.ref .bf16 [B] "rms_w")
    | .f8e4 => .castFloat .f8e4 .real (.ref .f8e4 [B] "rms_w")
    | .f8e5 => .castFloat .f8e5 .real (.ref .f8e5 [B] "rms_w")
  .assign .real [B] "out" (if weight = .fp64 then
    .compute (.alg .fp64 (.mul .real (.consSame .nil) (.withPrecision .real .fp32 x) w))
    else .compute (.alg .fp32 (.mul .real (.consSame .nil) x w)))

def outputStore (weight output : Storage) (O : RegionName) (B : Nat) : ComputeStmt :=
  let wide := decide (weight = .fp64)
  let raw : Op .real [B] := .ref .real [B] "out"
  let value : Op .real [B] := if wide then .withPrecision .real .fp64 raw
    else .withPrecision .real .fp32 raw
  let p (d : TileDType) : MemAccess d [B] := .region O.cast (.ref .nat [B] "out_off")
  let mask : Op .bool [B] := .ref .bool [B] "x_ptr_mask"
  match output with
  | .fp32 => .store .real [B] (p .real) (.compute (.alg .fp32
      (if wide then .castFloat .real .real value else raw))) (.mask mask)
  | .fp64 => .store .real [B] (p .real) (.compute (.alg .fp64
      (if wide then raw else .castFloat .real .real (.withPrecision .real .fp32 raw)))) (.mask mask)
  | .fp16 => .store .fp16 [B] (p .fp16) (.alg (.castFloat .real .fp16 value)) (.mask mask)
  | .bf16 => .store .bf16 [B] (p .bf16) (.alg (.castFloat .real .bf16 value)) (.mask mask)
  | .f8e4 => .store .f8e4 [B] (p .f8e4) (.alg (.castFloat .real .f8e4 value)) (.mask mask)
  | .f8e5 => .store .f8e5 [B] (p .f8e5) (.alg (.castFloat .real .f8e5 value)) (.mask mask)

/-- All 216 profiles retain an integer exponent and fp32 input conversion,
powi, accumulation, mean and rsqrt. Only fp64 weight widens the final product;
the destination conversion preserves that product's precision. -/
theorem computation_profiles (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    let body := (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).surfaceBody
    (loopBody body[5]!)[3]! =
      .assign .real [B] "var" (.compute (.alg .fp32
        (.add .real (.consSame .nil) (.ref .real [B] "var")
          (.libdevicePowi .scalarR (inputOperand input B) (.constInt 2))))) ∧
    body[6]! = .assign .real [] "var" (.compute (.alg .fp32
      (.div .real .nil (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [B] "var"))
        (.const (N : ℝ))))) ∧
    body[7]! = .assign .real [] "rstd" (.compute (.alg .fp32
      (.rsqrt (.add .real .nil (.ref .real [] "var") (.const eps))))) ∧
    (loopBody body[8]!)[5]! = weighted weight B ∧
    (loopBody body[8]!)[7]! = outputStore weight output O B := by
  cases input <;> cases weight <;> cases output <;> cases v <;>
    exact ⟨rfl, rfl, rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Precision
