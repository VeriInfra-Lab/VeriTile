import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.CallRandomABI
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def signed32 : CallSignature where
  pointers := [("out",.fp32)]
  scalars := [("seed",.i32)]
  constexpr := ["B"]
def signed64 : CallSignature := { signed32 with scalars := [("seed",.i64)] }
def unsigned64 : CallSignature := { signed32 with scalars := [("seed",.u64)] }

def sampleI32 (out : RegionName) (value : Int) : ComputeKernel :=
  let B := (4 : Nat)
  with_triton_call signed32 call_args [seed := value] in
  triton {
    offsets = tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }
def sampleI64 (out : RegionName) (value : Int) : ComputeKernel :=
  let B := (4 : Nat)
  with_triton_call signed64 call_args [seed := value] in
  triton {
    offsets = tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }
def sampleU64 (out : RegionName) (value : Nat) : ComputeKernel :=
  let B := (4 : Nat)
  with_triton_call unsigned64 call_args [seed := value] in
  triton {
    offsets = tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }

-- Distinct ABI profiles must reach the actual RNG instruction, including the
-- signed operand and the unsigned operand's mathematical Int injection.
example (out : RegionName) (seed : Int) :
    (sampleI32 out seed).surfaceBody[1]? = some (
      .assign .real [4] "random" (.compute (.alg .fp32
        (.rand { seedType := .i32, offsetType := .i32 }
          (.constInt seed) (.castNatToInt (.ref .nat [4] "offsets")))))) := rfl
example (out : RegionName) (seed : Int) :
    (sampleI64 out seed).surfaceBody[1]? = some (
      .assign .real [4] "random" (.compute (.alg .fp32
        (.rand { seedType := .i64, offsetType := .i32 }
          (.constInt seed) (.castNatToInt (.ref .nat [4] "offsets")))))) := rfl
example (out : RegionName) (seed : Nat) :
    (sampleU64 out seed).surfaceBody[1]? = some (
      .assign .real [4] "random" (.compute (.alg .fp32
        (.rand { seedType := .u64, offsetType := .i32 }
          (.constInt (Int.ofNat seed)) (.castNatToInt (.ref .nat [4] "offsets")))))) := rfl

/-- A constexpr seed cast is a source ABI declaration; range-bound
normalization must not erase its i64 provenance before the RNG prepass. -/
def literalCastSeed (out : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("out",.fp32)]} : CallSignature) in
  triton {
    seed = tl.cast(5, tl.int64)
    offsets = tl.arange(0, 4)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }
example (out : RegionName) :
    (literalCastSeed out).surfaceBody[2]? = some (
      .assign .real [4] "random" (.compute (.alg .fp32
        (.rand { seedType := .i64, offsetType := .i32 }
          (.ref .int [] "seed") (.castNatToInt (.ref .nat [4] "offsets")))))) := rfl

-- Width metadata cannot make an out-of-range seed executable.
theorem invalid_i32 {A : Type} [Inhabited A] (M : Algebra A) (out : RegionName) (s : State A) :
    FP.Structural.exec M (sampleI32 out (2^31)) s = none := by
  simp [sampleI32,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Random.sourceIntCallValue,
    Random.SourceProfile.inputsValid,Random.IntegerType.inRange]
theorem invalid_u64 {A : Type} [Inhabited A] (M : Algebra A) (out : RegionName) (s : State A) :
    FP.Structural.exec M (sampleU64 out (2^64)) s = none := by
  simp [sampleU64,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Random.sourceNatCallValue,
    Random.SourceProfile.inputsValid,Random.IntegerType.inRange]

def probabilitySignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("p",.fp32)]
def threshold (X Y : RegionName) (probability : ℝ) : ComputeKernel :=
  with_triton_call probabilitySignature call_args [p := probability] in
  triton {
    x = tl.load(X)
    keep = x > p
    output = tl.where(keep, x, 0.0)
    tl.store(Y, output)
  }
example (X Y : RegionName) (p : ℝ) :
    (threshold X Y p).surfaceBody[1]? = some (
      .assign .bool [] "keep" (.alg (.gt .real .nil
        (.withPrecision .real .fp32 (.ref .real [] "x"))
        (.withPrecision .real .fp32 (.const p))))) := rfl

/-- error: unsigned per-call integer value must have a Nat carrier -/
#guard_msgs (error) in
example (out : RegionName) (value : Int) : ComputeKernel :=
  let B := (4 : Nat)
  with_triton_call unsigned64 call_args [seed := value] in
  triton {
    offsets = tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }

/-- error: tl.rand: seed has no supported source integer profile; annotate a direct external scalar binding with tl.source_integer -/
#guard_msgs (error) in
example (out : RegionName) : ComputeKernel :=
  let B := (4 : Nat)
  with_triton_call ({ signed32 with scalars := [("seed",.integer)] } : CallSignature)
    call_args [seed := (5 : Int)] in
  triton {
    offsets = tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(out + offsets, random)
  }

#axiomsClean invalid_i32
#axiomsClean invalid_u64
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.CallRandomABI
