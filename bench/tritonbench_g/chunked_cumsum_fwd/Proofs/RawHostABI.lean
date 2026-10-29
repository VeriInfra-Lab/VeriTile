import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawFacadeSafety
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawABI
open VeriTile.Triton RawBoundary

/-- The ordinary Python `PyFloat_AsDouble` boundary. A successful result is
an unmodified binary64 word, including the sign and payload of a NaN. The
conversion-error branch covers Python type/overflow errors. No GPU result
is supplied by this host scalar boundary. -/
inductive Scalar where
  | converted (bits : BitVec 64)
  | conversionError (reason : String)
  deriving DecidableEq

/-- The pinned x86-64 default-rounding C double-to-float cast used by
Triton 3.7.1 driver.c extractFP32. Finite values round once to nearest-even,
without FTZ. NaNs retain sign/top payload and are quieted. This is an
explicit host ABI algorithm, not a deferred GPU numerical port. -/
def narrow64 (bits : BitVec 64) : BitVec 32 :=
  let negative := bits.toNat / 2^63 != 0
  let exponent : Nat := (bits.toNat / 2^52) % 2048
  let fraction : Nat := bits.toNat % 2^52
  if exponent = 2047 then
    if fraction = 0 then (FP.Value.infinity .fp32 negative).bits
    else BitVec.ofNat 32 ((if negative then 2^31 else 0) + 255*2^23 +
      ((fraction / 2^29) % 2^22) + 2^22)
  else
    let significand : Nat := if exponent = 0 then fraction else 2^52+fraction
    let scale : Int := if exponent = 0 then -1074 else (exponent : Int)-1075
    let magnitude := Rat.ofInt significand * FP.pow2 scale
    (FP.round .fp32 FP.Config.ieeeValue (if negative then -magnitude else magnitude) negative).bits

def extract : Scalar → Except String (BitVec 32)
  | .converted bits => .ok (narrow64 bits)
  | .conversionError reason => .error reason

def extractBounds (lower upper : Scalar) : Except String (BitVec 32 × BitVec 32) :=
  match extract lower with
  | .error reason => .error reason
  | .ok lo => match extract upper with
    | .error reason => .error reason
    | .ok hi => .ok (lo,hi)

/-- Materialized arguments keep both runtime words separately. Placeholders
on an extraction error are unreachable by every kernel launch. -/
def materialize (call : Call Scalar) : Call (BitVec 32) where
  dt := call.dt
  rate := call.rate
  bias := call.bias
  chunk := call.chunk
  softplus := call.softplus
  lower := (extract call.lower).toOption.getD 0
  upper := (extract call.upper).toOption.getD 0

theorem materialize_binding (call : Call Scalar) (bounds : BitVec 32 × BitVec 32)
    (ok : extractBounds call.lower call.upper = .ok bounds) :
    bounds = ((materialize call).lower,(materialize call).upper) := by
  cases hl : extract call.lower <;> cases hu : extract call.upper <;>
    simp [extractBounds,hl,hu] at ok
  subst bounds
  simp [materialize,hl,hu,Except.toOption]

noncomputable section
/-- The setup phase includes the exact launcher extraction after ordinary
compiler/device preparation and before any trial or selected kernel runs.
Extraction failures retain the source's runtime-error outcome. -/
def guardedRuntime (call : Call Scalar)
    (rt : Runtime (BitVec 32) (materialize call) entry preexisting) :
    Runtime (BitVec 32) (materialize call) entry preexisting :=
  { rt with setup := fun g a =>
      match rt.setup g a with
      | .error reason => .error reason
      | .ok plan => match extractBounds call.lower call.upper with
        | .error reason => .error reason
        | .ok _ => .ok plan }

theorem setup_binding (call : Call Scalar)
    (rt : Runtime (BitVec 32) (materialize call) entry preexisting)
    (g : Geometry (materialize call) rt.division)
    (a : Allocated (BitVec 32) (materialize call) g.host entry preexisting)
    (plan : AutotunePlan (BitVec 32) (materialize call) g.host)
    (ok : (guardedRuntime call rt).setup g a = .ok plan) :
    extractBounds call.lower call.upper = .ok ((materialize call).lower,(materialize call).upper) ∧
    rt.setup g a = .ok plan := by
  cases hs : rt.setup g a <;> cases hb : extractBounds call.lower call.upper <;>
    simp [guardedRuntime,hs,hb] at ok
  subst plan
  have binding := materialize_binding call _ hb
  exact ⟨congrArg Except.ok binding,rfl⟩

theorem guarded_safety (call : Call Scalar)
    (rt : Runtime (BitVec 32) (materialize call) entry preexisting)
    (safe : HostSafety rt) : HostSafety (guardedRuntime call rt) := by
  intro g a p setup launch member
  exact safe g a p (setup_binding call rt g a p setup).2 launch member

/-- All original host outcomes; successful launches consume exactly the two
materialized FP32 bound words. The theorem covers all binary64 patterns and
conversion errors, with no finite/positive/order restriction on bounds. -/
theorem host_outcomes (call : Call Scalar) (sizes : HostSizes (materialize call))
    (rt : Runtime (BitVec 32) (materialize call) entry preexisting) :
    HostSpec (materialize call) sizes (guardedRuntime call rt)
      (executeHost (materialize call) sizes (guardedRuntime call rt)) ∧
    (∀ g a p, (guardedRuntime call rt).setup g a = .ok p →
      extractBounds call.lower call.upper = .ok ((materialize call).lower,(materialize call).upper)) := by
  exact ⟨RawBoundary.host_outcomes _ _ _,fun g a p ok => (setup_binding call rt g a p ok).1⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawABI
