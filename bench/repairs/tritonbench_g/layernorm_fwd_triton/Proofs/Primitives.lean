import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Runtime

/-! Exact local numerical ports. These are obligations about scalar native
instructions and host-scalar conversion only. No field mentions a reduction,
memory, execution, output or Real mathematical equality. The source/core
theorems are parametric in these total operations and do not discharge their
bit-level conformance. All NaNs, infinities, signed zeros and subnormals remain
in scope; a finite-only interpretation is insufficient. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Numerics
open Compiler Runtime

/-- Target scalar semantics identified by actual emitted primitive names.
For the retained default SM90 artifacts the names are add.f32, sub.f32,
mul.f32, fma.rn.f32, div.full.f32 and sqrt.approx.ftz.f32. The count and epsilon
conversions retain their actual signed/unsigned/boolean/FP/constexpr ABI. -/
structure Target (α : Type) where
  constant32 : UInt32 → α
  add : String → α → α → α
  sub : String → α → α → α
  mul : String → α → α → α
  fma : String → α → α → α → α
  divide : String → α → α → α
  sqrt : String → α → α
  count : String → Nat → α
  epsilon : String → ScalarBinding → HostScalar → α

/-- This explicitly named list is the entire remaining numerical boundary
for one selected code object. `epsilon` is required only for a successfully
bound host value, including non-FP32 source ABI slots. `count` covers actual
nonnegative signed64 counts; specialization may erase division only at N=1.
The target callbacks must be interpreted as the selected target's total
scalar semantics, not exact Real arithmetic. No conformance is asserted here. -/
structure LocalConformance (platform : Platform α) (code : Object) (target : Target α) : Prop where
  zero : (platform.scalar code.primitives).zero=target.constant32 0x00000000
  one : (platform.scalar code.primitives).one=target.constant32 0x3f800000
  add : ∀ a b, (platform.scalar code.primitives).add a b=target.add code.primitives.add a b
  sub : ∀ a b, (platform.scalar code.primitives).sub a b=target.sub code.primitives.sub a b
  mul : ∀ a b, (platform.scalar code.primitives).mul a b=target.mul code.primitives.mul a b
  fma : ∀ a b c, (platform.scalar code.primitives).fma a b c=target.fma code.primitives.fma a b c
  divide : ∀ a b, (platform.scalar code.primitives).div a b=target.divide code.primitives.divide a b
  sqrt : ∀ a, (platform.scalar code.primitives).sqrt a=target.sqrt code.primitives.sqrt a
  count : ∀ n : Nat, n<2^63 → (platform.scalar code.primitives).count n=target.count code.primitives.countConvert n
  epsilon : ∀ host, platform.epsilonBinding code.signature.epsilon host →
    platform.epsilon code host=target.epsilon code.primitives.epsABI code.signature.epsilon host

/-- Both explicit source casts are within FP32 in the default profile; their
representation identity is structural and is not a floating arithmetic law. -/
theorem fp32_cast_identity (value : α) : Native.cast32 value=value := rfl

#axiomsClean fp32_cast_identity
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayernormFwdTriton.Numerics
