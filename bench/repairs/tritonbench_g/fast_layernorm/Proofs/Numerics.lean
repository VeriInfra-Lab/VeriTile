import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Compiler

/-! Exact numerical deferrals. Arithmetic is default FP32, selected by each
actual object's primitive identifiers and compiler options. The domain is all
UInt32 payloads: there is no finite-only premise and no assumed FMA=mul+add,
reduction reassociation, reciprocal identity, or rsqrt accuracy rule.

Compiler.LocalConformance enumerates zero, integer-count conversion, add,
sub, mul, div, negation, FMA and rsqrt. The independent host epsilon conversion
below retains its actual bool/integer/binary64 binding. Tensor lazy-negation
uses the ordinary local FP32 negation interpretation. These obligations say
nothing about a kernel result, successful launch, shape, pointer, byte frame,
allocation, saved statistics, or mathematical derivative. Offline compilation
and CPU source controls do not prove their native/GPU conformance. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Numerics
open Compiler

structure EpsilonConformance (platform : Platform)
    (convert : String → Option Shared.ScalarBinding → Shared.HostScalar → UInt32) : Prop where
  convertExact : ∀ object value,platform.epsilon object value=
    convert object.primitives.epsilonABI object.signature.epsilon value

structure TensorNegation (neg target : UInt32 → UInt32) : Prop where
  exactBits : ∀ bits,neg bits=target bits

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastLayernorm.Numerics
