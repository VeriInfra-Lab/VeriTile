import VeriTile.Triton.DSL.IntegerCallArguments
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.IntegerCallArguments
open VeriTile.Triton
noncomputable section

def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("seed",.integer)]
  constexpr := ["B"]
  writePorts := ["Y"]

def source (Y : RegionName) (seed : IntegerArgument) (B : Nat)
    (legal : Random.ArangeSizeDomain B) : ComputeKernel :=
  with_triton_call signature call_args [] integer_args [seed := seed]
      arange_args [B := legal] in triton {
    pid = tl.program_id(0)
    offsets = pid * B + tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(Y + offsets, random)
  }

/-- Exact complete program for a signed seed. No register cast is substituted
for the source ABI, and the symbolic arange retains its caller-checked size. -/
theorem signed (Y : RegionName) (seed : Int) (B : Nat) (legal : Random.ArangeSizeDomain B) :
    source Y (.i64 seed) B legal = (triton {
      tl.ports inputs=[], outputs=[Y]
      tl.region Y = tl.float32
      seed = $((seed : Int))
      tl.source_integer seed = tl.int64
      pid = tl.program_id(0)
      offsets = pid * $(_root_.VeriTile.Triton.Random.i32ArangeSize B legal) +
        tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B legal))
      random = tl.rand(seed, offsets)
      tl.store(Y + offsets, random)
    }) := rfl

/-- Exact complete program for an unsigned seed. No register cast is substituted
for the source ABI, and the symbolic arange retains its caller-checked size. -/
theorem unsigned (Y : RegionName) (seed : Nat) (B : Nat) (legal : Random.ArangeSizeDomain B) :
    source Y (.u64 seed) B legal = (triton {
      tl.ports inputs=[], outputs=[Y]
      tl.region Y = tl.float32
      seed = $((seed : Nat))
      tl.source_integer seed = tl.uint64
      pid = tl.program_id(0)
      offsets = pid * $(_root_.VeriTile.Triton.Random.i32ArangeSize B legal) +
        tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize B legal))
      random = tl.rand(seed, offsets)
      tl.store(Y + offsets, random)
    }) := rfl

/-- error: integer and arange call arguments must remain immutable -/
#guard_msgs in
def mutated (Y : RegionName) (seed : IntegerArgument) (B : Nat)
    (legal : Random.ArangeSizeDomain B) : ComputeKernel :=
  with_triton_call signature call_args [] integer_args [seed := seed]
      arange_args [B := legal] in triton {
    seed += 1
    pid = tl.program_id(0)
    offsets = pid * B + tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(Y + offsets, random)
  }

def floatSignature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("seed",.fp32)]
  constexpr := ["B"]
  writePorts := ["Y"]

/-- error: integer argument ABI differs from the scalar call type -/
#guard_msgs in
def wrongScalarType (Y : RegionName) (B : Nat) (legal : Random.ArangeSizeDomain B) : ComputeKernel :=
  with_triton_call floatSignature call_args [] integer_args [seed := IntegerArgument.i64 1]
      arange_args [B := legal] in triton {
    pid = tl.program_id(0)
    offsets = pid * B + tl.arange(0, B)
    random = tl.rand(seed, offsets)
    tl.store(Y + offsets, random)
  }

#axiomsClean signed
#axiomsClean unsigned
end
end VeriTile.Tests.IntegerCallArguments
