import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Kernels
import VeriTile.Triton.Float.BitValue

/-! Python admits finite positive `float`; its FP32 conversion may underflow or overflow. The host value
is its complete binary64 encoding. Runtime FP32 argument packing rounds the
exact finite value once, to nearest with ties to even, retaining the zero sign.
This local ABI law is separate from literals and from every kernel result. -/
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.EpsilonABI
open VeriTile.Triton
abbrev HostWord := BitVec 64
abbrev Word := BitVec 32

def sign (host : HostWord) : Bool := host.toNat / 2^63 != 0
def exponent (host : HostWord) : Nat := host.toNat / 2^52 % 2048
def fraction (host : HostWord) : Nat := host.toNat % 2^52

def magnitude (host : HostWord) : Rat :=
  let e:=exponent host
  let significand:=if e=0 then fraction host else 2^52+fraction host
  Rat.ofInt significand * FP.pow2 ((if e=0 then -1022 else (e:Int)-1023)-52)

def value (host : HostWord) : Rat := if sign host then -magnitude host else magnitude host

def convert (host : HostWord) : Word :=
  (FP.round .fp32 .ieeeValue (value host) (sign host)).bits

/-- `host` is the actual Python scalar's binary64 encoding; `runtime` is the
FP32 argument installed by the launcher. `finite` restates the existing guard,
`realValue` connects the mathematical parameter, and `converted` is the exact
single-scalar ABI binding. No source execution or output occurs in this law. -/
structure Binding (eps : ℝ) where
  host : HostWord
  runtime : Word
  finite : exponent host<2047
  realValue : (value host : ℝ)=eps
  converted : runtime=convert host

theorem one : convert 0x3ff0000000000000=0x3f800000 := by decide +kernel
theorem positive_underflow : convert 0x0000000000000001=0x00000000 := by decide +kernel
theorem positive_overflow : convert 0x7fefffffffffffff=0x7f800000 := by decide +kernel
theorem halfway_even : convert 0x3ff0000010000000=0x3f800000 := by decide +kernel
#axiomsClean one
#axiomsClean positive_underflow
#axiomsClean positive_overflow
#axiomsClean halfway_even
#guard_msgs (drop info) in
#auditStmt Binding
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsnormFusedLlama.EpsilonABI
