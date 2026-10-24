import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton.Float.BitValue

/-! Python admits finite `float`, including either signed zero. The host value
is its complete binary64 encoding. Runtime FP32 argument packing rounds the
exact finite value once, to nearest with ties to even, retaining the zero sign.
This local ABI law is separate from literals and from every kernel result. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.EpsilonABI
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

theorem positive_zero : convert 0x0000000000000000=0x00000000 := by decide +kernel
theorem negative_zero : convert 0x8000000000000000=0x80000000 := by decide +kernel
theorem one : convert 0x3ff0000000000000=0x3f800000 := by decide +kernel
theorem positive_underflow : convert 0x0000000000000001=0x00000000 := by decide +kernel
theorem negative_underflow : convert 0x8000000000000001=0x80000000 := by decide +kernel
theorem halfway_even : convert 0x3ff0000010000000=0x3f800000 := by decide +kernel

/-- The mathematical zero parameter admits both actual runtime zero signs. -/
noncomputable def positiveZeroBinding : Binding 0 where
  host := 0x0000000000000000
  runtime := 0x00000000
  finite := by decide
  realValue := by
    have h : value 0x0000000000000000=0 := by decide +kernel
    rw [h]; norm_num
  converted := positive_zero.symm

noncomputable def negativeZeroBinding : Binding 0 where
  host := 0x8000000000000000
  runtime := 0x80000000
  finite := by decide
  realValue := by
    have h : value 0x8000000000000000=0 := by decide +kernel
    rw [h]; norm_num
  converted := negative_zero.symm

theorem zero_runtime_distinct : positiveZeroBinding.runtime≠negativeZeroBinding.runtime := by decide +kernel

#axiomsClean zero_runtime_distinct
#axiomsClean positive_zero
#axiomsClean negative_zero
#axiomsClean one
#axiomsClean positive_underflow
#axiomsClean negative_underflow
#axiomsClean halfway_even
#auditStmt Binding
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastRmsLayernorm.EpsilonABI
