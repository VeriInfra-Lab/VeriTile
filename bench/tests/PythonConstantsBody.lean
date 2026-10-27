import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Tests.PythonConstantsBody
open VeriTile.Triton DSL.PythonConstants
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

-- Halfway ties, carries, both signs, subnormals and overflow.
#guard roundBinary 53 (-1074) 1023 (mkRat (2^53+1) (2^53)) == some 1
#guard roundBinary 53 (-1074) 1023 (mkRat (2^53+3) (2^53)) == some (mkRat (2^51+1) (2^51))
#guard roundBinary 53 (-1074) 1023 (mkRat (-(2^53+1)) (2^53)) == some (-1)
#guard roundBinary 53 (-1074) 1023 (mkRat 1 (2^1075)) == some 0
#guard roundBinary 53 (-1074) 1023 (mkRat 3 (2^1075)) == some (mkRat 1 (2^1073))
#guard roundBinary 53 (-1074) 1023 (Rat.ofInt (2^1024-2^970)) == none
#guard roundBinary 24 (-149) 127 (mkRat (2^24+1) (2^24)) == some 1
#guard roundBinary 24 (-149) 127 (mkRat (2^24+3) (2^24)) == some (mkRat (2^22+1) (2^22))
#guard roundBinary 24 (-149) 127 (mkRat 3 (2^150)) == some (mkRat 1 (2^148))
-- A binary64/FP32 double-rounding counterexample must not be accepted.
#guard !admissible ⟨mkRat (2^60+2^36+1) (2^60), mkRat (2^24+1) (2^24), Bool.true⟩

/-- Python binary64 decoding and the existing native FP32 encoder agree on
this source constant. The kernel checks the finite rational calculation. -/
theorem geglu_constant_binary64 :
    roundBinary 53 (-1074) 1023 (3 * mkRat 6444110634811895 144115188075855872) =
      some (mkRat 4833082976108921 36028797018963968) := by decide +kernel

theorem geglu_constant_native_word :
    (FP.round .fp32 .ieeeValue (mkRat 4833082976108921 36028797018963968)).raw =
      0x3e095d4f ∧
    FP.round .fp32 .ieeeValue (mkRat 4833082976108921 36028797018963968) =
      FP.round .fp32 .ieeeValue (mkRat 134145 1000000) := by decide +kernel

#axiomsClean geglu_constant_binary64
#axiomsClean geglu_constant_native_word

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = 3 * 0.044715 * x
    tl.store(Y + offsets, y)
  }

def expanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  y = 0.134145 * x
  tl.store(Y + offsets, y)
}

theorem only_literal_product_folded (X Y : RegionName) (B : Nat) :
    source X Y B = expanded X Y B := rfl

def runtime (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    y = x * 3 * 0.044715
    tl.store(Y + offsets, y)
  }

def runtimeExpanded (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  y = x * 3 * 0.044715
  tl.store(Y + offsets, y)
}

theorem runtime_order_retained (X Y : RegionName) (B : Nat) :
    runtime X Y B = runtimeExpanded X Y B := rfl

/-- error: Python literal arithmetic cannot be folded preserving binary64, FP32 promotion and the ideal Real projection -/
#guard_msgs in
def rounded_sum (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    tl.store(Y, 0.1 + 0.2)
  }

/-- error: zero or cancellation in Python floating literal arithmetic is outside the checked subset -/
#guard_msgs in
def cancellation (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    tl.store(Y, 1.0 - 1.0)
  }

#axiomsClean only_literal_product_folded
#axiomsClean runtime_order_retained
end VeriTile.Tests.PythonConstantsBody
