import Mathlib.Tactic
import bench.tritonbench_g.matrix_reduction.Proofs.OrderKey
set_option exponentiation.threshold 512
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion

/-- IEEE binary16 magnitude bits converted to binary32 magnitude bits. The ten
subnormal cases normalize the leading significand bit; normal values rebias the
exponent by 112 and append thirteen zero fraction bits. -/
def widenMagnitude (m : Nat) : Nat :=
  if 31744 ≤ m then 0x7f800000 + (m % 1024) * 8192
  else if 1024 ≤ m then (m + 0x1c000) * 8192
  else if 512 ≤ m then 939524096 + (m - 512) * 16384
  else if 256 ≤ m then 931135488 + (m - 256) * 32768
  else if 128 ≤ m then 922746880 + (m - 128) * 65536
  else if 64 ≤ m then 914358272 + (m - 64) * 131072
  else if 32 ≤ m then 905969664 + (m - 32) * 262144
  else if 16 ≤ m then 897581056 + (m - 16) * 524288
  else if 8 ≤ m then 889192448 + (m - 8) * 1048576
  else if 4 ≤ m then 880803840 + (m - 4) * 2097152
  else if 2 ≤ m then 872415232 + (m - 2) * 4194304
  else if 1 ≤ m then 0x33800000
  else 0

def widen (x : BitVec 16) : BitVec 32 :=
  BitVec.ofNat 32 ((x.toNat / 32768) * 2147483648 + widenMagnitude (x.toNat % 32768))

/-- Exact narrowing on the image of `widen`. General binary32 rounding is not
claimed: the max kernel only stores selected widened input values. -/
def narrowMagnitude (m : Nat) : Nat :=
  if 2139095040 ≤ m then 31744 + (m / 8192) % 1024
  else if 947912704 ≤ m then m / 8192 - 114688
  else if 939524096 ≤ m then (m % 8388608 + 8388608) / 16384
  else if 931135488 ≤ m then (m % 8388608 + 8388608) / 32768
  else if 922746880 ≤ m then (m % 8388608 + 8388608) / 65536
  else if 914358272 ≤ m then (m % 8388608 + 8388608) / 131072
  else if 905969664 ≤ m then (m % 8388608 + 8388608) / 262144
  else if 897581056 ≤ m then (m % 8388608 + 8388608) / 524288
  else if 889192448 ≤ m then (m % 8388608 + 8388608) / 1048576
  else if 880803840 ≤ m then (m % 8388608 + 8388608) / 2097152
  else if 872415232 ≤ m then (m % 8388608 + 8388608) / 4194304
  else if 864026624 ≤ m then 1
  else 0

def narrow (x : BitVec 32) : BitVec 16 :=
  BitVec.ofNat 16 ((x.toNat / 2147483648) * 32768 + narrowMagnitude (x.toNat % 2147483648))

/-- Exact unsigned finite magnitude in units of 2^-149, expressed independently
from conversion by each format's exponent and significand fields. -/
def magnitude16 (x : BitVec 16) : Nat :=
  let m := x.toNat % 32768
  let e := m / 1024
  let f := m % 1024
  if e = 0 then f * 2^125 else (1024 + f) * 2^(e + 124)

def magnitude32 (x : BitVec 32) : Nat :=
  let m := x.toNat % 2147483648
  let e := m / 8388608
  let f := m % 8388608
  if e = 0 then f else (8388608 + f) * 2^(e - 1)

end VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion
