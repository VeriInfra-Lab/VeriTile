import bench.tritonbench_g.sin_computation.Proofs.Precision
import bench.tritonbench_g.vector_addition.Proofs.NativeArithmetic

namespace SinComputationNative
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SinComputation
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
abbrev Input := InputStorage
structure Storage where
  input : Input
  output : DType
  deriving DecidableEq
abbrev Word (_d : Storage) := BitVec 64
def inputDType (d : Storage) : DType := match d.input with | .fp32 => .fp32 | .fp64 => .fp64
def dtype (d : Storage) : DType := d.output

/-- Exact scalar-only ports. Integer conversions below are concrete and partial;
no callback may define an undefined conversion or supply a memory/result oracle. -/
structure Numerics where
  sin32 : BitVec 32 → BitVec 32
  sin64 : BitVec 64 → BitVec 64
  cast32to64 : BitVec 32 → BitVec 64
  cast32to16 : BitVec 32 → BitVec 16
  cast32toBF16 : BitVec 32 → BitVec 16
  cast32toE4 : BitVec 32 → BitVec 8
  cast32toE5 : BitVec 32 → BitVec 8
  cast64to32 : BitVec 64 → BitVec 32
  cast64to16 : BitVec 64 → BitVec 16
  cast64toBF16 : BitVec 64 → BitVec 16

def sine (b : Numerics) : Input → BitVec 64 → BitVec 64
  | .fp32,x => (b.sin32 (x.setWidth 32)).setWidth 64
  | .fp64,x => b.sin64 x

/-- Exact IEEE binary decoding followed by truncation toward zero. Nonfinite
words have no integer image. Subnormal and signed-zero words are retained. -/
def truncatedInteger (input : Input) (x : BitVec 64) : Option Int :=
  let fractionBits : Nat := match input with | .fp32 => 23 | .fp64 => 52
  let exponentBits : Nat := match input with | .fp32 => 8 | .fp64 => 11
  let bias : Nat := 2^(exponentBits-1)-1
  let raw : Nat := x.toNat % 2^(fractionBits+exponentBits+1)
  let exponent : Nat := (raw / 2^fractionBits) % 2^exponentBits
  let fraction : Nat := raw % 2^fractionBits
  if exponent=2^exponentBits-1 then none
  else
    let significand : Nat := if exponent=0 then fraction else 2^fractionBits+fraction
    let power : Int := (if exponent=0 then 1 else exponent : Nat) - (bias : Int) - (fractionBits : Int)
    let magnitude : Nat := if 0≤power then significand*2^power.toNat else significand/2^(-power).toNat
    some (if raw/2^(fractionBits+exponentBits)=0 then (magnitude : Int) else -(magnitude : Int))

def integerCast (input : Input) (signed : Bool) (width : Nat) (x : BitVec 64) : Option (BitVec 64) := do
  let z ← truncatedInteger input x
  if (if signed then -(2^(width-1) : Int)≤z ∧ z<2^(width-1) else 0≤z ∧ z<2^width) then
    pure (BitVec.ofInt 64 z)
  else none

/-- Store conversion follows the frontend's actual element type. Bool pointers
are cast to i8 pointers and receive signed i8 values, without normalization.
The fp64→fp8 routes are rejected before this evaluator by backend admission. -/
def convert (b : Numerics) (d : Storage) (x : BitVec 64) : Option (BitVec 64) :=
  match d.output with
  | .b1 | .i8 => integerCast d.input Bool.true 8 x
  | .u8 => integerCast d.input Bool.false 8 x
  | .i16 => integerCast d.input Bool.true 16 x
  | .u16 => integerCast d.input Bool.false 16 x
  | .i32 => integerCast d.input Bool.true 32 x
  | .u32 => integerCast d.input Bool.false 32 x
  | .i64 => integerCast d.input Bool.true 64 x
  | .u64 => integerCast d.input Bool.false 64 x
  | .fp32 => some (match d.input with | .fp32 => x | .fp64 => (b.cast64to32 x).setWidth 64)
  | .fp64 => some (match d.input with | .fp32 => b.cast32to64 (x.setWidth 32) | .fp64 => x)
  | .fp16 => some (match d.input with | .fp32 => (b.cast32to16 (x.setWidth 32)).setWidth 64 | .fp64 => (b.cast64to16 x).setWidth 64)
  | .bf16 => some (match d.input with | .fp32 => (b.cast32toBF16 (x.setWidth 32)).setWidth 64 | .fp64 => (b.cast64toBF16 x).setWidth 64)
  | .fp8e4nv => match d.input with | .fp32 => some ((b.cast32toE4 (x.setWidth 32)).setWidth 64) | .fp64 => none
  | .fp8e5 => match d.input with | .fp32 => some ((b.cast32toE5 (x.setWidth 32)).setWidth 64) | .fp64 => none
  | .fp8e5b16 | .fp8e4b8 => none

inductive Expr (_d : Storage) where
  | input
  | sine (value : Expr _d)
  | convert (value : Expr _d)
def Expr.eval (b : Numerics) (x : Word d) : Expr d → Option (Word d)
  | .input => some x
  | .sine e => (e.eval b x).map (SinComputationNative.sine b d.input)
  | .convert e => (e.eval b x).bind (SinComputationNative.convert b d)
def source (d : Storage) : Expr d := .convert (.sine .input)
def result (b : Numerics) (d : Storage) (x : Word d) : Option (Word d) := convert b d (sine b d.input x)
theorem source_scalar (b : Numerics) (d : Storage) (x : Word d) :
    (source d).eval b x=result b d x := rfl

/-- Retains every earlier typed source family, including abstract fp64→fp8
translation even though the pinned backend rejects those two routes. -/
theorem retained_source_precision (input : Input) (output : OutputStorage) (X Y : RegionName) (n : Nat) :
    (typedKernel input output Bool.false X Y n 4).surfaceBody.drop 4 =
      [Precision.expectedLoad input X 4,Precision.expectedSin input 4,Precision.expectedStore input output Y 4] :=
  Precision.source_precision input output Bool.false X Y n 4

theorem integer_cast_nan_undefined : integerCast .fp32 Bool.true 8 (BitVec.ofNat 64 0x7fc00000)=none := by decide +kernel
theorem integer_cast_unsigned_negative_undefined : integerCast .fp32 Bool.false 8 (BitVec.ofNat 64 0xbf800000)=none := by decide +kernel
theorem integer_cast_negative_fraction : integerCast .fp32 Bool.false 8 (BitVec.ofNat 64 0xbf000000)=some 0 := by decide +kernel
theorem integer_cast_bool_raw_minus_one : integerCast .fp32 Bool.true 8 (BitVec.ofNat 64 0xbf800000)=some (BitVec.ofInt 64 (-1)) := by decide +kernel
#specNonCircular result avoiding [source,Expr.eval]
#axiomsClean source_scalar
#axiomsClean retained_source_precision
#axiomsClean integer_cast_nan_undefined
#axiomsClean integer_cast_unsigned_negative_undefined
#axiomsClean integer_cast_negative_fraction
#axiomsClean integer_cast_bool_raw_minus_one
#guard_msgs (drop info) in
#auditModuleAxioms
end SinComputationNative
