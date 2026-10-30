import bench.tritonbench_g.int8_matmul_quantization.Proofs.Numeric
import VeriTile.Triton.Float.IntegerConversion

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelNativeModel
open VeriTile.Triton
noncomputable section

/-- PTX signed saturation is an integer operation, never an experimental
approximation or an assumption that saturation equals modular arithmetic. -/
def clampSigned (width : Nat) (z : Int) : Int :=
  max (-(2^(width-1) : Int)) (min ((2^(width-1) : Int)-1) z)

def instructionTotal (xs ys : Fin 32 → Byte) (old : BitVec 32) : Int :=
  old.toInt + ∑ i, (xs i).toInt * (ys i).toInt

/-- One fixed K32 instruction coordinate. The satfinite MMA consumes the
actual old accumulator; non-saturating WGMMA wraps its signed32 destination.
See PTX ISA `mma` and `wgmma.mma_async` integer accumulation semantics. -/
def instructions : FixedISAProfile where
  result := fun kind xs ys old => BitVec.ofInt 32 <|
    match kind with
    | .mmaM16N8K32Satfinite => clampSigned 32 (instructionTotal xs ys old)
    | .wgmmaM64K32 _ => instructionTotal xs ys old

/-- The observed SM80 source conversion is cvt.rzi.s16.f32 followed by a
low-byte store. PTX float-to-integer conversion clamps overflow/infinities;
NaN converts to zero for this signed16 destination. All FP32 words remain
admitted, including zero-row NaN quotients. This is not toSigned-8 admission.
Reference: PTX ISA Data Movement and Conversion Instructions, `cvt`. -/
def integer8 (x : Word32) : Byte :=
  BitVec.ofInt 8 <| match x.decode with
    | .nan .. => 0
    | .infinity negative => if negative then -(2^15 : Int) else 2^15-1
    | .finite negative magnitude => clampSigned 16
        (FP.IntegerConversion.truncate (if negative then -magnitude else magnitude))

/-- Explicit native floating primitive selections. Each field is one scalar
instruction, at its actual width. These are model choices, not deferred
numerical rewrite laws and not row/kernel/output-state callbacks. -/
structure FloatingInstructions where
  maximum : Word32 → Word32 → Word32
  divide : Word32 → Word32 → Word32
  atomic32 : Word32 → Word32 → Word32
  atomic16 : Word16 → Word16 → Word16

def widen (x : Word16) : Word32 :=
  ⟨TritonBenchG.MatrixReduction.BinaryConversion.widen x.bits⟩

def operations (f : FloatingInstructions) : Ops (BitVec 32) Word32 Word16 where
  toOps := { Int8DequantMatmul.fpOps instructions with hadd := f.atomic16 }
  widen := widen
  add := f.atomic32
  absolute := FP.Value.abs
  maximum := f.maximum
  divide := f.divide
  scale127 := fun x => f.divide x (FP.Value.ofBits .fp32 0x42fe0000)
  integer8 := integer8

theorem integer8_nonfinite :
    integer8 (FP.Value.infinity .fp32) = (255 : Byte) ∧
    integer8 (FP.Value.infinity .fp32 (negative := Bool.true)) = (0 : Byte) ∧
    integer8 (FP.Value.qNaN .fp32) = (0 : Byte) := by
  decide +kernel

/-- A finite source quotient outside signed8 range is still executed at the
observed signed16 conversion width, then narrowed by the actual store. -/
theorem integer8_128 : integer8 (FP.Value.ofBits .fp32 0x43000000) = (128 : Byte) := by
  decide +kernel

theorem scale127_literal : (FP.Value.ofBits .fp32 0x42fe0000).toRat? = some 127 := by
  decide +kernel

#axiomsClean integer8_nonfinite
#axiomsClean integer8_128
#axiomsClean scale127_literal
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelNativeModel
