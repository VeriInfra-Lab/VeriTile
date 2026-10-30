import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelGrid
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelModels
open VeriTile.Triton FP.Structural KernelSource KernelResult
open VeriTile.Bench.Repairs.RowwiseQuantizationTriton (roundEven)
noncomputable section
open Classical

/-- Explicit signed64 range handling for the mathematical llrint interpretation.
It is not a theorem identifying Real arithmetic with native FP32 arithmetic. -/
def clamp64 (z : Int) : Int := min (2^63-1) (max (-(2^63)) z)
def realRound (x : ℝ) : BitVec 64 := BitVec.ofInt 64 (clamp64 (roundEven x))
def realByte (x inverse : ℝ) : BitVec 8 :=
  BitVec.ofInt 8 (clamp64 (roundEven (127 * (x * inverse))))

def realAlgebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with literal := fun _ _ x => x
              binary := fun p d op x y => match op with
                | .mul => x * y
                | _ => rest.binary p d op x y
              fp32Load := id }

theorem clamp64_bounds (z : Int) : -(2^63 : Int) ≤ clamp64 z ∧ clamp64 z < 2^63 := by
  unfold clamp64
  omega

theorem clamp64_word (z : Int) : (BitVec.ofInt 64 (clamp64 z)).toInt = clamp64 z :=
  BitVec.toInt_ofInt_eq_self (by decide) (by simpa using (clamp64_bounds z).1)
    (by simpa using (clamp64_bounds z).2)

theorem real_formula (rest : Algebra ℝ) (x inverse : ℝ) :
    quantized (realAlgebra rest) realRound x inverse = realByte x inverse := by
  change BitVec.ofInt 8 (BitVec.ofInt 64
    (clamp64 (roundEven (127 * (x * inverse))))).toInt = realByte x inverse
  rw [clamp64_word]
  rfl

/-- The normalized mathematical corollary is explicit about the supplied
inverse. The zero-maximum host branch is not assigned a Real reciprocal. -/
theorem normalized (x maximum inverse : ℝ) (_nonzero : maximum ≠ 0)
    (hinverse : inverse = 1 / maximum) :
    realByte x inverse = BitVec.ofInt 8 (clamp64 (roundEven (127 * (x / maximum)))) := by
  subst inverse
  simp [realByte, div_eq_mul_inv]

abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def levels : Word := FP.Value.ofBits .fp32 0x42fe0000

/-! Pinned conversion model for the observed Triton 3.7.1, CUDA sm80 lowering:
`cvt.rni.ftz.s64.f32`, followed by `st.global.b8`. The offline explicit i32
signature evidence is in kernel-evidence/quant_transpose_kernel; it is not a
captured Python JIT binding trace or GPU execution.

Official instruction references:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
The cvt reference specifies nearest/ties-even, signed destination clamping,
FTZ input handling, and MININT for NaN when the destination is signed64.
This last rule differs from conversion to small signed integers. These are
primitive semantics for this profile, not platform-independent libdevice
claims and not numerical experiments. -/
def nativeRound (x : Word) : BitVec 64 :=
  BitVec.ofInt 64 (match x.toRat? with
    | some q => clamp64 (roundEven (if x.isSubnormal then 0 else (q : ℝ)))
    | none => if x.isNaN then -(2^63) else if x.sign then -(2^63) else 2^63-1)

/-- Only the scalar FP32 multiply primitive is a model parameter. It receives
two words, never a tile, Tensor, or expected result. Both source multiplications
use the same primitive and remain ordered exactly as in the program. -/
def nativeAlgebra (multiply : Word → Word → Word) (rest : Algebra Word) : Algebra Word :=
  { rest with
    literal := fun p d x => if FP.Software.fp32Context p d && decide (x = 127)
      then levels else rest.literal p d x
    binary := fun p d op x y => match op with
      | .mul => if FP.Software.fp32Context p d then multiply x y else rest.binary p d op x y
      | _ => rest.binary p d op x y
    fp32Load := id }

def nativeByte (multiply : Word → Word → Word) (x inverse : Word) : BitVec 8 :=
  BitVec.ofInt 8 (nativeRound (multiply levels (multiply x inverse))).toInt

theorem native_formula (multiply : Word → Word → Word) (rest : Algebra Word)
    (x inverse : Word) :
    quantized (nativeAlgebra multiply rest) nativeRound x inverse = nativeByte multiply x inverse := by
  simp [quantized, rounded, nativeAlgebra, nativeByte, FP.Software.fp32Context]

theorem source_literal : levels.toRat? = some 127 := by decide +kernel

theorem nan_conversion (x : Word) (hx : x.isNaN = Bool.true) :
    nativeRound x = BitVec.ofInt 64 (-(2^63)) := by
  have he : x.exponent = 255 ∧ x.fraction ≠ 0 := by simpa [FP.Value.isNaN] using hx
  simp [nativeRound, FP.Value.toRat?, FP.Value.decode, he.1, he.2, hx]

theorem nan_byte (x : Word) (hx : x.isNaN = Bool.true) :
    BitVec.ofInt 8 (nativeRound x).toInt = 0 := by
  rw [nan_conversion x hx]
  decide +kernel

theorem subnormal_conversion (x : Word) (h : x.isSubnormal = Bool.true)
    (q : ℚ) (hq : x.toRat? = some q) : nativeRound x = 0 := by
  simp [nativeRound, hq, h, clamp64]

/-- The admitted zero/positive-infinity input pair reaches a NaN under the
explicit IEEE software multiplication profile, then the pinned signed64
conversion observes byte zero. This is an exact model calculation, not a GPU
measurement or a universal identification of every native multiply callback. -/
theorem zero_infinite_inverse :
    nativeByte (FP.mul .ieeeValue) (FP.Value.zero .fp32) (FP.Value.infinity .fp32) = 0 := by
  apply nan_byte
  decide +kernel

#axiomsClean real_formula
#axiomsClean normalized
#axiomsClean native_formula
#axiomsClean source_literal
#axiomsClean nan_conversion
#axiomsClean nan_byte
#axiomsClean subnormal_conversion
#axiomsClean zero_infinite_inverse
#specNonCircular realByte avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular nativeByte avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelModels
