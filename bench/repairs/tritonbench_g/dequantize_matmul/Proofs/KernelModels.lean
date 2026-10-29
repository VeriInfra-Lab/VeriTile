import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelGrid
import VeriTile.Torch.FP32
import VeriTile.Triton.Float.ScalarBackend

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelModels
open VeriTile.Triton FP.Structural KernelSource KernelResult
noncomputable section

abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Mathematical interpretation of the same typed source. The signed INT8
integer is embedded in Real before multiplication; Real is not a native dtype. -/
def realAlgebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    fromInt := fun _ z => (z : ℝ)
    binary := fun p d op x y => match op with
      | .mul => x * y
      | _ => rest.binary p d op x y
    fp32Load := id }

def mathematical (q : BitVec 8) (scale : ℝ) : ℝ := (q.toInt : ℝ) * scale

theorem real_formula (rest : Algebra ℝ) (q : BitVec 8) (scale : ℝ) :
    dequantized (realAlgebra rest) q.toInt scale = mathematical q scale := rfl

/-- Signed INT8 to FP32 conversion, followed by one native scalar multiply.
All signed INT8 values have exact FP32 representations. The conversion and
multiply are primitive semantics, not numerical experiment assumptions.
The offline Triton 3.7.1/sm80 profile emits ld.global.b8, cvt.s16.s8,
cvt.rn.f32.s16 and mul.f32; the final st.global.b32 preserves the FP32 word.
Official instruction semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
The probe used an explicit runtime-i32 signature, not a captured JIT binding
or GPU execution. No native-vs-Real equality or universal multiply mode is asserted. -/
def signedWord (q : BitVec 8) : Word := FP.round .fp32 .ieeeValue (Rat.ofInt q.toInt)

def nativeAlgebra (multiply : Word → Word → Word) (rest : Algebra Word) : Algebra Word :=
  { rest with
    fromInt := fun p z => if p = some .fp32
      then FP.round .fp32 .ieeeValue (Rat.ofInt z) else rest.fromInt p z
    binary := fun p d op x y => match op with
      | .mul => if FP.Software.fp32Context p d then multiply x y else rest.binary p d op x y
      | _ => rest.binary p d op x y
    fp32Load := id }

def nativeWord (multiply : Word → Word → Word) (q : BitVec 8) (scale : Word) : Word :=
  multiply (signedWord q) scale

theorem native_formula (multiply : Word → Word → Word) (rest : Algebra Word)
    (q : BitVec 8) (scale : Word) :
    dequantized (nativeAlgebra multiply rest) q.toInt scale = nativeWord multiply q scale := by
  simp [dequantized, nativeAlgebra, nativeWord, signedWord, FP.Software.fp32Context]

/-- Ordinary initial dtype decoding only; there is no expected-output premise. -/
def Reads {α : Type} (p : Parameters) (q : Nat → BitVec 8) (scale : Nat → α) (s : State α) : Prop :=
  (∀ i, i < p.K * p.N → s.mem p.b i = .mk .int (q i).toInt) ∧
  (∀ i, i < p.N → s.mem p.scale i = .mk .real (scale i))

def Result {α : Type} (p : Parameters) (expected : Nat → α) (initial final : State α) : Prop :=
  (∀ i, i < p.K * p.N → final.mem p.output i = .mk .real (expected i)) ∧
  (∀ r i, r ≠ p.output ∨ p.K * p.N ≤ i → final.mem r i = initial.mem r i)

theorem whole_correct {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p)
    (q : Nat → BitVec 8) (scale : Nat → α) (s : State α) (reads : Reads p q scale s) :
    ∃ final, KernelGrid.run M p Bool.false (KernelGrid.order p) s = some final ∧
      Result p (fun i => dequantized M (q i).toInt (scale (i % p.N))) s final := by
  obtain ⟨final, executed, values, frame⟩ := KernelGrid.correctness M p hK hN sep s
  refine ⟨final, executed, ⟨?_, frame⟩⟩
  intro i hi
  have hn : 0 < p.N := by nlinarith [Nat.zero_le p.K]
  rw [values i hi]
  simp [KernelGrid.reference, reads.1 i hi, reads.2 _ (Nat.mod_lt _ hn)]

#axiomsClean real_formula
#axiomsClean native_formula
#axiomsClean whole_correct
#specNonCircular mathematical avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular nativeWord avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular Reads avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular Result avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelModels
