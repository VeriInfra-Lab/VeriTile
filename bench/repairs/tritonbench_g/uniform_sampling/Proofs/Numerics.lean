import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Host
import bench.tritonbench_g.uniform_sampling.Proofs.WordArithmetic
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Scalar
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Bytes

namespace VeriTile.Bench.Repairs.UniformSampling.Numerics
open VeriTile.Triton
noncomputable section
abbrev Bits := Float32Bits
abbrev Byte := BitVec 8
abbrev Store := Nat→Byte
abbrev api := VeriTile.Bench.Repairs.UniformSampling.Shared.Bytes.api
abbrev project32 := VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar.project32
abbrev project64 := VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar.project64
abbrev negate := VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar.negate

/-- Actual local FP32 primitive semantics. There is no random-word oracle:
Philox words and sign-fold integers are fixed by the checked integer source.
The decoder retains its int→FP32 conversion and separate scale multiplication. -/
structure Primitives where
  extract : Float64Bits→Bits
  fromNat32 : Nat→Bits
  multiply32 : Bits→Bits→Bits
  subtract32 : Bits→Bits→Bits
  fused32 : Bits→Bits→Bits→Bits

def model (p : Primitives) : Arithmetic Bits where
  extract:=p.extract
  uniform32:=fun word=>p.multiply32 (p.fromNat32 (Random.foldSign word)) ⟨0x2fffffff⟩
  sub32:=p.subtract32
  fma32:=p.fused32
  neg32:=negate

theorem negate_twice (p : Primitives) (x : Bits) : (model p).neg32 ((model p).neg32 x)=x :=
  VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar.negate_twice x

/-- A software IEEE model is available without introducing hardware axioms. -/
def software (extract : Float64Bits→Bits) : Primitives where
  extract:=extract
  fromNat32:=fun n=>⟨(FP.round .fp32 FP.Config.ieeeValue (n : Rat)).bits⟩
  multiply32:=fun x y=>⟨(FP.mul (f:=.fp32) FP.Config.ieeeValue ⟨x.bits⟩ ⟨y.bits⟩).bits⟩
  subtract32:=fun x y=>⟨(FP.sub (f:=.fp32) FP.Config.ieeeValue ⟨x.bits⟩ ⟨y.bits⟩).bits⟩
  fused32:=fun x y z=>⟨(FP.fma (f:=.fp32) FP.Config.ieeeValue ⟨x.bits⟩ ⟨y.bits⟩ ⟨z.bits⟩).bits⟩

/-- The compatibility helper's signed32 negate/subtract, including INT32_MIN,
implements the fixed integer sign-fold exactly; this is not a numerical port. -/
theorem helper_fold (word : Random.Word) :
    (if word.toInt<0 then -word-1 else word).toNat=Random.foldSign word :=
  VeriTile.Bench.TritonBenchG.UniformSampling.WordArithmetic.fallback_machine_fold word

#axiomsClean helper_fold

/-- Exact extraction applies only where the FP64 payload is exactly represented
by the runtime FP32 argument. General conversion rounds, underflows or overflows. -/
def Extraction (p : Primitives) (host : Float64Bits) (value : ℝ) : Prop :=
  project64 host=some value ∧ project32 (p.extract host)=some value

def DecoderConformance (p : Primitives) (word : Random.Word) : Prop :=
  (model p).uniform32 word=⟨(Random.uniform word).bits⟩

/-- This exactness port is restricted to actual operands satisfying it. It is
not claimed for all words: the native two-rounding scaled decoder is generally
not the ideal exact integer quantile `foldSign/2^31`. -/
def DecoderExact (p : Primitives) (word : Random.Word) : Prop :=
  project32 ((model p).uniform32 word)=some (Spec.unit word)

def SubtractExact (p : Primitives) (upper lower : Bits) (u l : ℝ) : Prop :=
  project32 upper=some u ∧ project32 lower=some l ∧
    project32 (p.subtract32 upper lower)=some (u-l)

def FmaExact (p : Primitives) (x y z : Bits) (a b c : ℝ) : Prop :=
  project32 x=some a ∧ project32 y=some b ∧ project32 z=some c ∧
    project32 (p.fused32 x y z)=some (a*b+c)

def DecoderError (p : Primitives) (word : Random.Word) (tolerance : ℝ) : Prop :=
  ∃ u,project32 ((model p).uniform32 word)=some u ∧ |u-Spec.unit word|≤tolerance

/-- Each conjunct concerns one local evaluated primitive, with its actual
FP32 operands. No counter, layout, memory, launch, success or output condition
occurs among these numerical ports. -/
structure LocalExact (p : Primitives) (scalars : Scalars) (word : Random.Word) (lower upper : ℝ) : Prop where
  lowerExtract : Extraction p scalars.lower lower
  upperExtract : Extraction p scalars.upper upper
  decoder : DecoderExact p word
  width : SubtractExact p (p.extract scalars.upper) (p.extract scalars.lower) upper lower
  affine : FmaExact p ((model p).uniform32 word)
    (p.subtract32 (p.extract scalars.upper) (p.extract scalars.lower)) (p.extract scalars.lower)
    (Spec.unit word) (upper-lower) lower

theorem local_correspondence (p : Primitives) (scalars : Scalars) (word : Random.Word)
    (lower upper : ℝ) (h:LocalExact p scalars word lower upper) :
    project32 (affine (model p) scalars word)=some (lower+(upper-lower)*Spec.unit word) := by
  have he:=h.affine.2.2.2
  change project32 (p.fused32 ((model p).uniform32 word)
    (p.subtract32 (p.extract scalars.upper) (p.extract scalars.lower)) (p.extract scalars.lower))=_
  rw [he]
  congr 1
  ring

#axiomsClean negate_twice
#axiomsClean local_correspondence
#guard_msgs (drop info) in
#specNonCircular LocalExact avoiding [Execution.run,Host.entry,Native.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.Numerics
