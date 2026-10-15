import bench.tritonbench_g.matrix_transpose.HostCorrect
import Init.Data.Nat.Power2

/-! Exact source admission and host composition under the ordinary trusted
Python/Torch/Triton primitive boundary. There is no tensor arithmetic callback:
randn supplies arbitrary fp16 words, zeros creates fresh contiguous storage,
and the existing full kernel interpreter proves all copied words and frames.

The frontend checks below are bound by a separately frozen source audit to
Triton 3.7.1's actual arange/block-shape validators. Compilation, allocation and
launch may fail as ordinary primitive outcomes; no success of every device
backend is asserted. The entire frontend-success domain is retained. -/
namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.FullScope
open VeriTile.Triton FP.Structural FPBits Host
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

inductive RangeError where
  | indexWidth | nonPositive | nonPowerOfTwo | tensorSize
  deriving DecidableEq, Repr

/-- Original `tl.arange(0,n)` validation order for natural shape arguments. -/
def arangeCheck (n : Nat) : Option RangeError :=
  if 2^32 ≤ n then some .indexWidth
  else if n = 0 then some .nonPositive
  else if n &&& (n-1) ≠ 0 then some .nonPowerOfTwo
  else if 2^20 < n then some .tensorSize
  else none

inductive AdmissionError where
  | sizeMRange (reason : RangeError)
  | dHeadRange (reason : RangeError)
  | matrixPointerTensorSize
  deriving DecidableEq, Repr

/-- Both aranges precede the first outer-broadcast pointer expression, whose
block shape is [m,d]. Out's [d,m] shape has the same element count. -/
def admission (m d : Nat) : Option AdmissionError :=
  match arangeCheck m with
  | some why => some (.sizeMRange why)
  | none => match arangeCheck d with
    | some why => some (.dHeadRange why)
    | none => if 2^20 < m*d then some .matrixPointerTensorSize else none

/-- Independent complete successful shape domain, rather than selected tests. -/
def SourceShape (m d : Nat) : Prop :=
  ∃ a b : Nat, m = 2^a ∧ d = 2^b ∧ a+b ≤ 20

theorem arangeCheck_none (n : Nat) :
    arangeCheck n = none ↔ 0 < n ∧ n &&& (n-1) = 0 ∧ n ≤ 2^20 := by
  unfold arangeCheck
  split_ifs <;> simp_all <;> omega

theorem admission_none (m d : Nat) :
    admission m d = none ↔ SourceShape m d := by
  have characterized : admission m d = none ↔
      arangeCheck m = none ∧ arangeCheck d = none ∧ m*d ≤ 2^20 := by
    unfold admission
    cases hm : arangeCheck m <;> cases hd : arangeCheck d <;> simp_all
  rw [characterized, arangeCheck_none, arangeCheck_none]
  constructor
  · rintro ⟨⟨hm, hmPower, _⟩, ⟨hd, hdPower, _⟩, hprod⟩
    obtain ⟨a, ha⟩ := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 hmPower
    obtain ⟨b, hb⟩ := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).1 hdPower
    refine ⟨a,b,ha,hb,?_⟩
    rw [ha,hb,← Nat.pow_add] at hprod
    exact (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).1 hprod
  · rintro ⟨a,b,rfl,rfl,hab⟩
    have hp : 2^a * 2^b ≤ 2^20 := by
      rw [← Nat.pow_add]
      exact Nat.pow_le_pow_right (by decide) hab
    have ha : 0 < 2^a := Nat.two_pow_pos a
    have hb : 0 < 2^b := Nat.two_pow_pos b
    have ma : 2^a ≤ 2^20 := (Nat.le_mul_of_pos_right _ hb).trans hp
    have mb : 2^b ≤ 2^20 := by
      have := (Nat.le_mul_of_pos_right (2^b) ha)
      rw [Nat.mul_comm] at this
      exact this.trans hp
    exact ⟨⟨ha, (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).2 ⟨a,rfl⟩, ma⟩,
      ⟨hb, (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by omega)).2 ⟨b,rfl⟩, mb⟩, hp⟩

/-- Finite complete exponent inventory used by the independent frontend run. -/
def shapeCases : List (Nat × Nat) :=
  (List.range 21).flatMap fun a =>
    ((List.range 21).filter fun b => a+b ≤ 20).map fun b => (2^a,2^b)

theorem shapeCases_complete (m d : Nat) :
    (m,d) ∈ shapeCases ↔ SourceShape m d := by
  simp only [shapeCases, List.mem_flatMap, List.mem_range, List.mem_map,
    List.mem_filter, decide_eq_true_eq, Prod.mk.injEq]
  constructor
  · rintro ⟨a,_,b,⟨_,bound⟩,rfl,rfl⟩
    exact ⟨a,b,rfl,rfl,bound⟩
  · rintro ⟨a,b,rfl,rfl,bound⟩
    exact ⟨a,by omega,b,⟨by omega,bound⟩,rfl,rfl⟩

theorem shapeCases_length : shapeCases.length = 231 := by decide

theorem admitted_offset_and_byte_bounds (h : Host.Input)
    (admitted : admission h.m h.d = none) (i : TileIndex [h.d,h.m]) :
    inputOffset (parameters h) i < 2^20 ∧
    outputOffset (parameters h) i < 2^20 ∧
    2 * inputOffset (parameters h) i + 1 < 2^21 ∧
    2 * outputOffset (parameters h) i + 1 < 2^21 := by
  obtain ⟨a,b,ha,hb,hab⟩ := (admission_none _ _).1 admitted
  have bound : count h ≤ 2^20 := by
    simp only [count,ha,hb,← Nat.pow_add]
    exact Nat.pow_le_pow_right (by decide) hab
  have hin := (input_in_bounds h i).trans_le bound
  have hout := (output_in_bounds h i).trans_le bound
  omega

/-- Exact metadata supplied by the trusted Torch allocation contracts. -/
structure TensorMetadata where
  region : RegionName
  shape : List Nat
  strides : List Nat
  storage : FloatDType
  cuda : Bool

def returnedTensor (h : Host.Input) : TensorMetadata :=
  ⟨h.output, outputShape h, outputStrides h, .fp16, Bool.true⟩

inductive Event where
  | randn | zeros | launch | returned
  deriving DecidableEq, Repr

/-- Ordinary primitive errors are observable at their source call site. Their
messages and allocator/runtime implementations belong to the trusted API. -/
inductive Error where
  | randn | zeros | frontend (reason : AdmissionError) | launch | backend
  | interpreterFailure
  deriving DecidableEq, Repr

structure PrimitiveFailures where
  randn : Bool
  zeros : Bool
  launch : Bool
  backend : Bool

inductive Outcome where
  | error (reason : Error) (trace : List Event)
  | returned (tensor : TensorMetadata) (final : State Word) (trace : List Event)

/-- Source order: randn, zeros, one kernel[1] call, return its exact Out tensor.
An API-level launch error can occur before frontend entry. If launch setup
succeeds, the source-order frontend checks and actual interpreter are run. -/
noncomputable def wrapper (h : Host.Input) (initial : State Word)
    (samples : Nat → Word) (failure : PrimitiveFailures) : Outcome :=
  if failure.randn then .error .randn [.randn]
  else if failure.zeros then .error .zeros [.randn,.zeros]
  else if failure.launch then .error .launch [.randn,.zeros,.launch]
  else match admission h.m h.d with
    | some why => .error (.frontend why) [.randn,.zeros,.launch]
    | none => if failure.backend then .error .backend [.randn,.zeros,.launch]
      else match Host.run h initial samples with
      | none => .error .interpreterFailure [.randn,.zeros,.launch]
      | some final => .returned (returnedTensor h) final [.randn,.zeros,.launch,.returned]

/-- Independent result relation includes all primitive errors and all original
frontend rejections. In the successful route it specifies values, allocation
metadata, return identity, input preservation, and the entire memory frame. -/
def WrapperPost (h : Host.Input) (initial : State Word) (samples : Nat → Word)
    (failure : PrimitiveFailures) (outcome : Outcome) : Prop :=
  if failure.randn then outcome = .error .randn [.randn]
  else if failure.zeros then outcome = .error .zeros [.randn,.zeros]
  else if failure.launch then outcome = .error .launch [.randn,.zeros,.launch]
  else match admission h.m h.d with
    | some why => outcome = .error (.frontend why) [.randn,.zeros,.launch]
    | none => if failure.backend then outcome = .error .backend [.randn,.zeros,.launch]
      else ∃ final, outcome = .returned (returnedTensor h) final [.randn,.zeros,.launch,.returned] ∧
        final.mem = expectedMemory h initial samples ∧
        (inputShape h).prod = (outputShape h).prod ∧
        (∀ r : Fin h.m, ∀ c : Fin h.d,
          final.mem h.output (c.val*h.m+r.val) = .mk .fp16 (samples (r.val*h.d+c.val))) ∧
        (∀ o, o < count h → final.mem h.input o = .mk .fp16 (samples o)) ∧
        (∀ region o, (region ≠ h.input ∨ count h ≤ o) →
          (region ≠ h.output ∨ count h ≤ o) → final.mem region o = initial.mem region o)

specification matrix_transpose_full_source_outcomes
    (h : Host.Input) (initial : State Word) (samples : Nat → Word)
    (failure : PrimitiveFailures) (fresh : h.input ≠ h.output) :
    WrapperPost h initial samples failure (wrapper h initial samples failure) := by
  unfold WrapperPost wrapper
  split <;> rename_i hr
  · rfl
  · split <;> rename_i hz
    · rfl
    · split <;> rename_i hl
      · rfl
      · cases hadmit : admission h.m h.d with
        | some why => rfl
        | none =>
          cases hb : failure.backend
          · simp only [Bool.false_eq_true, ↓reduceIte]
            obtain ⟨final,hrun,hmem,hshape,hvalues,hinput,hframe⟩ :=
              matrix_transpose_host_correctness h initial samples fresh
            simp only [hrun]
            exact ⟨final,rfl,hmem,hshape,hvalues,hinput,hframe⟩
          · rfl

#axiomsClean arangeCheck_none
#axiomsClean admission_none
#axiomsClean shapeCases_complete
#axiomsClean shapeCases_length
#axiomsClean admitted_offset_and_byte_bounds
#axiomsClean matrix_transpose_full_source_outcomes
#specNonCircular WrapperPost avoiding [wrapper, Host.run, FP.Structural.exec, FPBits.execution]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose.FullScope
