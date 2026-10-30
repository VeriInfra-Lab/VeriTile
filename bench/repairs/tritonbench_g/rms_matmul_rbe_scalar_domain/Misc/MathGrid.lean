import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.Tensor
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.MathArithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel.Ideal
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 9000000
set_option linter.unusedSimpArgs false

/-- Arbitrary Real typed storage indexed by the caller's physical addresses,
plus the raw int8 input channel needed by E5 bitcasts. This ideal storage is
not an encoding of Real values into finite native bytes. -/
structure Store where
  scalar : Nat → ℝ
  raw : Nat → BitVec 8

def write (s : Store) (address : Nat) (value : ℝ) : Store :=
  {s with scalar:=Function.update s.scalar address value}

theorem apart_ne (fp8 : Bool) (a b width : Nat) (positive : 0<width)
    (apart : Apart a (outputWidth fp8) b width) : b≠a := by
  have wo : 0<outputWidth fp8 := by cases fp8 <;> decide
  unfold Apart at apart
  omega

def access : Access ℝ Store (ℝ × BitVec 8) where
  read32 := fun s a => s.scalar a
  read16 := fun s a => s.scalar a
  readE5 := fun s a => s.scalar a
  read8 := fun s a => s.raw a
  byte := fun s a => (s.scalar a,s.raw a)
  write := fun _ => write
  storable := fun _ _ => True
  write16_same := by intros; simp [write]
  writeE5_same := by intros; simp [write]
  read32_apart := by intro fp8 s a b value h; simp [write,apart_ne fp8 a b 4 (by decide) h]
  read16_apart := by intro fp8 s a b value h; simp [write,apart_ne fp8 a b 2 (by decide) h]
  readE5_apart := by intro fp8 s a b value h; simp [write,apart_ne fp8 a b 1 (by decide) h]
  read8_apart := by intros; rfl
  byte_apart := by
    intro fp8 s a b value h
    have wo : 0<outputWidth fp8 := by cases fp8 <;> decide
    have ne : b≠a := by omega
    simp [write,ne]

theorem output_storable : OutputStorable access mathArithmetic := fun _ _ => trivial

def target (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) : ℝ :=
  let own := owner i c
  mathematicalTile (useFP8 i) (qkvArgument (arguments i) c.1)
    (state access 0 i q initial own.id) epsilon theta own.lane

theorem event_mathematical (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) :
    (event access mathArithmetic mathMaterialization Bool.false i q initial epsilon theta (owner i c)).value=
      target i q initial epsilon theta c := by
  have formula := congrFun (bound_mathematical (selectedRotaryPlan c.1) (useFP8 i)
    (qkvArgument (arguments i) c.1) epsilon theta (state access 0 i q initial (owner i c).id)) (owner i c).lane
  have cast (fp8 : Bool) (x : ℝ) : castOutput mathArithmetic fp8 x=x := by cases fp8 <;> rfl
  simpa only [event,cast,target,calleeTarget,boundTarget,if_false,math_scalar] using formula

/-- Full readable-source grid execution in arbitrary-Real typed storage,
all QKV mathematical outputs and the complete typed storage frame. -/
theorem whole_tensor (i : Contract.Inputs) (checked : Contract.checkInputs i=.ok ())
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (initial : Store) (epsilon theta : HostScalar) (observed : ScalarObservations i epsilon theta) :
    (epsilonType epsilon=some .float32 ∧ thetaType theta=some .float32) ∧
    ∃ final,grid access mathArithmetic mathMaterialization Bool.false i q initial epsilon theta=some final ∧
      (∀ c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original),
        final.scalar (cellAddress i q c)=target i q initial epsilon theta c) ∧
      (∀ j,¬WrittenByte i q j →
        (final.scalar j,final.raw j)=(initial.scalar j,initial.raw j)) ∧
      (∀ t∈i.original.reads,∀ j,t.base≤j → j<t.base+t.bytes →
        (final.scalar j,final.raw j)=(initial.scalar j,initial.raw j)) := by
  have contract := (Contract.checkInputs_iff i).mp checked
  refine ⟨checked_scalar_types i contract epsilon theta observed,?_⟩
  obtain ⟨final,success,readback,frame⟩ := grid_results access mathArithmetic math_zeroLaws mathMaterialization
    Bool.false i contract ordinary q hq initial epsilon theta output_storable
  refine ⟨final,success,?_,frame,?_⟩
  · intro c
    have result := readback (owner i c) (owner_valid i c)
    rw [owner_address,event_mathematical] at result
    cases hf : qkvOutputFP8 (useFP8 i) (owner i c).phase <;>
      simpa only [Access.read,hf,access,if_true,if_false] using result
  · intro t ht j lo hi
    exact frame j (input_backing_unwritten i contract ordinary q hq t ht j ⟨lo,hi⟩)

#axiomsClean access
#axiomsClean event_mathematical
#axiomsClean whole_tensor
#specNonCircular target avoiding [grid,gridAux,program,programTrace,storedEvents,event]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel.Ideal
