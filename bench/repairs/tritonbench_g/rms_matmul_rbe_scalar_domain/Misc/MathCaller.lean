import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.Tensor
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.MathArithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 9000000
set_option linter.unusedSimpArgs false

/-- Partial ordinary binary16 decoding. Infinity/NaN have no Real value. -/
def halfRat (bits : BitVec 16) : Option ℚ :=
  let exponent := bits.toNat/1024%32
  let fraction := bits.toNat%1024
  if exponent=31 then none else
    let magnitude : ℚ := if exponent=0 then (fraction:ℚ)/2^24
      else if exponent<25 then (1024+fraction:ℚ)/2^(25-exponent)
      else (1024+fraction:ℚ)*2^(exponent-25)
    some (if 32768≤bits.toNat then -magnitude else magnitude)

def decodedCell : Cell Native.Word → Cell ℝ
  | .mk .real x => .mk .real (((Float32Bits.decodeRat ⟨x⟩).getD 0 : ℚ):ℝ)
  | .mk .fp16 x => .mk .fp16 (((halfRat (Native.unpack16 x)).getD 0 : ℚ):ℝ)
  | .mk .f8e5 x => .mk .f8e5 (e5Real ⟨(Native.unpack8 x).toNat,(Native.unpack8 x).isLt⟩)
  | .mk .int x => .mk .int x
  | .mk .nat x => .mk .nat x
  | .mk .bool x => .mk .bool x
  | .mk .ptr x => .mk .ptr x
  | .mk .blockPtr x => .mk .blockPtr x
  | .mk .fp32 _ => .mk .fp32 0
  | .mk .bf16 _ => .mk .bf16 0
  | .mk .f8e4 _ => .mk .f8e4 0

theorem decodedCell_int (c : Cell Native.Word) : (decodedCell c).read .int=c.read .int := by
  cases c with
  | mk d v => cases d <;> simp [decodedCell,Cell.read,defaultValue]

/-- Ideal input interpretation uses the same caller byte store, Tensor
offsets and aliasing. Partial decoders are totalized by zero only outside
finite values; this is never a native nonfinite-to-Real equality claim. -/
def mathematicalState (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (initial : PhysicalKernel.Bytes.Store) (id : PhysicalKernel.Program) : State ℝ where
  mem := fun r j => decodedCell ((PhysicalKernel.state PhysicalKernel.Bytes.access 0 i q initial id).mem r j)
  regs := fun _ _ _ => none
  pids := id.pids
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem mathematicalState_int8 (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (initial : PhysicalKernel.Bytes.Store) (id : PhysicalKernel.Program) :
    Int8Memory (mathematicalState i q initial id) := by
  intro r j
  simpa only [mathematicalState,decodedCell_int] using
    PhysicalKernel.state_int8 PhysicalKernel.Bytes.access 0 i q initial id r j

theorem callee_mathematical (plan : RotaryPlan) (fp8 : Bool) (p : Arguments)
    (epsilon theta : HostScalar) (s : State ℝ) :
    calleeTarget mathArithmetic Bool.false plan fp8 p s epsilon.value theta.value=
      mathematicalTile fp8 p s epsilon theta := by
  have h := bound_mathematical plan fp8 p epsilon theta s
  simpa only [boundTarget,calleeTarget,if_false,math_scalar] using h

def mathematicalQKV (i : Contract.Inputs) (epsilon theta : HostScalar) (s : State ℝ) : State ℝ :=
  let tile (phase : Fin 3) (current : State ℝ) :=
    let p := qkvArgument (PhysicalKernel.arguments i) phase
    tileWrite mathArithmetic (qkvOutputFP8 (PhysicalKernel.useFP8 i) phase) p
      (s.pids 0) (programRow p s) (programCol p s)
      (mathematicalTile (PhysicalKernel.useFP8 i) p s epsilon theta) current
  tile 2 (tile 1 (tile 0 s))

/-- Complete readable-source Q→K→V execution under exact Real arithmetic,
from the actual caller-byte interpretation, has the independent RMS/rotary
formula as its final memory. No native-word/Real equivalence is assumed. -/
theorem mathematical_caller (i : Contract.Inputs) (checked : Contract.checkInputs i=.ok ())
    (q : RmsMatmulRbe.Contract.Tensor) (initial : PhysicalKernel.Bytes.Store) (id : PhysicalKernel.Program)
    (epsilon theta : HostScalar) (observed : ScalarObservations i epsilon theta) :
    (epsilonType epsilon=some .float32 ∧ thetaType theta=some .float32) ∧
    ∃ final,run mathArithmetic.model (qkvCode Bool.false (PhysicalKernel.useFP8 i) (PhysicalKernel.arguments i))
      (bindScalars (materializedScalars mathMaterialization epsilon theta) (mathematicalState i q initial id))=some final ∧
      final.mem=(mathematicalQKV i epsilon theta (mathematicalState i q initial id)).mem := by
  have contract := (Contract.checkInputs_iff i).mp checked
  refine ⟨checked_scalar_types i contract epsilon theta observed,?_⟩
  let start := bindScalars (materializedScalars mathMaterialization epsilon theta) (mathematicalState i q initial id)
  have scalars := scalar_registers mathMaterialization epsilon theta (mathematicalState i q initial id)
  obtain ⟨final,success,memory,_⟩ := qkv_complete mathArithmetic math_zeroLaws Bool.false
    (PhysicalKernel.useFP8 i) (PhysicalKernel.arguments i) start
    (mathMaterialization.apply epsilon) (mathMaterialization.apply theta) scalars.1 scalars.2
    (mathematicalState_int8 i q initial id) (PhysicalKernel.arguments_even i contract) (PhysicalKernel.arguments_apart i)
  refine ⟨final,success,memory.trans ?_⟩
  simp only [math_scalar,qkvWrite,calleeWrite,callee_mathematical]
  rfl

#axiomsClean decodedCell_int
#axiomsClean mathematical_caller
#specNonCircular mathematicalQKV avoiding [run,qkvCode,qkvWrite,sourceTileValues,nativeTileValues]
#specNonCircular mathematicalState avoiding [run,qkvCode,qkvWrite,mathematicalQKV]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
