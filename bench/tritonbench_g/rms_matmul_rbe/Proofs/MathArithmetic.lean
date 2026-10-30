import bench.tritonbench_g.rms_matmul_rbe.Proofs.Bound
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Real

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

/-- Finite byte decoding for the ideal interpretation. The default is only
the totalization outside finite E5 values; no native nonfinite-to-Real
correspondence is asserted. Native words use their separate bit semantics. -/
def e5Real (bits : Float8E5Bits) : ℝ := ((bits.decodeRat).getD 0 : ℚ)

def mathArithmetic : Arithmetic ℝ where
  toScalar := realOps
  literal := fun _ x => x
  cast := fun _ _ x => x
  e5Bits := fun _ => e5Real
  fp32Bits := fun bits => ((Float32Bits.decodeRat bits).getD 0 : ℚ)
  plan := RmsMatmulRbe.orderedK8
  reduction := RmsMatmulRbe.butterfly64
  natLiteral := fun _ => rfl

theorem math_zeroLaws : ZeroLaws mathArithmetic := by
  constructor
  · rfl
  · simp [kernelCast,mathArithmetic,realOps,RmsMatmulRbe.realOps]
  · norm_num [kernelCast,mathArithmetic,realOps,RmsMatmulRbe.realOps,e5Real,
      Float8E5Bits.ofInt,Float8E5Bits.decodeRat,Float8E5Bits.exponent,
      Float8E5Bits.fraction,Float8E5Bits.sign]
  · norm_num [kernelCast,mathArithmetic,realOps,RmsMatmulRbe.realOps,e5Real,
      Float8E5Bits.ofInt,Float8E5Bits.decodeRat,Float8E5Bits.exponent,
      Float8E5Bits.fraction,Float8E5Bits.sign]

def mathMaterialization : ScalarMaterialization ℝ where
  boolean := fun b => if b then 1 else 0
  integer := Int.cast
  floating := id

theorem math_scalar (s : HostScalar) : mathMaterialization.apply s=s.value := by
  cases s <;> rfl

def mathematicalTile (fp8 : Bool) (p : Arguments) (s : State ℝ)
    (epsilon theta : HostScalar) : Values ℝ .real [16,64] := fun i =>
  realTarget (parameters p epsilon.value theta.value)
    (rawRow mathArithmetic fp8 p s (s.pids 0) (programRow p s*16+i.1.val))
    p.rotary (programRow p s*16+i.1.val) (programCol p s*64+i.2.1.val)

theorem bound_mathematical (plan : RotaryPlan) (fp8 : Bool) (p : Arguments)
    (epsilon theta : HostScalar) (s : State ℝ) :
    boundTarget mathArithmetic mathMaterialization Bool.false plan fp8 p epsilon theta s=
      mathematicalTile fp8 p s epsilon theta := by
  funext i
  simp only [boundTarget,if_false,rawTile,rawValue,math_scalar,mathematicalTile]
  exact reference_real _ _ _ _ _ _ _

/-- The mathematical theorem runs the quoted source under exact Real
arithmetic, from the same host scalar values. It is separate from native
word equality and retains the typed destination and complete cell frame. -/
theorem mathematical_program (fp8 outputFP8 : Bool) (p : Arguments)
    (epsilon theta : HostScalar) (s : State ℝ) (bytes : Int8Memory s) (even : p.N%2=0) :
    ∃ final,execution mathArithmetic mathMaterialization Bool.false ⟨Bool.false,Bool.false⟩
        fp8 outputFP8 p epsilon theta s=some final ∧
      final.mem=(tileWrite mathArithmetic outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (mathematicalTile fp8 p s epsilon theta) s).mem := by
  obtain ⟨final,hx,hm⟩ := bound_complete mathArithmetic math_zeroLaws mathMaterialization Bool.false
    ⟨Bool.false,Bool.false⟩ fp8 outputFP8 p epsilon theta s bytes even
  exact ⟨final,hx,by simpa only [bound_mathematical] using hm⟩

#axiomsClean math_zeroLaws
#axiomsClean math_scalar
#axiomsClean bound_mathematical
#axiomsClean mathematical_program
#specNonCircular mathematicalTile avoiding [execution,boundTarget,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
