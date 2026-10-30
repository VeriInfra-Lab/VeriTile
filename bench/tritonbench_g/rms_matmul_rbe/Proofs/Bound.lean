import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeSourceReference
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Scalars
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Selected

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

/-- The same Q/K choices occur in the retained successful HEAD_DIM=4 scalar
dispatch controls and HEAD_DIM=64 signature/specialization controls. V does
not execute rotary. These are finite syntax choices, not FP equivalences. -/
def selectedRotaryPlan (phase : Fin 3) : RotaryPlan :=
  if phase.val=1 then ⟨Bool.true,Bool.true⟩ else ⟨Bool.false,Bool.false⟩

def selectedArithmetic (a : Arithmetic α) : Arithmetic α :=
  {a with plan:=RmsMatmulRbe.orderedK8,reduction:=RmsMatmulRbe.butterfly64}

def execution {α : Type} [Inhabited α] (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (epsilon theta : HostScalar) (s : State α) :=
  run a.model (if native then selectedSource plan fp8 outputFP8 p
    else (source fp8 outputFP8 p).surfaceBody)
    (bindScalars (materializedScalars materialize epsilon theta) s)

def boundTarget {α : Type} [Inhabited α] (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (plan : RotaryPlan)
    (fp8 : Bool) (p : Arguments) (epsilon theta : HostScalar) (s : State α) :=
  if native then nativeRawTile a plan fp8 p s (materialize.apply epsilon) (materialize.apply theta)
  else rawTile a fp8 p s (materialize.apply epsilon) (materialize.apply theta)

/-- The host scalar objects feed both quoted execution registers and the
independent target. In particular EPS integer and finite-float materialization
stay distinct; zero/infinity after FP32 promotion are not excluded. -/
theorem bound_complete {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (epsilon theta : HostScalar) (s : State α)
    (bytes : Int8Memory s) (even : p.N%2=0) :
    ∃ final,execution a materialize native plan fp8 outputFP8 p epsilon theta s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (boundTarget a materialize native plan fp8 p epsilon theta s) s).mem := by
  let start := bindScalars (materializedScalars materialize epsilon theta) s
  have scalars := scalar_registers materialize epsilon theta s
  have ib : Int8Memory start := bytes
  cases hn : native with
  | «false» =>
    obtain ⟨final,hx,hm⟩ := source_reference a z fp8 outputFP8 p start
      (materialize.apply epsilon) (materialize.apply theta) scalars.1 scalars.2 ib even
    refine ⟨final,hx,hm.trans ?_⟩
    exact tileWrite_memory a outputFP8 p _ _ _ _ start s rfl
  | «true» =>
    obtain ⟨final,hx,hm⟩ := native_reference a z plan fp8 outputFP8 p start
      (materialize.apply epsilon) (materialize.apply theta) scalars.1 scalars.2 ib even
    refine ⟨final,hx,hm.trans ?_⟩
    exact tileWrite_memory a outputFP8 p _ _ _ _ start s rfl

theorem checked_bound_complete {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (i : Contract.Inputs)
    (checked : Contract.checkInputs i=.ok ()) (epsilon theta : HostScalar)
    (observations : ScalarObservations i epsilon theta) (s : State α)
    (bytes : Int8Memory s) (even : p.N%2=0) :
    (epsilonType epsilon=some .float32 ∧ thetaType theta=some .float32) ∧
    ∃ final,execution a materialize native plan fp8 outputFP8 p epsilon theta s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (boundTarget a materialize native plan fp8 p epsilon theta s) s).mem :=
  ⟨checked_scalar_types i ((Contract.checkInputs_iff i).mp checked) epsilon theta observations,
    bound_complete a z materialize native plan fp8 outputFP8 p epsilon theta s bytes even⟩

#axiomsClean bound_complete
#axiomsClean checked_bound_complete
#specNonCircular boundTarget avoiding [execution,sourceTileValues,nativeTileValues,run]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
