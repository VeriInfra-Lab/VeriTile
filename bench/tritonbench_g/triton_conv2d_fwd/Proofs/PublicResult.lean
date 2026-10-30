import bench.tritonbench_g.triton_conv2d_fwd.Proofs.PublicSafety
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalReference
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural KernelNumeric
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1500000
variable {A : Type} [Inhabited A]

/-- Every arbitrary old raw initial state supplies finite logical inputs,
including symbolic dot payloads and overlapping stride coordinates. -/
def inputsOfState (s : Shape) (t : State (Payload A)) : Inputs A s :=
  Fin.cases (fun j => (t.mem "input" (inputAddress s j)).read .fp32)
    (Fin.cases (fun j => (t.mem "weight" (weightAddress s j)).read .fp32) (fun i => Fin.elim0 i))

theorem inputsOfState_binding (o : Operations A) (s : Shape) (f : Flags)
    (t : State (Payload A)) : (program o s f).InputBinding (inputsOfState s t) t := by
  intro i j _
  fin_cases i <;> rfl

def referenceTarget (o : Operations A) (s : Shape) (f : Flags) (vals : Inputs A s) : Outputs A s :=
  Fin.cases (fun j => .scalar (reference o s f (logicalScalars o s vals) (outputCell s j)))
    (fun i => Fin.elim0 i)

/-- One connection of the already proved complete original launch to the
shared IO relation, including actual source safety and exact exterior frame. -/
theorem implements (o : Operations A) (s : Shape) (v : Valid s) (f : Flags) :
    (program o s f).Implements (referenceTarget o s f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact whole_safe o s v f vals bounds hb initial
  · intro vals initial bound
    obtain ⟨final,ran,inputs,values,frame⟩ := launch_reference o s v f initial
    refine ⟨final,?_,?_,frame_of_exterior o s v f vals initial final frame⟩
    · exact (schedule_run o s f (programs s) initial).trans ran
    · intro k j _
      fin_cases k
      change mapCell id (final.mem "output" (outputPointer s (outputCell s j)))=
        FP.Structural.Cell.mk .fp32 (.scalar (reference o s f (logicalScalars o s vals) (outputCell s j)))
      rw [values (outputCell s j) (active_output s j),
        reference_eq o s f _ _ (array_reads o s f vals initial bound) _ j.1.isLt]
      rfl

theorem real_implements (s : Shape) (v : Valid s) (f : Flags) :
    (realIO s f).Implements (mathematical s) := by
  have eq : referenceTarget KernelReal.operations s f=mathematical s := by
    funext vals i j
    fin_cases i
    change KernelNumeric.Value.scalar (reference KernelReal.operations s f _ (outputCell s j))=
      KernelNumeric.Value.scalar (Conv2d.mathematical s _ (outputCell s j))
    rw [KernelReal.reference_real s _ f _ (active_output s j)]
  simpa only [realIO,eq] using implements KernelReal.operations s v f

theorem native_implements (s : Shape) (v : Valid s) (f : Flags)
    (actual numerics : KernelNative.Instructions)
    (laws : KernelNative.LocalLaws actual (KernelNative.software numerics)) :
    (nativeIO actual s f).Implements (nativeTarget numerics s f) := by
  rw [KernelNative.instructions_eq actual _ laws]
  exact implements (KernelNative.operations (KernelNative.software numerics) f) s v f

theorem default_implements (s : Shape) (v : Valid s) (f : Flags)
    (defaults : f=⟨Bool.false,Bool.false⟩) (actual : KernelNative.Instructions)
    (law : ∀ a b c,actual.fmaRN a b c=Software.fmaRN a b c) :
    (nativeIO actual s f).Implements (defaultTarget actual s f) := by
  have eq : referenceTarget (KernelNative.operations actual f) s f=defaultTarget actual s f := by
    funext vals i j
    fin_cases i
    change KernelNumeric.Value.scalar (reference (KernelNative.operations actual f) s f _ (outputCell s j))=
      KernelNumeric.Value.scalar (KernelNative.defaultReference s _ (outputCell s j))
    rw [defaults,KernelNative.default_reference actual law]
  simpa only [nativeIO,eq] using implements (KernelNative.operations actual f) s v f

#axiomsClean real_implements
#axiomsClean native_implements
#axiomsClean default_implements
end
end VeriTile.Bench.Repairs.Conv2d.ConvIO
