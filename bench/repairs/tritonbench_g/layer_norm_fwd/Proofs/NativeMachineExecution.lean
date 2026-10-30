import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineAffine
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeRowPost

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def execute {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M) (physical : Memory c) : Memory c :=
  affineStage c row (normalizationStage c row (inputStage c row physical))

theorem input_registers {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) (template : State α) :
    (inputStage c row physical).view c template=
      NativeSchedule.inputStage c.ops c.flags c.regions c.strides c.M c.N row.val c.p block (view c physical template) := by
  obtain ⟨hm,hv⟩ := input_correspondence c row physical template
  change NativeSchedule.InputRegisters.mk _ _=NativeSchedule.InputRegisters.mk _ _
  congr 1

/-- Whole native row execution, with accepted launch words and actual guarded
physical loads/stores composed through every stage. No output or numerical
identity is an input to this theorem. -/
theorem execute_correspondence {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) (template : State α) :
    view c (execute c row physical) template=
      NativeSchedule.execute c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides c.M c.N row.val
        c.eps c.p c.accepted.blockPositive (view c physical template) := by
  rw [execute,affine_correspondence,normalization_correspondence,input_registers]
  rfl

/-- Complete seven-path row readback and tagged frame of the actual native
row. The independent target is parameterized only by initial input memory. -/
theorem execute_post {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) (template : State α)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N) :
    TypedSource.Frame c.flags c.regions c.strides c.M c.N row.val
      (view c physical template) (view c (execute c row physical) template) ∧
    NativeSchedule.RowValues c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides c.M c.N row.val
      c.eps c.p (view c physical template) (view c (execute c row physical) template) := by
  rw [execute_correspondence]
  exact NativeSchedule.execute_row_post c.ops c.reductions c.plan c.plan1 c.flags c.regions c.strides
    c.M c.N row.val c.eps c.p c.accepted.blockPositive (view c physical template) hsep hgeo row.isLt

#axiomsClean input_registers
#axiomsClean execute_correspondence
#axiomsClean execute_post
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
