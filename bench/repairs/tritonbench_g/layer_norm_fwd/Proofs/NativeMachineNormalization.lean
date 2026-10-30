import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineInput

/-! Native normalization graphs and actual finite-width scalar stores. The
mean owner of each centered reduction leaf is retained in the graph. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def InputRegisters.view {α : Type} (c : Config α block) (input : InputRegisters c)
    (template : State α) : NativeSchedule.InputRegisters α block :=
  ⟨NativeMachine.view c input.memory template,input.value⟩

structure NormalRegisters {α : Type} (c : Config α block) where
  memory : Memory c
  value : NativeSchedule.Vector α block
  mean : Nat → Expr α
  centered : NativeSchedule.Vector α block
  rstd : Nat → Expr α

def NormalRegisters.view {α : Type} (c : Config α block) (input : NormalRegisters c)
    (template : State α) : NativeSchedule.NormalRegisters α block :=
  ⟨NativeMachine.view c input.memory template,input.value,input.mean,input.centered,input.rstd⟩

def normalizationStage {α : Type} (c : Config α block) (row : Fin c.M)
    (input : InputRegisters c) : NormalRegisters c :=
  let zero : Fin block := ⟨0,c.accepted.blockPositive⟩
  let mean := fun owner => if c.N=1 then input.value zero else
    Expr.div (sumExpr (c.reductions owner).mean input.value) (.literal c.N)
  let s := if c.flags.rms then input.memory else
    NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .mean
      input.memory (.mk .real (c.ops.cast32 ((mean 0).eval c.ops)))
  let centered := fun j : Fin block => if c.flags.rms then input.value j else .sub (input.value j) (mean j.val)
  let masked := fun j : Fin block => if j.val<c.N then centered j else Expr.literal 0
  let variance := fun owner => Expr.div (squareExpr (c.reductions owner).variance masked) (.literal c.N)
  let rstd := fun owner => Expr.div (.literal 1) (.sqrt (if c.N=1 then
    .fma (centered zero) (centered zero) (.input c.eps) else .add (.input c.eps) (variance owner)))
  let t := NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .rstd
    s (.mk .real (c.ops.cast32 ((rstd 0).eval c.ops)))
  ⟨t,input.value,mean,centered,rstd⟩

theorem normalization_memory {α : Type} (c : Config α block) (row : Fin c.M)
    (input : InputRegisters c) (template : State α) :
    view c (normalizationStage c row input).memory template=
      (NativeSchedule.normalizationStage c.ops c.reductions c.flags c.regions c.N row.val c.eps
        c.accepted.blockPositive (input.view c template)).memory := by
  let meanValue : Cell α := .mk .real (c.ops.cast32 (((normalizationStage c row input).mean 0).eval c.ops))
  let rstdValue : Cell α := .mk .real (c.ops.cast32 (((normalizationStage c row input).rstd 0).eval c.ops))
  let middle := if c.flags.rms then input.memory else
    NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .mean input.memory meanValue
  have hmiddle : view c middle template=
      if c.flags.rms then view c input.memory template else
        NativeSchedule.scalarStore (view c input.memory template) c.regions.mean row.val meanValue := by
    cases hr : c.flags.rms
    · simp only [middle,hr,Bool.false_eq_true,ite_false]
      exact NativeMachineMemory.scalar_store_commutes c.flags c.regions c.strides c.M c.N block
        c.allocations c.accepted c.storage row.val row.isLt .mean trivial hr rfl input.memory meanValue template
    · simp only [middle,hr,ite_true]
  change view c (NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .rstd
      middle rstdValue) template=
    NativeSchedule.scalarStore (if c.flags.rms then view c input.memory template else
      NativeSchedule.scalarStore (view c input.memory template) c.regions.mean row.val meanValue)
      c.regions.rstd row.val rstdValue
  rw [show view c (NativeMachineMemory.store c.allocations c.regions c.strides c.M c.N row.val 0 c.flags .rstd
      middle rstdValue) template=
      NativeSchedule.scalarStore (view c middle template) c.regions.rstd row.val rstdValue from
    NativeMachineMemory.scalar_store_commutes c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val row.isLt .rstd trivial trivial rfl middle rstdValue template]
  rw [hmiddle]

/-- The entire normalization register record and current memory agree, without
an arithmetic assumption. Actual mean/rstd stores include the N=0 case. -/
theorem normalization_correspondence {α : Type} (c : Config α block) (row : Fin c.M)
    (input : InputRegisters c) (template : State α) :
    (normalizationStage c row input).view c template=
      NativeSchedule.normalizationStage c.ops c.reductions c.flags c.regions c.N row.val c.eps
        c.accepted.blockPositive (input.view c template) := by
  have hm := normalization_memory c row input template
  change NativeSchedule.NormalRegisters.mk _ _ _ _ _=NativeSchedule.NormalRegisters.mk _ _ _ _ _
  congr 1

#axiomsClean normalization_memory
#axiomsClean normalization_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
