import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineNormalization

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

def affineStage {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (input : NormalRegisters c) : Memory c :=
  let normalized := fun j : Fin block => if c.flags.rms then
    Expr.mul (input.rstd j.val) (input.centered j) else .mul (input.centered j) (input.rstd j.val)
  let w := fun j : Fin block => NativeMachineMemory.raw c.allocations c.regions c.strides
    c.M c.N row.val j.val c.flags .weight input.memory
  let b := fun j : Fin block => NativeMachineMemory.raw c.allocations c.regions c.strides
    c.M c.N row.val j.val c.flags .bias input.memory
  let y := fun j => affine c.plan c.flags.bias (normalized j) (w j) (b j)
  let s := NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y
    (fun j => .mk .real ((y j).eval c.ops)) input.memory
  if c.flags.w1 then
    let w1 := fun j : Fin block => NativeMachineMemory.raw c.allocations c.regions c.strides
      c.M c.N row.val j.val c.flags .weight1 s
    let b1 := fun j : Fin block => NativeMachineMemory.raw c.allocations c.regions c.strides
      c.M c.N row.val j.val c.flags .bias1 s
    let y1 := fun j => affine c.plan1 c.flags.b1 (normalized j) (w1 j) (b1 j)
    NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y1
      (fun j => .mk .real ((y1 j).eval c.ops)) s
  else s

/-- A weight/bias affine expression uses actual physical operands. Disabled
bias does not require a fictitious bias load or a bias-storage premise. -/
theorem affine_operands {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) (template : State α) (weight bias : Role) (flag : Bool)
    (hw : active c.flags weight) (hb : flag=Bool.true → active c.flags bias)
    (hwv : scalarRole weight=Bool.false) (hbv : scalarRole bias=Bool.false)
    (hwo : ∀ j, offset c.strides c.M c.N row.val j weight=0+j)
    (hbo : ∀ j, offset c.strides c.M c.N row.val j bias=0+j)
    (plan : AffinePlan) (x : NativeSchedule.Vector α block) :
    (fun j : Fin block => affine plan flag (x j)
      (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags weight physical)
      (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags bias physical))=
    (fun j : Fin block => affine plan flag (x j)
      (NativeSchedule.raw (view c physical template) (region c.regions weight) 0 c.N j.val)
      (NativeSchedule.raw (view c physical template) (region c.regions bias) 0 c.N j.val)) := by
  funext j
  have hweight := NativeMachineMemory.raw_correspondence c.flags c.regions c.strides c.M c.N block
    c.allocations c.accepted c.storage row.val j.val row.isLt j.isLt weight hw hwv 0 (hwo j.val) physical template
  have hbias : flag=Bool.true →
      NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags bias physical=
      NativeSchedule.raw (view c physical template) (region c.regions bias) 0 c.N j.val := by
    intro hf
    exact NativeMachineMemory.raw_correspondence c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val j.val row.isLt j.isLt bias (hb hf) hbv 0 (hbo j.val) physical template
  cases hf : flag
  · simp only [affine,Bool.false_eq_true,ite_false,hweight,view]
  · rw [hbias hf,hweight]
    rfl

theorem affine_correspondence {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (input : NormalRegisters c) (template : State α) :
    view c (affineStage c row input) template=
      NativeSchedule.affineStage c.ops c.plan c.plan1 c.flags c.regions c.strides c.N row.val (input.view c template) := by
  let normalized := fun j : Fin block => if c.flags.rms then
    Expr.mul (input.rstd j.val) (input.centered j) else .mul (input.centered j) (input.rstd j.val)
  let y := fun j : Fin block => affine c.plan c.flags.bias (normalized j)
    (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .weight input.memory)
    (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .bias input.memory)
  let yRef := fun j : Fin block => affine c.plan c.flags.bias (normalized j)
    (NativeSchedule.raw (view c input.memory template) c.regions.weight 0 c.N j.val)
    (NativeSchedule.raw (view c input.memory template) c.regions.bias 0 c.N j.val)
  let middle := NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y
    (fun j => Cell.mk .real ((y j).eval c.ops)) input.memory
  let middleRef := NativeSchedule.vectorStore (view c input.memory template) c.regions.y (row.val*c.strides.y) c.N block
    (fun j => Cell.mk .real ((yRef j).eval c.ops))
  have hy : y=yRef := affine_operands c row input.memory template .weight .bias c.flags.bias trivial (fun h => h)
    rfl rfl (fun _ => by simp [offset]) (fun _ => by simp [offset]) c.plan normalized
  have hm : view c middle template=middleRef := by
    change view c (NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y _ input.memory) template=_
    rw [show view c (NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y
        (fun j => Cell.mk .real ((y j).eval c.ops)) input.memory) template=
        NativeSchedule.vectorStore (view c input.memory template) c.regions.y (row.val*c.strides.y) c.N block
          (fun j => Cell.mk .real ((y j).eval c.ops)) from
      NativeMachineMemory.scatter_commutes c.flags c.regions c.strides c.M c.N block c.allocations c.accepted c.storage
        row.val row.isLt .y trivial trivial rfl (row.val*c.strides.y) (fun _ => rfl) _ input.memory template]
    rw [hy]
  cases hw : c.flags.w1
  · simpa only [affineStage,NativeSchedule.affineStage,hw,Bool.false_eq_true,ite_false] using hm
  · let y1 := fun j : Fin block => affine c.plan1 c.flags.b1 (normalized j)
      (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .weight1 middle)
      (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .bias1 middle)
    let y1Ref := fun j : Fin block => affine c.plan1 c.flags.b1 (normalized j)
      (NativeSchedule.raw middleRef c.regions.weight1 0 c.N j.val)
      (NativeSchedule.raw middleRef c.regions.bias1 0 c.N j.val)
    have hy1 : y1=y1Ref := by
      have he := affine_operands c row middle template .weight1 .bias1 c.flags.b1 hw (fun h => ⟨hw,h⟩)
        rfl rfl (fun _ => by simp [offset]) (fun _ => by simp [offset]) c.plan1 normalized
      change y1=(fun j : Fin block => affine c.plan1 c.flags.b1 (normalized j)
        (NativeSchedule.raw (view c middle template) c.regions.weight1 0 c.N j.val)
        (NativeSchedule.raw (view c middle template) c.regions.bias1 0 c.N j.val)) at he
      rw [hm] at he
      exact he
    change view c (if c.flags.w1 then NativeMachineMemory.scatter c.allocations c.regions c.strides
      c.M c.N row.val block c.flags .y1 (fun j => Cell.mk .real ((y1 j).eval c.ops)) middle else middle) template=
      (if c.flags.w1 then NativeSchedule.vectorStore middleRef c.regions.y1 (row.val*c.strides.y1) c.N block
        (fun j => Cell.mk .real ((y1Ref j).eval c.ops)) else middleRef)
    rw [if_pos hw,if_pos hw]
    rw [show view c (NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags .y1
        (fun j => Cell.mk .real ((y1 j).eval c.ops)) middle) template=
        NativeSchedule.vectorStore (view c middle template) c.regions.y1 (row.val*c.strides.y1) c.N block
          (fun j => Cell.mk .real ((y1 j).eval c.ops)) from
      NativeMachineMemory.scatter_commutes c.flags c.regions c.strides c.M c.N block c.allocations c.accepted c.storage
        row.val row.isLt .y1 trivial hw rfl (row.val*c.strides.y1) (fun _ => rfl) _ middle template]
    rw [hm,hy1]

#axiomsClean affine_operands
#axiomsClean affine_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
