import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineStream

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

structure InputRegisters {α : Type} (c : Config α block) where
  memory : Memory c
  value : NativeSchedule.Vector α block

def inputStage {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) : InputRegisters c :=
  let first := stream c row Bool.false physical
  let t := storeMask c row Bool.false first physical
  let second := stream c row Bool.true t
  let u := if c.flags.x1 then storeMask c row Bool.true second t else t
  let both := fun j => if c.flags.x1 then Expr.add (first.value j) (second.value j) else first.value j
  let combined := fun j => if c.flags.residual then
      let res := NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .residual u
      if c.flags.rowscale && !c.flags.dropout && !c.flags.x1 then Expr.fma (first.raw j) first.scale res
      else .add (both j) res
    else both j
  let final := if c.flags.residualOut then NativeMachineMemory.scatter c.allocations c.regions c.strides
    c.M c.N row.val block c.flags .residualOut (fun j => .mk .real (c.ops.cast32 ((combined j).eval c.ops))) u else u
  ⟨final,combined⟩

theorem StreamMatch.storeMask {α : Type} {f : Flags} {word logical : NativeSchedule.StreamRegisters α block}
    (h : StreamMatch f word logical) (r : Regions) (index N : Nat) (s : State α) :
    NativeSchedule.storeMask f r index N block word s=NativeSchedule.storeMask f r index N block logical s := by
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp only [NativeSchedule.storeMask,hd,hm,Bool.false_and,Bool.true_and,Bool.false_eq_true,ite_false,ite_true]
  all_goals first | rfl | rw [h.keep hd]

/-- Complete physical input stage: parameter words, pid, guards, every load,
both mask stores and the residual store simulate the checked native schedule.
This is a local program composition, with no arithmetic or output premise. -/
theorem input_correspondence {α : Type} [Inhabited α] (c : Config α block) (row : Fin c.M)
    (physical : Memory c) (template : State α) :
    view c (inputStage c row physical).memory template=
      (NativeSchedule.inputStage c.ops c.flags c.regions c.strides c.M c.N row.val c.p block (view c physical template)).memory ∧
    (inputStage c row physical).value=
      (NativeSchedule.inputStage c.ops c.flags c.regions c.strides c.M c.N row.val c.p block (view c physical template)).value := by
  let first := stream c row Bool.false physical
  let t := storeMask c row Bool.false first physical
  let second := stream c row Bool.true t
  let u := if c.flags.x1 then storeMask c row Bool.true second t else t
  let firstRef := NativeSchedule.stream c.ops c.flags c.regions c.regions.x (row.val*c.strides.x)
    row.val c.N c.p block (view c physical template)
  let tRef := NativeSchedule.storeMask c.flags c.regions row.val c.N block firstRef (view c physical template)
  let secondRef := NativeSchedule.stream c.ops c.flags c.regions c.regions.x1 (row.val*c.strides.x1)
    (c.M+row.val) c.N c.p block tRef
  let uRef := if c.flags.x1 then NativeSchedule.storeMask c.flags c.regions (c.M+row.val) c.N block secondRef tRef else tRef
  have hfirst : StreamMatch c.flags first firstRef :=
    stream_correspondence c row Bool.false (by simp) physical template
  have ht : view c t template=tRef := by
    rw [mask_correspondence c row Bool.false (by simp) first physical template]
    exact hfirst.storeMask c.regions row.val c.N _
  have hsecond : c.flags.x1=Bool.true → StreamMatch c.flags second secondRef := by
    intro hx
    have he := stream_correspondence c row Bool.true (fun _ => hx) t template
    change StreamMatch c.flags second (NativeSchedule.stream c.ops c.flags c.regions c.regions.x1
      (row.val*c.strides.x1) (c.M+row.val) c.N c.p block (view c t template)) at he
    rw [ht] at he
    exact he
  have hu : view c u template=uRef := by
    cases hx : c.flags.x1
    · simpa only [u,uRef,hx,Bool.false_eq_true,ite_false] using ht
    · change view c (if c.flags.x1 then storeMask c row Bool.true second t else t) template=_
      rw [if_pos hx,mask_correspondence c row Bool.true (fun _ => hx) second t template,ht]
      simpa only [uRef,hx,ite_true] using (hsecond hx).storeMask c.regions (c.M+row.val) c.N tRef
  have hres : c.flags.residual=Bool.true → ∀ j : Fin block,
      NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .residual u=
      NativeSchedule.raw uRef c.regions.residual (row.val*c.strides.residual) c.N j.val := by
    intro hf j
    have he := NativeMachineMemory.raw_correspondence c.flags c.regions c.strides c.M c.N block
      c.allocations c.accepted c.storage row.val j.val row.isLt j.isLt .residual hf rfl
      (row.val*c.strides.residual) rfl u template
    change _=NativeSchedule.raw (view c u template) c.regions.residual (row.val*c.strides.residual) c.N j.val at he
    rw [hu] at he
    exact he
  have hv : (inputStage c row physical).value=
      (NativeSchedule.inputStage c.ops c.flags c.regions c.strides c.M c.N row.val c.p block (view c physical template)).value := by
    funext j
    have hfv := congrFun hfirst.value j
    have hfr := congrFun hfirst.raw j
    have hsv : c.flags.x1=Bool.true → second.value j=secondRef.value j :=
      fun hx => congrFun (hsecond hx).value j
    change (if c.flags.residual then
      if c.flags.rowscale && !c.flags.dropout && !c.flags.x1 then
        Expr.fma (first.raw j) first.scale
          (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .residual u)
      else Expr.add (if c.flags.x1 then .add (first.value j) (second.value j) else first.value j)
        (NativeMachineMemory.raw c.allocations c.regions c.strides c.M c.N row.val j.val c.flags .residual u)
      else if c.flags.x1 then .add (first.value j) (second.value j) else first.value j)=
      (if c.flags.residual then
      if c.flags.rowscale && !c.flags.dropout && !c.flags.x1 then
        Expr.fma (firstRef.raw j) firstRef.scale (NativeSchedule.raw uRef c.regions.residual (row.val*c.strides.residual) c.N j.val)
      else Expr.add (if c.flags.x1 then .add (firstRef.value j) (secondRef.value j) else firstRef.value j)
        (NativeSchedule.raw uRef c.regions.residual (row.val*c.strides.residual) c.N j.val)
      else if c.flags.x1 then .add (firstRef.value j) (secondRef.value j) else firstRef.value j)
    cases hx : c.flags.x1 <;> cases hr : c.flags.residual <;> cases hs : c.flags.rowscale <;> cases hd : c.flags.dropout
    all_goals simp [hx,hr,hs,hd,hfv,hfr,hsv,hfirst.scale,hres]
  refine ⟨?_,hv⟩
  change view c (if c.flags.residualOut then NativeMachineMemory.scatter c.allocations c.regions c.strides
    c.M c.N row.val block c.flags .residualOut
      (fun j => Cell.mk .real (c.ops.cast32 (((inputStage c row physical).value j).eval c.ops))) u else u) template=
    (if c.flags.residualOut then NativeSchedule.vectorStore uRef c.regions.residualOut (row.val*c.strides.residualOut) c.N block
      (fun j => Cell.mk .real (c.ops.cast32 (((NativeSchedule.inputStage c.ops c.flags c.regions c.strides
        c.M c.N row.val c.p block (view c physical template)).value j).eval c.ops))) else uRef)
  cases ho : c.flags.residualOut
  · simpa only [ho,Bool.false_eq_true,ite_false] using hu
  · simp only [ho,ite_true]
    rw [show view c (NativeMachineMemory.scatter c.allocations c.regions c.strides c.M c.N row.val block c.flags
      .residualOut (fun j => Cell.mk .real (c.ops.cast32 (((inputStage c row physical).value j).eval c.ops))) u) template=
      NativeSchedule.vectorStore (view c u template) c.regions.residualOut (row.val*c.strides.residualOut) c.N block
        (fun j => Cell.mk .real (c.ops.cast32 (((inputStage c row physical).value j).eval c.ops))) from
      NativeMachineMemory.scatter_commutes c.flags c.regions c.strides c.M c.N block c.allocations c.accepted c.storage
        row.val row.isLt .residualOut trivial ho rfl (row.val*c.strides.residualOut) (fun _ => rfl) _ u template]
    rw [hu,hv]

#axiomsClean input_correspondence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
