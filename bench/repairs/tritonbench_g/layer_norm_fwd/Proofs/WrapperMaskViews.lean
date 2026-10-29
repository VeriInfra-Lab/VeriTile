import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorage

/-! Ordinary chunk/view coordinate contract used by the actual wrapper return
suffix. It says nothing about kernel values; the source mask is the current
post-dispatch tensor, and split outputs read its two existing row halves. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 2000000
variable {S Key Value Descriptor : Type} {world : World S Key Value}

structure MaskSplit (store : TensorStore S Descriptor Value) (api : DispatchAPI world) : Prop where
  views : ∀ (mask : Tensor) (M N : Nat) (s : S) (first second : Tensor),
    mask.shape=[2*M,N] → ((api.split mask).run s).2=.ok (first,second) →
    first.shape=[M,N] ∧ second.shape=[M,N] ∧
    first.dtype=mask.dtype ∧ second.dtype=mask.dtype ∧
    first.device=mask.device ∧ second.device=mask.device ∧
    store.present ((api.split mask).run s).1 first ∧ store.present ((api.split mask).run s).1 second ∧
    (∀ (i : Fin M) (j : Fin N), store.read ((api.split mask).run s).1 first [i.val,j.val]=store.read s mask [i.val,j.val]) ∧
    (∀ (i : Fin M) (j : Fin N), store.read ((api.split mask).run s).1 second [i.val,j.val]=store.read s mask [M+i.val,j.val])

def returnedWithMasks (p : Prepared) (mask mask1 : Option Tensor) : Returned :=
  ⟨p.outputs.y,p.outputs.y1,p.outputs.mean,p.outputs.rstd,
    p.outputs.residualOut.getD p.inputs.originalX,p.outputs.seeds,mask,mask1⟩

theorem finish_split_run (api : DispatchAPI world) (p : Prepared) (s : S)
    (mask x1 first second : Tensor) (hm : p.outputs.mask=some mask) (hx : p.inputs.x1=some x1)
    (hs : ((api.split mask).run s).2=.ok (first,second)) :
    (finish api p).run s=
      ⟨((api.split mask).run s).1,.ok (returnedWithMasks p (some first) (some second)),[.splitMask,.returnTuple]⟩ := by
  simp [finish,splitMasks,hm,hx,Bind.bind,Action.bind,Pure.pure,Action.pure,call,hs,note,returnedWithMasks,Except.mapError]

theorem finish_no_second_run (api : DispatchAPI world) (p : Prepared) (s : S) (hx : p.inputs.x1=none) :
    (finish api p).run s=⟨s,.ok (returnedWithMasks p p.outputs.mask none),[.returnTuple]⟩ := by
  cases hm : p.outputs.mask <;>
    simp [finish,splitMasks,hm,hx,Bind.bind,Action.bind,Pure.pure,Action.pure,note,returnedWithMasks]

theorem finish_no_mask_run (api : DispatchAPI world) (p : Prepared) (s : S) (hm : p.outputs.mask=none) :
    (finish api p).run s=⟨s,.ok (returnedWithMasks p none none),[.returnTuple]⟩ := by
  cases hx : p.inputs.x1 <;>
    simp [finish,splitMasks,hm,hx,Bind.bind,Action.bind,Pure.pure,Action.pure,note,returnedWithMasks]

theorem finish_split_failure (api : DispatchAPI world) (p : Prepared) (s : S)
    (mask x1 : Tensor) (hm : p.outputs.mask=some mask) (hx : p.inputs.x1=some x1) (code : Nat)
    (hs : ((api.split mask).run s).2=.error code) :
    (finish api p).run s=⟨((api.split mask).run s).1,.error (.api .splitMask code),[.splitMask]⟩ := by
  simp [finish,splitMasks,hm,hx,Bind.bind,Action.bind,call,hs,Except.mapError]

/-- Split output descriptors and every returned mask cell refer to the actual
post-dispatch tensor halves. For M=0 or N=0 the view shapes/identities remain
explicit while the corresponding coordinate assertions are vacuous. -/
theorem finish_split_cells (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hc : MaskSplit store api) (p : Prepared) (s : S) (mask x1 first second : Tensor)
    (hm : p.outputs.mask=some mask) (hx : p.inputs.x1=some x1)
    (hshape : mask.shape=[2*p.validated.M,p.validated.N])
    (hs : ((api.split mask).run s).2=.ok (first,second)) :
    ((finish api p).run s).result=.ok (returnedWithMasks p (some first) (some second)) ∧
    first.shape=[p.validated.M,p.validated.N] ∧ second.shape=[p.validated.M,p.validated.N] ∧
    store.present ((finish api p).run s).state first ∧ store.present ((finish api p).run s).state second ∧
    (∀ (i : Fin p.validated.M) (j : Fin p.validated.N),
      store.read ((finish api p).run s).state first [i.val,j.val]=store.read s mask [i.val,j.val]) ∧
    (∀ (i : Fin p.validated.M) (j : Fin p.validated.N),
      store.read ((finish api p).run s).state second [i.val,j.val]=store.read s mask [p.validated.M+i.val,j.val]) := by
  rw [finish_split_run api p s mask x1 first second hm hx hs]
  obtain ⟨hfirst,hsecond,_,_,_,_,hp,hq,hv,hw⟩ := hc.views mask p.validated.M p.validated.N s first second hshape hs
  exact ⟨rfl,hfirst,hsecond,hp,hq,hv,hw⟩

#axiomsClean finish_split_run
#axiomsClean finish_no_second_run
#axiomsClean finish_no_mask_run
#axiomsClean finish_split_failure
#axiomsClean finish_split_cells
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
