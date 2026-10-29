import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedLaunch

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

def OutsideCaller (store : TensorStore S Descriptor Value) (caller : Address → Prop)
    (s : S) (tensor : Tensor) : Prop :=
  ∀ i, caller i → ∀ byte, byte<(store.descriptor s tensor).extent*(store.descriptor s tensor).width →
    (store.descriptor s tensor).base+byte≠i.val

/-- Ordinary fresh-allocation ownership: new storage is disjoint from every
protected pre-call byte. The caller set may include full backing allocations,
including bytes outside the spans of current tensor views. -/
structure CallerAllocationAPI (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (caller : Address → Prop) : Prop where
  fresh : ∀ role request s output, ((api.allocate role request).run s).2=.ok output →
    OutsideCaller store caller ((api.allocate role request).run s).1 output

structure CallerBatch (store : TensorStore S Descriptor Value) (caller : Address → Prop)
    (initial : S) (outputs : List Tensor) (current : S) : Prop where
  preserved : store.Extends initial current
  present : ∀ output ∈ outputs, store.present current output
  outside : ∀ output ∈ outputs, OutsideCaller store caller current output

theorem CallerBatch.empty (store : TensorStore S Descriptor Value) (caller : Address → Prop) (s : S) :
    CallerBatch store caller s [] s := ⟨store.extends_refl s,by simp,by simp⟩

theorem CallerBatch.transfer {store : TensorStore S Descriptor Value} {caller : Address → Prop}
    {initial s t : S} {outputs : List Tensor} (h : CallerBatch store caller initial outputs s)
    (he : store.Extends s t) : CallerBatch store caller initial outputs t := by
  refine ⟨h.preserved.trans he,fun output ho => (he output (h.present output ho)).1,?_⟩
  intro output ho
  unfold OutsideCaller
  rw [(he output (h.present output ho)).2.1]
  exact h.outside output ho

theorem CallerBatch.append {store : TensorStore S Descriptor Value} {caller : Address → Prop}
    {initial s t : S} {outputs : List Tensor} (h : CallerBatch store caller initial outputs s)
    (he : store.Extends s t) (output : Tensor) (hp : store.present t output)
    (hf : OutsideCaller store caller t output) : CallerBatch store caller initial (outputs++[output]) t := by
  have ht := h.transfer he
  refine ⟨ht.preserved,?_,?_⟩
  · intro value hm
    rcases List.mem_append.mp hm with hm | hm
    · exact ht.present value hm
    · have hv : value=output := List.mem_singleton.mp hm
      exact hv.symm ▸ hp
  · intro value hm
    rcases List.mem_append.mp hm with hm | hm
    · exact ht.outside value hm
    · have hv : value=output := List.mem_singleton.mp hm
      exact hv.symm ▸ hf

theorem allocate_caller_batch (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : CallerAllocationAPI store api caller) (role : Wrapper.Role)
    (request : Wrapper.Allocation) (initial s : S) (outputs : List Tensor)
    (hb : CallerBatch store caller initial outputs s) (output : Tensor)
    (h : ((call (.allocate role) (api.allocate role request)).run s).result=.ok output) :
    CallerBatch store caller initial (outputs++[output])
      ((call (.allocate role) (api.allocate role request)).run s).state := by
  have hh := call_success _ _ _ _ h
  exact hb.append (hs.allocate_preserves role request s) output
    (hs.allocate_present role request s output hh) (hf.fresh role request s output hh)

theorem allocateOptional_caller_batch (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : CallerAllocationAPI store api caller) (role : Wrapper.Role)
    (condition : Bool) (request : Wrapper.Allocation) (assertStride : Bool)
    (initial s : S) (outputs : List Tensor) (hb : CallerBatch store caller initial outputs s)
    (output : Option Tensor)
    (h : ((allocateOptional api role condition request assertStride).run s).result=.ok output) :
    CallerBatch store caller initial (outputs++optionTensors output)
      ((allocateOptional api role condition request assertStride).run s).state := by
  cases condition
  · have ho : output=none := by simpa [allocateOptional,Pure.pure,Action.pure] using h.symm
    simpa [ho,allocateOptional,optionTensors,Pure.pure,Action.pure] using hb
  · unfold allocateOptional at h ⊢
    simp only [Bool.true_eq,ite_true,Bind.bind] at h ⊢
    obtain ⟨tensor,htensor,hrest,hstate,_⟩ := bind_success _ _ _ _ h
    have ht := allocate_caller_batch store api hs hf role request initial s outputs hb tensor htensor
    cases hg : assertStride
    · simp only [hg,Bool.false_eq_true,ite_false] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact ht
    · simp only [hg,ite_true] at hrest hstate ⊢
      obtain ⟨_,_,hfinal,hstate2,_⟩ := bind_success _ _ _ _ hrest
      have ho : output=some tensor := by simpa [Pure.pure,Action.pure] using hfinal.symm
      rw [ho,hstate,hstate2]
      exact ht.transfer (persists_checkStride store role tensor _)

theorem allocateOutputs_caller_storage (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hs : StorageAPI store api) (hf : CallerAllocationAPI store api caller)
    (q : Request) (v : Validated) (input : Inputs) (s : S) (output : Outputs)
    (h : ((allocateOutputs api q v input).run s).result=.ok output) :
    CallerBatch store caller s (outputTensors output) ((allocateOutputs api q v input).run s).state := by
  have hh := h
  unfold allocateOutputs at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨y,hy,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨_,_,hh,h₁,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨y1,hy1,hh,h₂,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨res,hres,hh,h₃,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨mean,hmean,hh,h₄,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨rstd,hrstd,hh,h₅,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨seeds,hseeds,hh,h₆,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨mask,hmask,hh,h₇,_⟩ := bind_success _ _ _ _ hh
  let out : Outputs := ⟨y,y1,res,mean,rstd,seeds,mask⟩
  have ho : output=out := by simpa [out,Pure.pure,Action.pure] using hh.symm
  let req := outputRequests api q v input out
  let sy := ((call (.allocate .y) (api.allocate .y req.y)).run s).state
  let st := ((checkStride (world:=world) .y y).run sy).state
  let sy1 := ((allocateOptional api .y1 input.weight1.isSome req.y1 Bool.true).run st).state
  let sr := ((allocateOptional api .residualOut (needsResidual q v input) req.residualOut Bool.true).run sy1).state
  let sm := ((allocateOptional api .mean (!q.rms) req.mean Bool.false).run sr).state
  let ss := ((call (.allocate .rstd) (api.allocate .rstd req.rstd)).run sm).state
  let se := ((allocateOptional api .seeds q.dropout req.seeds Bool.false).run ss).state
  let sf := ((allocateOptional api .mask (q.returnMask && q.dropout) req.mask Bool.false).run se).state
  have hfinal : ((allocateOutputs api q v input).run s).state=sf := by
    unfold allocateOutputs
    dsimp only [Bind.bind]
    rw [h₀,h₁,h₂,h₃,h₄,h₅,h₆,h₇]
    rfl
  have by0 : CallerBatch store caller s ([]++[y]) sy :=
    allocate_caller_batch store api hs hf .y req.y s s [] (CallerBatch.empty store caller s) y hy
  have byt : CallerBatch store caller s [y] st := by0.transfer (persists_checkStride store .y y sy)
  have by1 := allocateOptional_caller_batch store api hs hf .y1 input.weight1.isSome req.y1 Bool.true
    s st [y] byt y1 hy1
  have br := allocateOptional_caller_batch store api hs hf .residualOut (needsResidual q v input) req.residualOut Bool.true
    s sy1 _ by1 res hres
  have bm := allocateOptional_caller_batch store api hs hf .mean (!q.rms) req.mean Bool.false
    s sr _ br mean hmean
  have bs := allocate_caller_batch store api hs hf .rstd req.rstd s sm _ bm rstd hrstd
  have be := allocateOptional_caller_batch store api hs hf .seeds q.dropout req.seeds Bool.false
    s ss _ bs seeds hseeds
  have bf := allocateOptional_caller_batch store api hs hf .mask (q.returnMask && q.dropout) req.mask Bool.false
    s se _ be mask hmask
  rw [ho,hfinal]
  exact bf


#axiomsClean CallerBatch.transfer
#axiomsClean CallerBatch.append
#axiomsClean allocate_caller_batch
#axiomsClean allocateOptional_caller_batch
#axiomsClean allocateOutputs_caller_storage
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
