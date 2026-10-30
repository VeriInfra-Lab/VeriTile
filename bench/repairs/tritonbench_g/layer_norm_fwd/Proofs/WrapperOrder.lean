import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperProtocol

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 8000
variable {S Key Value : Type} {world : World S Key Value}

def optionalResolveTrace (role : Role) : Option Tensor → List Stage
  | none => [] | some _ => [.resolve role,.transfer role]

def resolveTrace (q : Request) : List Stage :=
  [.saveOriginal,.resolve .x,.transfer .x,.resolve .weight,.transfer .weight] ++ optionalResolveTrace .bias q.bias ++
    optionalResolveTrace .residual q.residual ++ optionalResolveTrace .x1 q.x1 ++
    optionalResolveTrace .weight1 q.weight1 ++ optionalResolveTrace .bias1 q.bias1 ++
    optionalResolveTrace .rowscale q.rowscale

theorem materialize_trace (api : TensorAPI world) (target : Nat) (role : Role) (input : Tensor)
    (s : S) (output : Tensor) (h : ((materialize api target role input).run s).result=.ok output) :
    ((materialize api target role input).run s).trace=[.resolve role,.transfer role] := by
  unfold materialize at h ⊢
  dsimp only [Bind.bind] at h ⊢
  obtain ⟨_,_,_,_,ht⟩ := bind_success _ _ _ _ h
  rw [ht]
  rfl

theorem resolveOption_trace (api : TensorAPI world) (target : Nat) (role : Role) (input : Option Tensor)
    (s : S) (output : Option Tensor) (h : ((resolveOption api target role input).run s).result=.ok output) :
    ((resolveOption api target role input).run s).trace=optionalResolveTrace role input := by
  cases input with
  | none => rfl
  | some input =>
    unfold resolveOption at h ⊢
    dsimp only [Bind.bind] at h ⊢
    obtain ⟨value,hvalue,_,_,ht⟩ := bind_success _ _ _ _ h
    rw [ht,materialize_trace api target role input s value hvalue]
    rfl

theorem resolve_original (api : TensorAPI world) (q : Request) (s : S) (input : Inputs)
    (h : ((resolveInputs api q).run s).result=.ok input) : input.originalX=q.x := by
  unfold resolveInputs at h
  dsimp only [Bind.bind] at h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨w,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨b,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨r,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨x1,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨w1,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨b1,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨rs,_,h,_⟩ := bind_success _ _ _ _ h
  have hi : input=⟨q.x,x,w,b,r,x1,w1,b1,rs⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [hi]

theorem resolve_inputs_trace (api : TensorAPI world) (q : Request) (s : S) (input : Inputs)
    (h : ((resolveInputs api q).run s).result=.ok input) :
    ((resolveInputs api q).run s).trace=resolveTrace q := by
  unfold resolveInputs at h ⊢
  dsimp only [Bind.bind] at h ⊢
  obtain ⟨_,_,h,_,t₀⟩ := bind_success _ _ _ _ h
  rw [t₀]
  obtain ⟨x,hx,h,_,t₁⟩ := bind_success _ _ _ _ h
  rw [t₁,materialize_trace api q.x.device .x q.x _ x hx]
  obtain ⟨w,hw,h,_,t₂⟩ := bind_success _ _ _ _ h
  rw [t₂,materialize_trace api q.x.device .weight q.weight _ w hw]
  obtain ⟨b,hb,h,_,tb⟩ := bind_success _ _ _ _ h
  rw [tb,resolveOption_trace api q.x.device .bias q.bias _ b hb]
  obtain ⟨r,hr,h,_,tr⟩ := bind_success _ _ _ _ h
  rw [tr,resolveOption_trace api q.x.device .residual q.residual _ r hr]
  obtain ⟨x1,hx1,h,_,tx1⟩ := bind_success _ _ _ _ h
  rw [tx1,resolveOption_trace api q.x.device .x1 q.x1 _ x1 hx1]
  obtain ⟨w1,hw1,h,_,tw1⟩ := bind_success _ _ _ _ h
  rw [tw1,resolveOption_trace api q.x.device .weight1 q.weight1 _ w1 hw1]
  obtain ⟨b1,hb1,h,_,tb1⟩ := bind_success _ _ _ _ h
  rw [tb1,resolveOption_trace api q.x.device .bias1 q.bias1 _ b1 hb1]
  obtain ⟨rs,hrs,h,_,trs⟩ := bind_success _ _ _ _ h
  rw [trs,resolveOption_trace api q.x.device .rowscale q.rowscale _ rs hrs]
  simp [resolveTrace,note,call,Pure.pure,Action.pure,List.append_assoc]

/-- The width check receives the state after every successful allocation.
Its failure state and trace survive, including next_power_of_2 failures. -/
theorem prepare_after_allocations (api : TensorAPI world) (q : Request) (s : S)
    (v : Validated) (input : Inputs) (output : Outputs)
    (hv : ((validate (world:=world) q).run s).result=.ok v)
    (hi : ((resolveInputs api q).run ((validate (world:=world) q).run s).state).result=.ok input)
    (ho : ((allocateOutputs api q v input).run
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).result=.ok output) :
    let validated := (validate (world:=world) q).run s
    let resolved := (resolveInputs api q).run validated.state
    let allocated := (allocateOutputs api q v input).run resolved.state
    let checkedWidth := (width api v input).run allocated.state
    (prepare api q).run s=⟨checkedWidth.state,
      checkedWidth.result.map (fun block => Prepared.mk v input output block),
      validated.trace++resolved.trace++allocated.trace++checkedWidth.trace⟩ := by
  dsimp only
  cases hw : ((width api v input).run
    ((allocateOutputs api q v input).run
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).state).result <;>
    simp [prepare,Bind.bind,Action.bind,hv,hi,ho,hw,Pure.pure,Action.pure,Except.map,List.append_assoc]

theorem prepare_success (api : TensorAPI world) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) :
    p.validated.N≤p.block ∧ p.inputs.originalX=q.x ∧
    ∃ v input output block,
      ((validate (world:=world) q).run s).result=.ok v ∧
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).result=.ok input ∧
      ((allocateOutputs api q v input).run
        ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).result=.ok output ∧
      ((width api v input).run ((allocateOutputs api q v input).run
        ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).state).result=.ok block ∧
      p=⟨v,input,output,block⟩ := by
  unfold prepare at h
  dsimp only [Bind.bind] at h
  obtain ⟨v,hv,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨input,hi,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨output,ho,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨block,hb,h,_⟩ := bind_success _ _ _ _ h
  have hp : p=⟨v,input,output,block⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [hp]
  exact ⟨width_bound api v input _ block hb,resolve_original api q _ input hi,
    v,input,output,block,hv,hi,ho,hb,rfl⟩

/-- This exact state/trace decomposition is also the successful-prefix
certificate for a late width failure; it does not roll back allocations. -/
theorem width_failure_keeps_allocations (api : TensorAPI world) (q : Request) (s : S)
    (v : Validated) (input : Inputs) (output : Outputs)
    (hv : ((validate (world:=world) q).run s).result=.ok v)
    (hi : ((resolveInputs api q).run ((validate (world:=world) q).run s).state).result=.ok input)
    (ho : ((allocateOutputs api q v input).run
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).result=.ok output)
    (hw : ((width api v input).run ((allocateOutputs api q v input).run
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).state).result=.error .width) :
    ((prepare api q).run s).result=.error .width ∧
    ((prepare api q).run s).state=((width api v input).run ((allocateOutputs api q v input).run
      ((resolveInputs api q).run ((validate (world:=world) q).run s).state).state).state).state := by
  rw [prepare_after_allocations api q s v input output hv hi ho]
  simp [hw,Except.map]

#axiomsClean resolve_original
#axiomsClean materialize_trace
#axiomsClean resolve_inputs_trace
#axiomsClean prepare_after_allocations
#axiomsClean prepare_success
#axiomsClean width_failure_keeps_allocations
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
