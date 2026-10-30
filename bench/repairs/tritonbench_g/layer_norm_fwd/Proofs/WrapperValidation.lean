import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperArguments

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

theorem checkShape_success (role : Role) (tensor : Tensor) (expected : List Nat) (s : S) (u : Unit)
    (h : ((checkShape (world:=world) role tensor expected).run s).result=.ok u) : tensor.shape=expected := by
  unfold checkShape at h
  dsimp only [Bind.bind] at h
  obtain ⟨actual,ha,hrest,_⟩ := bind_success _ _ _ _ h
  have he : actual=tensor.shape := by simpa [shape,checked] using ha.symm
  subst actual
  exact check_ok _ _ _ _ u hrest

theorem checkStride_success (role : Role) (tensor : Tensor) (s : S) (u : Unit)
    (h : ((checkStride (world:=world) role tensor).run s).result=.ok u) : tensor.strides.getLast?=some 1 := by
  unfold checkStride at h
  dsimp only [Bind.bind] at h
  obtain ⟨value,hv,hrest,_⟩ := bind_success _ _ _ _ h
  have he : value=1 := check_ok _ _ _ _ u hrest
  cases ht : tensor.strides.getLast? with
  | none => simp [stride,checked,ht] at hv
  | some v =>
    have hh : v=value := by simpa [stride,checked,ht] using hv
    rw [hh,he]

def OptionalFacts (input : Option Tensor) (P : Tensor → Prop) : Prop :=
  ∀ tensor, input=some tensor → P tensor

theorem forOption_success (input : Option Tensor) (body : Tensor → Action world Unit)
    (P : Tensor → Prop) (s : S) (u : Unit)
    (h : ((forOption input body).run s).result=.ok u)
    (hb : ∀ tensor, ((body tensor).run s).result=.ok u → P tensor) : OptionalFacts input P := by
  intro tensor ht
  rw [ht] at h
  exact hb tensor h

theorem unpackShape_success (tensor : Tensor) (s : S) (M N : Nat)
    (h : ((checked (world:=world) .unpackShape
      (match tensor.shape with | [a,b] => .ok (a,b) | _ => .error .rank)).run s).result=.ok (M,N)) :
    tensor.shape=[M,N] := by
  cases hs : tensor.shape with
  | nil => simp [checked,hs] at h
  | cons a rest =>
    cases rest with
    | nil => simp [checked,hs] at h
    | cons b tail =>
      cases tail with
      | nil =>
        have he : (a,b)=(M,N) := by simpa [checked,hs] using h
        obtain ⟨rfl,rfl⟩ := Prod.mk.inj he
        rfl
      | cons c tail => simp [checked,hs] at h

/-- Definitional factoring of the source do/match, used only to expose its
ordinary bind structure to the proof. -/
def validationTail (q : Request) (residualDtype : Option DType) : Action world Validated := do
  let (M,N) ← checked .unpackShape (match q.x.shape with
    | [M,N] => .ok (M,N) | _ => .error .rank)
  checkStride .x q.x
  forOption q.residual fun t => do
    checkStride .residual t
    checkShape .residual t [M,N]
  checkShape .weight q.weight [N]
  checkStride .weight q.weight
  forOption q.bias fun t => do
    checkStride .bias t
    checkShape .bias t [N]
  forOption q.x1 fun t => do
    checkShape .x1 t q.x.shape
    check .assertNoRowscale (q.rowscale=none) (.assertion .assertNoRowscale)
    checkStride .x1 t
  forOption q.weight1 fun t => do
    checkShape .weight1 t [N]
    checkStride .weight1 t
  forOption q.bias1 fun t => do
    checkShape .bias1 t [N]
    checkStride .bias1 t
  forOption q.rowscale fun t => do
    let c ← checked .contiguous (.ok t.contiguous)
    check .assertContiguous (c=Bool.true) (.assertion .assertContiguous)
    checkShape .rowscale t [M]
  pure ⟨M,N,residualDtype⟩

theorem validate_factor (q : Request) : validate (world:=world) q=
    (match q.residual with
      | none => Action.pure q.residualDtype
      | some tensor => checked .residualDtype (.ok (some tensor.dtype))).bind (validationTail q) := by
  unfold validate validationTail
  cases h : q.residual <;> simp only [h,Bind.bind,Pure.pure]
  all_goals rfl

structure ValidatedFacts (q : Request) (v : Validated) : Prop where
  x : q.x.shape=[v.M,v.N] ∧ q.x.strides.getLast?=some 1
  residual : OptionalFacts q.residual (fun t => t.shape=[v.M,v.N] ∧ t.strides.getLast?=some 1)
  weight : q.weight.shape=[v.N] ∧ q.weight.strides.getLast?=some 1
  bias : OptionalFacts q.bias (fun t => t.shape=[v.N] ∧ t.strides.getLast?=some 1)
  x1 : OptionalFacts q.x1 (fun t => t.shape=q.x.shape ∧ q.rowscale=none ∧ t.strides.getLast?=some 1)
  weight1 : OptionalFacts q.weight1 (fun t => t.shape=[v.N] ∧ t.strides.getLast?=some 1)
  bias1 : OptionalFacts q.bias1 (fun t => t.shape=[v.N] ∧ t.strides.getLast?=some 1)
  rowscale : OptionalFacts q.rowscale (fun t => t.shape=[v.M] ∧ t.contiguous=Bool.true)
  residualDtype : v.residualDtype=match q.residual with | none => q.residualDtype | some t => some t.dtype

/-- Every successful source validation supplies the actual checked shapes,
inner strides and x1/rowscale exclusion, including zero-sized dimensions. -/
theorem validate_facts (q : Request) (s : S) (v : Validated)
    (h : ((validate (world:=world) q).run s).result=.ok v) : ValidatedFacts q v := by
  rw [validate_factor] at h
  obtain ⟨dtype,hd,h,_⟩ := bind_success _ _ _ _ h
  unfold validationTail at h
  dsimp only [Bind.bind] at h
  obtain ⟨dims,hshape,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨M,N⟩ := dims
  obtain ⟨ux,hx,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨ur,hr,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨uw,hw,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨us,hs,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨ub,hb,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨ux1,hx1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨uw1,hw1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨ub1,hb1,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨urs,hrs,h,_⟩ := bind_success _ _ _ _ h
  have hv : v=⟨M,N,dtype⟩ := by simpa [Pure.pure,Action.pure] using h.symm
  rw [hv]
  refine ⟨⟨unpackShape_success q.x _ M N hshape,checkStride_success .x q.x _ ux hx⟩,?_,
    ⟨checkShape_success .weight q.weight [N] _ uw hw,checkStride_success .weight q.weight _ us hs⟩,
    ?_,?_,?_,?_,?_,?_⟩
  · refine forOption_success _ _ _ _ ur hr ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ ht
    exact ⟨checkShape_success .residual tensor [M,N] _ ur hrest,checkStride_success .residual tensor _ u hu⟩
  · refine forOption_success _ _ _ _ ub hb ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ ht
    exact ⟨checkShape_success .bias tensor [N] _ ub hrest,checkStride_success .bias tensor _ u hu⟩
  · refine forOption_success _ _ _ _ ux1 hx1 ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ ht
    obtain ⟨un,hn,hrest,_⟩ := bind_success _ _ _ _ hrest
    exact ⟨checkShape_success .x1 tensor q.x.shape _ u hu,check_ok _ _ _ _ un hn,
      checkStride_success .x1 tensor _ ux1 hrest⟩
  · refine forOption_success _ _ _ _ uw1 hw1 ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ ht
    exact ⟨checkShape_success .weight1 tensor [N] _ u hu,checkStride_success .weight1 tensor _ uw1 hrest⟩
  · refine forOption_success _ _ _ _ ub1 hb1 ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ ht
    exact ⟨checkShape_success .bias1 tensor [N] _ u hu,checkStride_success .bias1 tensor _ ub1 hrest⟩
  · refine forOption_success _ _ _ _ urs hrs ?_
    intro tensor ht
    dsimp only [Bind.bind] at ht
    obtain ⟨c,hc,hrest,_⟩ := bind_success _ _ _ _ ht
    obtain ⟨u,hu,hrest,_⟩ := bind_success _ _ _ _ hrest
    have he : c=tensor.contiguous := by simpa [checked] using hc.symm
    have hh : c=Bool.true := check_ok _ _ _ _ u hu
    exact ⟨checkShape_success .rowscale tensor [M] _ urs hrest,he.symm.trans hh⟩
  · cases ht : q.residual <;> simpa [ht,Pure.pure,Action.pure,checked] using hd.symm

theorem prepare_validated (api : TensorAPI world) (q : Request) (s : S) (p : Prepared)
    (h : ((prepare api q).run s).result=.ok p) : ValidatedFacts q p.validated := by
  unfold prepare at h
  dsimp only [Bind.bind] at h
  obtain ⟨v,hv,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  obtain ⟨_,_,h,_⟩ := bind_success _ _ _ _ h
  have hp : p.validated=v := by
    have he := congrArg (fun p : Prepared => p.validated) (Except.ok.inj h)
    exact he.symm
  rw [hp]
  exact validate_facts q s v hv

#axiomsClean checkShape_success
#axiomsClean checkStride_success
#axiomsClean unpackShape_success
#axiomsClean validate_facts
#axiomsClean prepare_validated
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
