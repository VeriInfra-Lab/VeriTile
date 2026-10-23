import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperArguments
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperMaterialized

namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {S Key Value : Type} {world : World S Key Value}

theorem inDevice_success (enter leave : Local world Unit) (body : Action world α)
    (s : S) (value : α) (h : ((inDevice enter leave body).run s).result=.ok value) :
    (enter.run s).2=.ok () ∧
    (body.run (enter.run s).1).result=.ok value ∧
    (leave.run (body.run (enter.run s).1).state).2=.ok () ∧
    ((inDevice enter leave body).run s).state=(leave.run (body.run (enter.run s).1).state).1 := by
  unfold inDevice at h ⊢
  obtain ⟨u,hu,hrest,hstate,_⟩ := bind_success _ _ _ _ h
  have he : (enter.run s).2=.ok () := by
    cases u
    exact call_success _ _ _ _ hu
  cases hl : (leave.run (body.run (enter.run s).1).state).2 with
  | error code => simp [call,hl,Except.mapError] at hrest
  | ok u =>
    cases u
    refine ⟨he,?_,rfl,?_⟩
    · simpa [call,hl,Except.mapError] using hrest
    · exact hstate

def deviceBody (api : DispatchAPI world) (kernel : LaunchArguments → Action world Unit)
    (q : Request) (p : Prepared) : Action world Unit := do
  let args ← arguments api q p
  call .admission (api.admit args)
  note .execute
  kernel args

def preparedState (api : TensorAPI world) (q : Request) (s : S) : S := ((prepare api q).run s).state
def enteredState (api : DispatchAPI world) (p : Prepared) (s : S) : S := (api.enter p.inputs.x.device).run s |>.1
def argumentsState (api : DispatchAPI world) (q : Request) (p : Prepared) (s : S) : S :=
  ((arguments api q p).run s).state
def admittedState (api : DispatchAPI world) (args : LaunchArguments) (s : S) : S := (api.admit args).run s |>.1

/-- A successful wrapper result determines its actual intermediate states.
No existential state can replace a failed prefix or a benchmark write. -/
structure SuccessRoute (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S) (result : Returned)
    (p : Prepared) (args : LaunchArguments) : Prop where
  prepared : ((prepare tensor q).run s).result=.ok p
  entered : ((device.enter p.inputs.x.device).run (preparedState tensor q s)).2=.ok ()
  arguments : ((Wrapper.arguments device q p).run
    (enteredState device p (preparedState tensor q s))).result=.ok args
  admitted : ((device.admit args).run (argumentsState device q p
    (enteredState device p (preparedState tensor q s)))).2=.ok ()
  executed : ((kernel args).run (admittedState device args (argumentsState device q p
    (enteredState device p (preparedState tensor q s))))).result=.ok ()
  exited : ((device.leave p.inputs.x.device).run (((kernel args).run
    (admittedState device args (argumentsState device q p
      (enteredState device p (preparedState tensor q s))))).state)).2=.ok ()
  finished : ((finish device p).run (((device.leave p.inputs.x.device).run
    (((kernel args).run (admittedState device args (argumentsState device q p
      (enteredState device p (preparedState tensor q s))))).state)).1)).result=.ok result
  finalState : ((surround tensor device kernel q).run s).state=
    ((finish device p).run (((device.leave p.inputs.x.device).run
      (((kernel args).run (admittedState device args (argumentsState device q p
        (enteredState device p (preparedState tensor q s))))).state)).1)).state

theorem surround_success_route (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S) (result : Returned)
    (h : ((surround tensor device kernel q).run s).result=.ok result) :
    ∃ p args, SuccessRoute tensor device kernel q s result p args := by
  have hh := h
  unfold surround at hh
  dsimp only [Bind.bind] at hh
  obtain ⟨p,hp,hh,h₀,_⟩ := bind_success _ _ _ _ hh
  obtain ⟨u,hu,hfinish,h₁,_⟩ := bind_success _ _ _ _ hh
  cases u
  have hdevice := inDevice_success (device.enter p.inputs.x.device) (device.leave p.inputs.x.device)
    (deviceBody device kernel q p) (preparedState tensor q s) () hu
  have hbody := hdevice.2.1
  unfold deviceBody at hbody
  dsimp only [Bind.bind] at hbody
  obtain ⟨args,ha,hbody,h₂,_⟩ := bind_success _ _ _ _ hbody
  obtain ⟨_,hadmit,hbody,h₃,_⟩ := bind_success _ _ _ _ hbody
  obtain ⟨_,_,hk,h₄,_⟩ := bind_success _ _ _ _ hbody
  have hbodyState : ((deviceBody device kernel q p).run
      (enteredState device p (preparedState tensor q s))).state=
      ((kernel args).run (admittedState device args (argumentsState device q p
        (enteredState device p (preparedState tensor q s))))).state := by
    unfold deviceBody enteredState
    dsimp only [Bind.bind]
    rw [h₂,h₃,h₄]
    rfl
  have hexit := hdevice.2.2.1
  have hcontext := hdevice.2.2.2
  change ((device.leave p.inputs.x.device).run ((deviceBody device kernel q p).run
    (enteredState device p (preparedState tensor q s))).state).2=.ok () at hexit
  rw [hbodyState] at hexit
  change ((inDevice (device.enter p.inputs.x.device) (device.leave p.inputs.x.device)
      (deviceBody device kernel q p)).run (preparedState tensor q s)).state=_ at hcontext
  change _=((device.leave p.inputs.x.device).run ((deviceBody device kernel q p).run
    (enteredState device p (preparedState tensor q s))).state).1 at hcontext
  rw [hbodyState] at hcontext
  refine ⟨p,args,hp,hdevice.1,ha,call_success _ _ _ _ hadmit,hk,hexit,?_,?_⟩
  · change ((finish device p).run ((inDevice (device.enter p.inputs.x.device)
        (device.leave p.inputs.x.device) (deviceBody device kernel q p)).run
        (preparedState tensor q s)).state).result=.ok result at hfinish
    rw [hcontext] at hfinish
    exact hfinish
  · unfold surround
    dsimp only [Bind.bind]
    rw [h₀,h₁]
    change ((finish device p).run ((inDevice (device.enter p.inputs.x.device)
      (device.leave p.inputs.x.device) (deviceBody device kernel q p)).run
      (preparedState tensor q s)).state).state=_
    rw [hcontext]

#axiomsClean inDevice_success
#axiomsClean surround_success_route
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
