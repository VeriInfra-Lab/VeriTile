import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPrelude

/-! Ordered device-context and return suffix. The kernel body is an explicit
parameter to this surrounding host program; it is not a TensorAPI operation.
Consequently the unconditional results here concern wrapper control and the
caller-cell frame, not native kernel correctness. A checked kernel action must
be supplied by a separate execution composition. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

variable {S Key Value : Type} {world : World S Key Value}

structure LaunchArguments where
  prepared : Prepared
  request : Request
  strideX : Int
  strideY : Int
  strideResidual : Int
  strideResidualOut : Int
  strideX1 : Int
  strideY1 : Int

/-- The grid is passed unchanged, including zero rows and row counts that may
be rejected by launch admission. No signed-i32 restriction is silently imposed
on the wrapper domain. -/
def LaunchArguments.grid (args : LaunchArguments) : List Nat := [args.prepared.validated.M]

structure DispatchAPI (world : World S Key Value) where
  enter : Nat → Local world Unit
  leave : Nat → Local world Unit
  select : List Nat → Local world Unit
  admit : LaunchArguments → Local world Unit
  split : Tensor → Local world (Tensor × Tensor)

def optionalStride (role : Role) (value : Option Tensor) : Action world Int :=
  match value with | none => pure 0 | some t => stride role t 0

def arguments (api : DispatchAPI world) (q : Request) (p : Prepared) : Action world LaunchArguments := do
  call .selectKernel (api.select [p.validated.M])
  let sx ← stride .x p.inputs.x 0
  let sy ← stride .y p.outputs.y 0
  let sr ← optionalStride .residual p.inputs.residual
  let so ← optionalStride .residualOut p.outputs.residualOut
  let sx1 ← optionalStride .x1 p.inputs.x1
  let sy1 ← optionalStride .y1 p.outputs.y1
  pure ⟨p,q,sx,sy,sr,so,sx1,sy1⟩

structure Returned where
  y : Tensor
  y1 : Option Tensor
  mean : Option Tensor
  rstd : Tensor
  residual : Tensor
  seeds : Option Tensor
  mask : Option Tensor
  mask1 : Option Tensor

def splitMasks (api : DispatchAPI world) (p : Prepared) : Action world (Option Tensor × Option Tensor) :=
  match p.outputs.mask,p.inputs.x1 with
  | some mask,some _ => do
    let pair ← call .splitMask (api.split mask)
    pure (some pair.1,some pair.2)
  | mask,_ => pure (mask,none)

def finish (api : DispatchAPI world) (p : Prepared) : Action world Returned := do
  let masks ← splitMasks api p
  note .returnTuple
  pure ⟨p.outputs.y,p.outputs.y1,p.outputs.mean,p.outputs.rstd,
    p.outputs.residualOut.getD p.inputs.originalX,p.outputs.seeds,masks.1,masks.2⟩

/-- Host orchestration with an explicit kernel action. `admit` handles ordinary
dispatch/compilation/setup rejection only; it is not allowed to supply output
cells. The actual execution is the separate `kernel` argument. -/
def surround (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) : Action world Returned := do
  let p ← prepare tensor q
  inDevice (device.enter p.inputs.x.device) (device.leave p.inputs.x.device) (do
    let args ← arguments device q p
    call .admission (device.admit args)
    note .execute
    kernel args)
  finish device p

theorem surrounding_frame (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S) (key : Key)
    (hk : world.caller key) :
    world.observe ((surround tensor device kernel q).run s).state key=world.observe s key :=
  (surround tensor device kernel q).frame s key hk

theorem finish_return_identity (api : DispatchAPI world) (p : Prepared) (s : S) (result : Returned)
    (h : ((finish api p).run s).result=.ok result) :
    result.y=p.outputs.y ∧ result.y1=p.outputs.y1 ∧ result.mean=p.outputs.mean ∧
    result.rstd=p.outputs.rstd ∧ result.residual=p.outputs.residualOut.getD p.inputs.originalX ∧
    result.seeds=p.outputs.seeds := by
  unfold finish at h
  dsimp only [Bind.bind] at h
  obtain ⟨masks,_,hrest,_⟩ := bind_success _ _ _ _ h
  obtain ⟨u,_,hrest,_⟩ := bind_success _ _ _ _ hrest
  have hr : result=⟨p.outputs.y,p.outputs.y1,p.outputs.mean,p.outputs.rstd,
    p.outputs.residualOut.getD p.inputs.originalX,p.outputs.seeds,masks.1,masks.2⟩ := by
    simpa [Pure.pure,Action.pure] using hrest.symm
  rw [hr]
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem finish_fallback_identity (api : DispatchAPI world) (p : Prepared) (s : S) (result : Returned)
    (h : ((finish api p).run s).result=.ok result) (hn : p.outputs.residualOut=none) :
    result.residual=p.inputs.originalX := by
  simpa [hn] using (finish_return_identity api p s result h).2.2.2.2.1

/-- A failed launch admission runs no kernel action. The enclosing device
context still runs its exit, as established by `exit_after_body_failure`. -/
theorem admission_failure (api : DispatchAPI world) (args : LaunchArguments)
    (kernel : LaunchArguments → Action world Unit) (s : S) (code : Nat)
    (h : ((api.admit args).run s).2=.error code) :
    ((do call .admission (api.admit args); note .execute; kernel args : Action world Unit).run s)=
      ⟨((api.admit args).run s).1,.error (.api .admission code),[.admission]⟩ := by
  simp [Bind.bind,Action.bind,call,h,Except.mapError]

theorem empty_grid_preserved (args : LaunchArguments) (h : args.prepared.validated.M=0) :
    args.grid=[0] := by simp [LaunchArguments.grid,h]

theorem surrounding_outcomes (tensor : TensorAPI world) (device : DispatchAPI world)
    (kernel : LaunchArguments → Action world Unit) (q : Request) (s : S) :
    (∃ e, ((surround tensor device kernel q).run s).result=.error e) ∨
    (∃ returned, ((surround tensor device kernel q).run s).result=.ok returned) := by
  cases h : ((surround tensor device kernel q).run s).result with
  | error e => exact Or.inl ⟨e,rfl⟩
  | ok r => exact Or.inr ⟨r,rfl⟩

#axiomsClean surrounding_frame
#axiomsClean finish_return_identity
#axiomsClean finish_fallback_identity
#axiomsClean admission_failure
#axiomsClean empty_grid_preserved
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
