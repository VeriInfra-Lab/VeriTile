import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperAction
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.SourceBinding

/-! Source-ordered host prefix for actual tensor arguments. Host dropout uses
binary64 comparison, including NaN/infinities; no positivity assumption on M/N
is added. The allocation and resolve contracts are per-call library contracts.
This module supplies no successful-kernel or complete-output API oracle. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

structure DType where
  tag : Nat
  bytes : Nat
  bytes_pos : 0 < bytes
  deriving DecidableEq

def DType.fp32 : DType := ⟨0,4,by decide⟩
def DType.int64 : DType := ⟨1,8,by decide⟩
def DType.bool : DType := ⟨2,1,by decide⟩

structure Tensor where
  object : Nat
  shape : List Nat
  strides : List Int
  dtype : DType
  device : Nat
  contiguous : Bool
  negative : Bool
  deriving DecidableEq

structure Request where
  x : Tensor
  weight : Tensor
  bias : Option Tensor
  residual : Option Tensor
  x1 : Option Tensor
  weight1 : Option Tensor
  bias1 : Option Tensor
  rowscale : Option Tensor
  outDtype : Option DType
  residualDtype : Option DType
  eps : Float
  dropoutP : Float
  rms : Bool
  returnMask : Bool

def Request.dropout (q : Request) : Bool := 0.0 < q.dropoutP

structure Allocation where
  shape : List Nat
  dtype : DType
  device : Nat
  like : Option Tensor
  randomSeed : Bool
  used : List Nat

variable {S Key Value : Type} {world : World S Key Value}

/-- Metadata is immutable in a tensor handle; content is read from the current
host state. `resolve_values` is exactly coordinate preservation, with no stride
or identity equality required. Freshness checks all live caller and already
created objects supplied at that allocation. -/
structure TensorAPI (world : World S Key Value) where
  read : S → Tensor → List Nat → Value
  callerObjects : List Nat
  resolve : Role → Tensor → Local world Tensor
  resolve_metadata : ∀ role input s output,
    ((resolve role input).run s).2=.ok output →
    output.shape=input.shape ∧ output.dtype=input.dtype ∧ output.device=input.device ∧ output.negative=Bool.false
  resolve_values : ∀ role input s output,
    ((resolve role input).run s).2=.ok output →
    ∀ index, read ((resolve role input).run s).1 output index=read s input index
  /-- Actual Tensor.to(device=target, non_blocking=False), after resolve_neg.
  Same-owner calls may alias; cross-owner copies retain dtype and logical values.
  Stream readiness is separately composed from local stream contracts. -/
  transfer : Nat → Role → Tensor → Local world Tensor
  transfer_metadata : ∀ target role input s output,
    ((transfer target role input).run s).2=.ok output →
    output.shape=input.shape ∧ output.dtype=input.dtype ∧ output.device=target ∧ output.negative=input.negative
  transfer_values : ∀ target role input s output,
    ((transfer target role input).run s).2=.ok output →
    ∀ index, read ((transfer target role input).run s).1 output index=read s input index
  allocate : Role → Allocation → Local world Tensor
  allocate_metadata : ∀ role request s output,
    ((allocate role request).run s).2=.ok output →
    output.shape=request.shape ∧ output.dtype=request.dtype ∧ output.device=request.device ∧
      output.negative=Bool.false ∧ ∀ old ∈ request.used, output.object≠old
  nextPower : Nat → Local world Nat
  nextPower_value : ∀ N s block, ((nextPower N).run s).2=.ok block →
    block=if N=0 then 0 else if N≤1 then 1 else 2^(Nat.log2 (N-1)+1)

def shape (role : Role) (t : Tensor) : Action world (List Nat) :=
  checked (.shape role) (.ok t.shape)

def stride (role : Role) (t : Tensor) (ax : Int) : Action world Int :=
  checked (.stride role ax)
    (match if ax = -1 then t.strides.getLast? else t.strides[ax.toNat]? with
    | some value => .ok value | none => .error (.index role ax))

def checkStride (role : Role) (t : Tensor) : Action world Unit := do
  let v ← stride role t (-1)
  check (.assertStride role) (v=1) (.assertion (.assertStride role))

def checkShape (role : Role) (t : Tensor) (expected : List Nat) : Action world Unit := do
  let actual ← shape role t
  check (.assertShape role) (actual=expected) (.assertion (.assertShape role))

def forOption (value : Option Tensor) (body : Tensor → Action world Unit) : Action world Unit :=
  match value with | none => pure () | some t => body t

structure Validated where
  M : Nat
  N : Nat
  residualDtype : Option DType

/-- This is the Python assertion order, including bias stride before shape,
weight shape before stride, and x1 shape before the rowscale exclusion. -/
def validate (q : Request) : Action world Validated := do
  let residualDtype ← match q.residual with
    | none => pure q.residualDtype
    | some t => checked .residualDtype (.ok (some t.dtype))
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

structure Inputs where
  originalX : Tensor
  x : Tensor
  weight : Tensor
  bias : Option Tensor
  residual : Option Tensor
  x1 : Option Tensor
  weight1 : Option Tensor
  bias1 : Option Tensor
  rowscale : Option Tensor

def materialize (api : TensorAPI world) (target : Nat) (role : Role) (input : Tensor) : Action world Tensor := do
  let resolved ← call (.resolve role) (api.resolve role input)
  call (.transfer role) (api.transfer target role resolved)

def resolveOption (api : TensorAPI world) (target : Nat) (role : Role) : Option Tensor → Action world (Option Tensor)
  | none => pure none
  | some t => do let result ← materialize api target role t; pure (some result)

def resolveInputs (api : TensorAPI world) (q : Request) : Action world Inputs := do
  note .saveOriginal
  let x ← materialize api q.x.device .x q.x
  let weight ← materialize api q.x.device .weight q.weight
  let bias ← resolveOption api q.x.device .bias q.bias
  let residual ← resolveOption api q.x.device .residual q.residual
  let x1 ← resolveOption api q.x.device .x1 q.x1
  let weight1 ← resolveOption api q.x.device .weight1 q.weight1
  let bias1 ← resolveOption api q.x.device .bias1 q.bias1
  let rowscale ← resolveOption api q.x.device .rowscale q.rowscale
  pure ⟨q.x,x,weight,bias,residual,x1,weight1,bias1,rowscale⟩

def objectIds (values : List (Option Tensor)) : List Nat :=
  values.filterMap (fun value => value.map Tensor.object)

def Inputs.objects (input : Inputs) : List Nat :=
  objectIds [some input.originalX,some input.x,some input.weight,input.bias,input.residual,
    input.x1,input.weight1,input.bias1,input.rowscale]

def needsResidual (q : Request) (v : Validated) (input : Inputs) : Bool :=
  input.residual.isSome || (v.residualDtype.isSome && decide (v.residualDtype≠some input.x.dtype)) ||
    q.dropout || input.rowscale.isSome || input.x1.isSome

structure Outputs where
  y : Tensor
  y1 : Option Tensor
  residualOut : Option Tensor
  mean : Option Tensor
  rstd : Tensor
  seeds : Option Tensor
  mask : Option Tensor

def Outputs.objects (output : Outputs) : List Nat :=
  objectIds [some output.y,output.y1,output.residualOut,output.mean,some output.rstd,output.seeds,output.mask]

def allocateOptional (api : TensorAPI world) (role : Role) (condition : Bool)
    (spec : Allocation) (assertStride : Bool) : Action world (Option Tensor) := do
  if condition then
    let value ← call (.allocate role) (api.allocate role spec)
    if assertStride then checkStride role value
    pure (some value)
  else pure none

/-- Output allocations occur before element_size, next_power_of_2 and width
validation. Freshness includes every preceding materialization/allocation. -/
def allocateOutputs (api : TensorAPI world) (q : Request) (v : Validated) (input : Inputs) : Action world Outputs := do
  let used := api.callerObjects++input.objects
  let y ← call (.allocate .y) (api.allocate .y
    ⟨input.x.shape,q.outDtype.getD input.x.dtype,input.x.device,some input.x,Bool.false,used⟩)
  checkStride .y y
  let used := used++[y.object]
  let y1 ← allocateOptional api .y1 input.weight1.isSome
    ⟨y.shape,y.dtype,y.device,some y,Bool.false,used⟩ Bool.true
  let used := used++objectIds [y1]
  let residualOut ← allocateOptional api .residualOut (needsResidual q v input)
    ⟨[v.M,v.N],v.residualDtype.getD input.x.dtype,input.x.device,none,Bool.false,used⟩ Bool.true
  let used := used++objectIds [residualOut]
  let mean ← allocateOptional api .mean (!q.rms)
    ⟨[v.M],.fp32,input.x.device,none,Bool.false,used⟩ Bool.false
  let used := used++objectIds [mean]
  let rstd ← call (.allocate .rstd) (api.allocate .rstd
    ⟨[v.M],.fp32,input.x.device,none,Bool.false,used⟩)
  let used := used++[rstd.object]
  let count := if input.x1.isSome then 2*v.M else v.M
  let seeds ← allocateOptional api .seeds q.dropout
    ⟨[count],.int64,input.x.device,none,Bool.true,used⟩ Bool.false
  let used := used++objectIds [seeds]
  let mask ← allocateOptional api .mask (q.returnMask && q.dropout)
    ⟨[count,v.N],.bool,input.x.device,none,Bool.false,used⟩ Bool.false
  pure ⟨y,y1,residualOut,mean,rstd,seeds,mask⟩

structure Prepared where
  validated : Validated
  inputs : Inputs
  outputs : Outputs
  block : Nat

def width (api : TensorAPI world) (v : Validated) (input : Inputs) : Action world Nat := do
  let bytes ← checked .elementSize (.ok input.x.dtype.bytes)
  let next ← call .nextPower (api.nextPower v.N)
  let block := min (65536 / bytes) next
  check .width (v.N≤block) .width
  pure block

def prepare (api : TensorAPI world) (q : Request) : Action world Prepared := do
  let validated ← validate q
  let inputs ← resolveInputs api q
  let outputs ← allocateOutputs api q validated inputs
  let block ← width api validated inputs
  pure ⟨validated,inputs,outputs,block⟩

def Prepared.flags (q : Request) (p : Prepared) : Flags where
  rms := q.rms
  residual := p.inputs.residual.isSome
  residualOut := p.outputs.residualOut.isSome
  bias := p.inputs.bias.isSome
  dropout := q.dropout
  dropoutMask := p.outputs.mask.isSome
  rowscale := p.inputs.rowscale.isSome
  x1 := p.inputs.x1.isSome
  w1 := p.inputs.weight1.isSome
  b1 := p.inputs.bias1.isSome

theorem prepare_frame (api : TensorAPI world) (q : Request) (s : S) (key : Key)
    (hk : world.caller key) : world.observe ((prepare api q).run s).state key=world.observe s key :=
  (prepare api q).frame s key hk

theorem check_ok (stage : Stage) (condition : Prop) [Decidable condition] (e : Failure)
    (s : S) (u : Unit) (h : ((check (world:=world) stage condition e).run s).result=.ok u) : condition := by
  by_contra hc
  simp [check,hc] at h

theorem width_bound (api : TensorAPI world) (v : Validated) (input : Inputs) (s : S) (block : Nat)
    (h : ((width api v input).run s).result=.ok block) : v.N≤block := by
  unfold width at h
  dsimp only [Bind.bind] at h
  obtain ⟨bytes,hbytes,hrest,_⟩ := bind_success _ _ _ _ h
  have hb : bytes=input.x.dtype.bytes := by simpa [checked] using hbytes.symm
  subst bytes
  obtain ⟨next,hnext,hrest,_⟩ := bind_success _ _ _ _ hrest
  obtain ⟨u,hcheck,hrest,_⟩ := bind_success _ _ _ _ hrest
  have hn := check_ok _ _ _ _ u hcheck
  have he : min (65536/input.x.dtype.bytes) next=block := by simpa [Pure.pure,Action.pure] using hrest
  simpa [he] using hn

#axiomsClean prepare_frame
#axiomsClean width_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
