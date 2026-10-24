import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageReady
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageView

/-! Arbitrary address-indexed Real values. No floating payload is decoded:
equal FP payloads at distinct addresses may denote different Real numbers.
Integer seeds, booleans and pointer metadata retain their actual values. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {α Host : Type} {caller : Address → Prop}

abbrev RealMemory := Address → MemCell

def interpretCell (x : ℝ) : Cell α → MemCell
  | .mk .real _ | .mk .fp32 _ | .mk .fp16 _ | .mk .bf16 _ | .mk .f8e4 _ | .mk .f8e5 _ => MemCell.real x
  | .mk .int v => MemCell.of .int v
  | .mk .nat v => MemCell.of .nat v
  | .mk .bool v => MemCell.of .bool v
  | .mk .ptr v => MemCell.of .ptr v
  | .mk .blockPtr v => MemCell.of .blockPtr v

/-- The parameter is a Real value at each physical cell, independently of bits. -/
def interpret (ρ : HostState α Host → Address → ℝ) (s : HostState α Host) : RealMemory :=
  fun i => interpretCell (ρ s i) (s.memory i)

def negateCell : MemCell → MemCell
  | ⟨.real,v⟩ => MemCell.of .real (v.map Neg.neg)
  | v => v

def tensorAddress (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (value : Tensor) (index : List Nat) : Nat :=
  LayerNormFwd.Address.byteAddress (store.descriptor s value) (coordinateOffset value.strides index)

def rawRead (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (memory : RealMemory) (value : Tensor) (index : List Nat) : MemCell :=
  memory (Fin.ofNat (2^64) (tensorAddress store s value index))

/-- Lazy negative views share storage but negate their floating logical cells. -/
def logicalRead (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (s : HostState α Host) (memory : RealMemory) (value : Tensor) (index : List Nat) : MemCell :=
  if Coordinate value index then
    if value.negative then negateCell (rawRead store s memory value index)
    else rawRead store s memory value index
  else 0

def shadowStore (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) : TensorStore (HostState α Host) Descriptor MemCell where
  present := store.present
  descriptor := store.descriptor
  read s value index := logicalRead store s (interpret ρ s) value index

/-- Observation-only world: effects and failures remain the actual Local.run.
The actual caller frame is proved separately from the production action. -/
def observationWorld (S : Type) : World S Unit MemCell := ⟨fun _ _ => 0,fun _ => False⟩

def revalueLocal {S Key Value β : Type} {w : World S Key Value}
    (operation : Local w β) : Local (observationWorld S) β :=
  ⟨operation.run,fun _ _ h => False.elim h⟩

structure Transport (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) : Prop where
  resolve : ∀ role input s output, ((tensor.resolve role input).run s).2=.ok output → ∀ index,
    (shadowStore store ρ).read ((tensor.resolve role input).run s).1 output index=
      (shadowStore store ρ).read s input index
  transfer : ∀ target role input s output, ((tensor.transfer target role input).run s).2=.ok output → ∀ index,
    (shadowStore store ρ).read ((tensor.transfer target role input).run s).1 output index=
      (shadowStore store ρ).read s input index

def revalueTensor (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) : TensorAPI (observationWorld (HostState α Host)) where
  read := (shadowStore store ρ).read
  callerObjects := tensor.callerObjects
  resolve role input := revalueLocal (tensor.resolve role input)
  resolve_metadata := tensor.resolve_metadata
  resolve_values := laws.resolve
  transfer target role input := revalueLocal (tensor.transfer target role input)
  transfer_metadata := tensor.transfer_metadata
  transfer_values := laws.transfer
  allocate role request := revalueLocal (tensor.allocate role request)
  allocate_metadata := tensor.allocate_metadata
  nextPower N := revalueLocal (tensor.nextPower N)
  nextPower_value := tensor.nextPower_value

def revalueDispatch (device : DispatchAPI (world (α:=α) (Host:=Host) caller)) :
    DispatchAPI (observationWorld (HostState α Host)) where
  enter n := revalueLocal (device.enter n)
  leave n := revalueLocal (device.leave n)
  select grid := revalueLocal (device.select grid)
  admit args := revalueLocal (device.admit args)
  split mask := revalueLocal (device.split mask)

theorem bind_run_congr {S K₁ V₁ K₂ V₂ β γ : Type} {w₁ : World S K₁ V₁} {w₂ : World S K₂ V₂}
    (a : Action w₁ β) (b : Action w₂ β) (f : β → Action w₁ γ) (g : β → Action w₂ γ)
    (ha : a.run=b.run) (hf : ∀ x, (f x).run=(g x).run) : (a.bind f).run=(b.bind g).run := by
  funext s
  simp only [Action.bind,ha]
  cases h : (b.run s).result with
  | error e => rfl
  | ok x => simp only [hf x]

theorem forOption_run_congr {S K₁ V₁ K₂ V₂ : Type} {w₁ : World S K₁ V₁} {w₂ : World S K₂ V₂}
    (v : Option Tensor) (f : Tensor → Action w₁ Unit) (g : Tensor → Action w₂ Unit)
    (hf : ∀ x, (f x).run=(g x).run) : (forOption v f).run=(forOption v g).run := by
  cases v with
  | none => rfl
  | some x => exact hf x

theorem validate_run {S K₁ V₁ K₂ V₂ : Type} {w₁ : World S K₁ V₁} {w₂ : World S K₂ V₂}
    (q : Request) : (validate (world:=w₁) q).run=(validate (world:=w₂) q).run := by
  cases hr : q.residual <;> simp only [validate,hr]
  all_goals
    apply bind_run_congr
    · rfl
    · intro residual
      apply bind_run_congr
      · rfl
      · rintro ⟨M,N⟩
        repeat' first | rfl | apply bind_run_congr | apply forOption_run_congr | intro

theorem resolveOption_run (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (target : Nat) (role : Wrapper.Role) (input : Option Tensor) :
    (resolveOption (revalueTensor store ρ tensor laws) target role input).run=(resolveOption tensor target role input).run := by
  cases input <;> rfl

theorem resolveInputs_run (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (q : Request) :
    (resolveInputs (revalueTensor store ρ tensor laws) q).run=(resolveInputs tensor q).run := by
  unfold resolveInputs
  repeat' first | rfl | apply resolveOption_run | apply bind_run_congr | intro

theorem allocateOptional_run (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (role : Wrapper.Role) (condition : Bool) (spec : Wrapper.Allocation) (check : Bool) :
    (allocateOptional (revalueTensor store ρ tensor laws) role condition spec check).run=
      (allocateOptional tensor role condition spec check).run := by
  cases condition <;> cases check <;> rfl

theorem allocateOutputs_run (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (q : Request) (v : Validated) (input : Wrapper.Inputs) :
    (allocateOutputs (revalueTensor store ρ tensor laws) q v input).run=(allocateOutputs tensor q v input).run := by
  unfold allocateOutputs
  repeat' first | rfl | apply allocateOptional_run | apply bind_run_congr | intro

/-- Exact control/effect equality, not a separate Real-wrapper success premise. -/
theorem prepare_run (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (q : Request) (s : HostState α Host) :
    (prepare (revalueTensor store ρ tensor laws) q).run s=(prepare tensor q).run s := by
  apply congrFun
  unfold prepare
  repeat' first | rfl | apply validate_run | apply resolveInputs_run | apply allocateOutputs_run | apply bind_run_congr | intro

theorem arguments_run (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (p : Prepared) (s : HostState α Host) :
    (arguments (revalueDispatch device) q p).run s=(arguments device q p).run s := by
  have optional (role : Wrapper.Role) (value : Option Tensor) :
      (optionalStride (world:=observationWorld (HostState α Host)) role value).run=
        (optionalStride (world:=world (α:=α) (Host:=Host) caller) role value).run := by cases value <;> rfl
  apply congrFun
  unfold arguments
  repeat' first | rfl | apply optional | apply bind_run_congr | intro

theorem finish_run (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (p : Prepared) (s : HostState α Host) :
    (finish (revalueDispatch device) p).run s=(finish device p).run s := by
  have split : (splitMasks (revalueDispatch device) p).run=(splitMasks device p).run := by
    cases hm : p.outputs.mask <;> cases hx : p.inputs.x1 <;> simp only [splitMasks,hm,hx] <;> rfl
  apply congrFun
  unfold finish
  repeat' first | rfl | exact split | apply bind_run_congr | intro

theorem launchState_eq (store : TensorStore (HostState α Host) Descriptor (Cell α))
    (ρ : HostState α Host → Address → ℝ) (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (laws : Transport store ρ tensor) (device : DispatchAPI (world caller))
    (q : Request) (s : HostState α Host) (p : Prepared) (args : LaunchArguments) :
    launchState (revalueTensor store ρ tensor laws) (revalueDispatch device) q s p args=
      launchState tensor device q s p args := by
  unfold launchState admittedState argumentsState enteredState preparedState
  rw [prepare_run,arguments_run]
  rfl

#axiomsClean prepare_run
#axiomsClean arguments_run
#axiomsClean finish_run
#axiomsClean launchState_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision
