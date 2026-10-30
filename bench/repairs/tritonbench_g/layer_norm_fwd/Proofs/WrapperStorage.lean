import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperProtocol

/-! Ordinary tensor-library persistence, separate from kernel execution.
Descriptors and cells belonging to already live tensors survive each resolve,
allocation and host bookkeeping step, including failed calls. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
set_option maxHeartbeats 3000000

structure TensorStore (S Descriptor Value : Type) where
  present : S → Tensor → Prop
  descriptor : S → Tensor → Descriptor
  read : S → Tensor → List Nat → Value

def TensorStore.Extends {S Descriptor Value : Type} (store : TensorStore S Descriptor Value) (s t : S) : Prop :=
  ∀ tensor, store.present s tensor → store.present t tensor ∧
    store.descriptor t tensor=store.descriptor s tensor ∧
    ∀ index, store.read t tensor index=store.read s tensor index

theorem TensorStore.extends_refl {S Descriptor Value : Type} (store : TensorStore S Descriptor Value) (s : S) :
    store.Extends s s := fun _ h => ⟨h,rfl,fun _ => rfl⟩

theorem TensorStore.Extends.trans {S Descriptor Value : Type} {store : TensorStore S Descriptor Value}
    {s t u : S} (h : store.Extends s t) (g : store.Extends t u) : store.Extends s u := by
  intro tensor hp
  obtain ⟨ht,hd,hv⟩ := h tensor hp
  obtain ⟨hu,he,hw⟩ := g tensor ht
  exact ⟨hu,he.trans hd,fun index => (hw index).trans (hv index)⟩

variable {S Key Value Descriptor : Type} {world : World S Key Value}

def Persists (store : TensorStore S Descriptor Value) (a : Action world α) : Prop :=
  ∀ s, store.Extends s (a.run s).state

theorem persists_pure (store : TensorStore S Descriptor Value) (value : α) :
    Persists (world:=world) store (Action.pure value) := store.extends_refl

theorem persists_checked (store : TensorStore S Descriptor Value) (stage : Stage) (value : Except Failure α) :
    Persists (world:=world) store (checked stage value) := store.extends_refl

theorem persists_note (store : TensorStore S Descriptor Value) (stage : Stage) :
    Persists (world:=world) store (note stage) := store.extends_refl

theorem persists_check (store : TensorStore S Descriptor Value) (stage : Stage)
    (condition : Prop) [Decidable condition] (error : Failure) :
    Persists (world:=world) store (check stage condition error) := store.extends_refl

theorem persists_call (store : TensorStore S Descriptor Value) (stage : Stage) (operation : Local world α)
    (h : ∀ s, store.Extends s (operation.run s).1) : Persists store (call stage operation) := h

theorem persists_bind (store : TensorStore S Descriptor Value) (a : Action world α) (f : α → Action world β)
    (ha : Persists store a) (hf : ∀ x, Persists store (f x)) : Persists store (a.bind f) := by
  intro s
  cases h : (a.run s).result with
  | error e => simpa [Action.bind,h] using ha s
  | ok x => simpa [Action.bind,h] using (ha s).trans (hf x (a.run s).state)

/-- No output or successful-kernel equation occurs in these contracts. The
freshly produced descriptor is made live, and previously live tensors persist. -/
structure StorageAPI (store : TensorStore S Descriptor Value) (api : TensorAPI world) : Prop where
  read : store.read=api.read
  resolve_preserves : ∀ role input s, store.Extends s (((api.resolve role input).run s).1)
  resolve_present : ∀ role input s output, ((api.resolve role input).run s).2=.ok output →
    store.present (((api.resolve role input).run s).1) output
  transfer_preserves : ∀ target role input s, store.Extends s (((api.transfer target role input).run s).1)
  transfer_present : ∀ target role input s output, ((api.transfer target role input).run s).2=.ok output →
    store.present (((api.transfer target role input).run s).1) output
  allocate_preserves : ∀ role request s, store.Extends s (((api.allocate role request).run s).1)
  allocate_present : ∀ role request s output, ((api.allocate role request).run s).2=.ok output →
    store.present (((api.allocate role request).run s).1) output
  power_preserves : ∀ N s, store.Extends s (((api.nextPower N).run s).1)

theorem persists_stride (store : TensorStore S Descriptor Value) (role : Role) (tensor : Tensor) (ax : Int) :
    Persists (world:=world) store (stride role tensor ax) := persists_checked store _ _

theorem persists_checkStride (store : TensorStore S Descriptor Value) (role : Role) (tensor : Tensor) :
    Persists (world:=world) store (checkStride role tensor) :=
  persists_bind store _ _ (persists_stride store role tensor (-1)) (fun _ => persists_check store _ _ _)

theorem persists_checkShape (store : TensorStore S Descriptor Value) (role : Role) (tensor : Tensor) (expected : List Nat) :
    Persists (world:=world) store (checkShape role tensor expected) := by
  unfold checkShape shape
  exact persists_bind store _ _ (persists_checked store _ _) (fun _ => persists_check store _ _ _)

theorem persists_forOption (store : TensorStore S Descriptor Value) (value : Option Tensor)
    (body : Tensor → Action world Unit) (h : ∀ tensor, Persists store (body tensor)) :
    Persists store (forOption value body) := by
  cases value with
  | none => exact persists_pure store ()
  | some tensor => exact h tensor

theorem persists_materialize (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (target : Nat) (role : Role) (input : Tensor) :
    Persists store (materialize api target role input) :=
  persists_bind store _ _ (persists_call store _ _ (hc.resolve_preserves role input))
    (fun resolved => persists_call store _ _ (hc.transfer_preserves target role resolved))

theorem persists_resolveOption (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (target : Nat) (role : Role) (value : Option Tensor) :
    Persists store (resolveOption api target role value) := by
  cases value with
  | none => exact persists_pure store none
  | some tensor =>
    exact persists_bind store _ _ (persists_materialize store api hc target role tensor)
      (fun output => persists_pure store (some output))

theorem persists_resolveInputs (store : TensorStore S Descriptor Value) (api : TensorAPI world)
    (hc : StorageAPI store api) (q : Request) : Persists store (resolveInputs api q) := by
  unfold resolveInputs
  apply persists_bind store _ _ (persists_note store _)
  intro _
  apply persists_bind store _ _ (persists_materialize store api hc q.x.device .x q.x)
  intro x
  apply persists_bind store _ _ (persists_materialize store api hc q.x.device .weight q.weight)
  intro weight
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .bias q.bias)
  intro bias
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .residual q.residual)
  intro residual
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .x1 q.x1)
  intro x1
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .weight1 q.weight1)
  intro weight1
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .bias1 q.bias1)
  intro bias1
  apply persists_bind store _ _ (persists_resolveOption store api hc q.x.device .rowscale q.rowscale)
  intro rowscale
  exact persists_pure store _

/-- Host-only dispatch/context operations preserve every currently live
tensor. Autotune benchmark writes belong to `WrapperNative.dispatchAction`,
whose input/frame preservation is derived from actual machine execution. -/
structure DispatchStorage (store : TensorStore S Descriptor Value) (api : DispatchAPI world) : Prop where
  enter : ∀ device s, store.Extends s ((api.enter device).run s).1
  leave : ∀ device s, store.Extends s ((api.leave device).run s).1
  select : ∀ grid s, store.Extends s ((api.select grid).run s).1
  admit : ∀ args s, store.Extends s ((api.admit args).run s).1
  split : ∀ tensor s, store.Extends s ((api.split tensor).run s).1

theorem persists_splitMasks (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hc : DispatchStorage store api) (p : Prepared) : Persists store (splitMasks api p) := by
  unfold splitMasks
  cases hm : p.outputs.mask <;> cases hx : p.inputs.x1
  all_goals first | exact persists_pure store _ | skip
  exact persists_bind store _ _ (persists_call store _ _ (hc.split _)) (fun _ => persists_pure store _)

theorem persists_finish (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hc : DispatchStorage store api) (p : Prepared) : Persists store (finish api p) :=
  persists_bind store _ _ (persists_splitMasks store api hc p) (fun _ =>
    persists_bind store _ _ (persists_note store _) (fun _ => persists_pure store _))

theorem finish_existing_cells (store : TensorStore S Descriptor Value) (api : DispatchAPI world)
    (hc : DispatchStorage store api) (p : Prepared) (s : S) (tensor : Tensor) (hp : store.present s tensor)
    (index : List Nat) : store.read ((finish api p).run s).state tensor index=store.read s tensor index :=
  (persists_finish store api hc p s tensor hp).2.2 index

#axiomsClean TensorStore.Extends.trans
#axiomsClean persists_bind
#axiomsClean persists_materialize
#axiomsClean persists_resolveInputs
#axiomsClean persists_splitMasks
#axiomsClean persists_finish
#axiomsClean finish_existing_cells
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
