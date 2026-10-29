import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperTensorMemory

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper NativeMachine
noncomputable section
set_option maxHeartbeats 2000000

/-- A region uses the actual prepared tensor pointer descriptor. The index is
a logical coordinate, while `offset` is the native element-offset expression.
This is a local address correspondence, not an assumed kernel result. -/
theorem descriptor_read {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (c : Config α block) (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor)
    (hn : tensor.negative=Bool.false) (region : RegionName)
    (hd : c.allocations region=store.descriptor s tensor)
    (index : List Nat) (hi : Coordinate tensor index) (offset : Nat)
    (ho : offset=coordinateOffset tensor.strides index) (template : State α) :
    (view c s.memory template).mem region offset=store.read s tensor index := by
  have hl : (PhysicalDomain.layout c.allocations).live (region,offset) := by
    change offset<(c.allocations region).extent
    rw [hd,ho]
    exact hm.bounds s tensor hp index hi
  change TypedPhysical.read (PhysicalDomain.layout c.allocations) s.memory (region,offset)=_
  rw [TypedPhysical.read,dif_pos hl,hm.read s tensor hp hn index hi]
  congr 1
  apply Fin.ext
  change LayerNormFwd.Address.byteAddress (c.allocations region) offset=_
  rw [hd,ho]

theorem descriptor_materialized_read {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (c : Config α block) (initial current : HostState α Host) (original tensor : Tensor)
    (h : Materialized store original initial tensor current) (region : RegionName)
    (hd : c.allocations region=store.descriptor current tensor)
    (index : List Nat) (hi : Coordinate tensor index) (offset : Nat)
    (ho : offset=coordinateOffset tensor.strides index) (template : State α) :
    (view c current.memory template).mem region offset=store.read initial original index :=
  (descriptor_read store hm c current tensor h.present h.positive region hd index hi offset ho template).trans
    (h.values index)

/-- Ordinary tensor descriptors and liveness depend on host metadata, not the
values written by GPU kernels into their storage. -/
structure HostDescriptors {α Host : Type}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) : Prop where
  present : ∀ s t, s.host=t.host → ∀ tensor, store.present s tensor → store.present t tensor
  descriptor : ∀ s t, s.host=t.host → ∀ tensor, store.descriptor s tensor=store.descriptor t tensor

theorem dispatch_tensor_present {α Host : Type} [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hh : HostDescriptors store)
    (c : Config α block) (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block)
    (s : HostState α Host) (tensor : Tensor) (hp : store.present s tensor) :
    store.present ((dispatchAction c caller hf benchmarks selected).run s).state tensor :=
  hh.present s ((dispatchAction c caller hf benchmarks selected).run s).state rfl tensor hp

theorem dispatch_tensor_descriptor {α Host : Type} [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hh : HostDescriptors store)
    (c : Config α block) (caller : Address → Prop) (hf : CallerFresh c caller)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block)
    (s : HostState α Host) (tensor : Tensor) :
    store.descriptor ((dispatchAction c caller hf benchmarks selected).run s).state tensor=store.descriptor s tensor :=
  hh.descriptor ((dispatchAction c caller hf benchmarks selected).run s).state s rfl tensor

#axiomsClean descriptor_read
#axiomsClean descriptor_materialized_read
#axiomsClean dispatch_tensor_present
#axiomsClean dispatch_tensor_descriptor
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
