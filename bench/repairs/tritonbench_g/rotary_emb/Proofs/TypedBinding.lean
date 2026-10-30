import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedNative
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Binding
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.RotaryEmb.TypedBinding
open VeriTile.Triton FP.Structural
open Checked
noncomputable section

/-- Actual Torch Tensor objects and their ordinary API observations. The
shared laws bind the logical elements to storage, without assuming outputs. -/
structure Objects (c : Call) where
  tensor : Fin 4 → VeriTile.Torch.FP32.Tensor
  laws : ∀ i,VeriTile.Torch.FP32.TensorLaws (tensor i)
  object : ∀ i,(tensor i).object=(c.tensor i).object
  shape : ∀ i,(tensor i).metadata.shape=(c.tensor i).shape
  pointer : ∀ i,(tensor i).dataPtr.toNat=(c.tensor i).base
  device : ∀ i,(tensor i).metadata.device=⟨"cuda",(c.tensor i).device⟩
  contiguous : ∀ i,(tensor i).metadata.isContiguous=(c.tensor i).contiguous
  negative : ∀ i,(tensor i).metadata.isNeg=(c.tensor i).negative
  storageOffset : ∀ i,(tensor i).storageOffset=(c.tensor i).storageOffset
  backing : ∀ i,(tensor i).storage.base=(c.tensor i).allocationBase ∧
    4*(tensor i).storage.elements=(c.tensor i).allocationBytes

theorem descriptor (c : Call) (accepted : check c=Bool.true) (i : Fin 4) :
    (c.tensor i).shape=expectedShape c.shape i ∧ (c.tensor i).contiguous=Bool.true ∧
      (c.tensor i).negative=Bool.false := by
  simp only [check,Bool.and_eq_true,metadataCheck,decide_eq_true_eq] at accepted
  have h := accepted.1.2.2 i
  exact ⟨h.2.2.2.2.2.2.2.2.2.1,h.2.2.1,h.2.2.2.1⟩

def logical {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (objects : Objects c) (m : M) (i : Fin 4) (o : Nat) : Native.Word :=
  (VeriTile.Torch.FP32.logicalValue
    (fun a => ⟨api.read m ⟨(c.tensor i).device,a⟩⟩)
      (objects.tensor i) (objects.laws i) o).bits

theorem direct {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (accepted : check c=Bool.true) (objects : Objects c) (m : M) (i : Fin 4) (o : Nat)
    (bound : o<extent c.shape i) :
    api.read m (Binding.tensorKey (c.tensor i) o)=logical api c objects m i o := by
  have desc := descriptor c accepted i
  have hc : (objects.tensor i).metadata.isContiguous=Bool.true :=
    (objects.contiguous i).trans desc.2.1
  have hn : (objects.tensor i).metadata.isNeg=Bool.false :=
    (objects.negative i).trans desc.2.2
  have size : (objects.tensor i).metadata.numel=extent c.shape i := by
    simp [VeriTile.Torch.TensorMetadata.numel,objects.shape,desc.1,extent]
  have hp := congrArg BitVec.toNat
    (VeriTile.Torch.FP32.actual_pointer (objects.tensor i) (objects.laws i) hc (by omega))
  have span := (VeriTile.Torch.FP32.dataView (objects.tensor i) (objects.laws i) hc).spanFits
  have baseBound : (VeriTile.Torch.FP32.dataView (objects.tensor i) (objects.laws i) hc).base<2^64 := by
    change _+4*(objects.tensor i).metadata.numel≤2^64 at span
    omega
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt baseBound,objects.pointer] at hp
  have eq := VeriTile.Torch.FP32.direct_read_eq_logical
    (fun a => (⟨api.read m ⟨(c.tensor i).device,a⟩⟩ : VeriTile.Torch.FP32.Word))
    (objects.tensor i) (objects.laws i) hc hn o (by omega)
  simpa only [logical,Binding.tensorKey,VeriTile.Torch.FP32.DataView.address,hp] using
    congrArg (fun w : VeriTile.Torch.FP32.Word => w.bits) eq

/-- Every typed load comes from the actual caller address. Other typed cells,
registers, program IDs, undefined values and launch metadata are retained. -/
def state {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (m : M) (surrounding : State Native.Node) : State Native.Node :=
  { surrounding with mem := fun r o =>
      if r="Q" then .mk .real (Native.atom (api.read m (Binding.tensorKey (c.tensor 0) o)))
      else if r="K" then .mk .real (Native.atom (api.read m (Binding.tensorKey (c.tensor 1) o)))
      else if r="Cos" then .mk .real (Native.atom (api.read m (Binding.tensorKey (c.tensor 2) o)))
      else if r="Sin" then .mk .real (Native.atom (api.read m (Binding.tensorKey (c.tensor 3) o)))
      else surrounding.mem r o }

theorem graph_caller {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (p : Native.Instructions) (c : Call) (m : M) (surrounding : State Native.Node)
    (key odd : Bool) (i : Typed.Lane (Typed.layout c.shape)) :
    Typed.graphValue p c.shape (state api c m surrounding) key odd i=.mk .real
      (Native.atom (Native.pairWord c.shape p (Typed.work c.shape surrounding key i)
        (Binding.inputs api c m) odd)) := by
  simp only [Typed.graphValue,Typed.offset_cell]
  cases key <;> cases odd <;>
    simp [Typed.region,Typed.layout,Typed.coefficient,Typed.token,Typed.head,
      Typed.dimension,Typed.work,state,Native.pairWord,Native.graph,Native.atom,Binding.inputs,
      dataRegion,Shape.coefficient,Shape.cell,Shape.address,Shape.heads,Nat.mul_comm]

/-- Relate the actual typed launch result to the caller-addressed physical
result already proved for the complete grid. This equality is derived from
the source and the independent graph, not a caller premise. -/
theorem physical_agreement {M : Type} (api : Binding.API M Native.Word Physical.Byte)
    (p : Native.Instructions) (c : Call) (accepted : check c=Bool.true) (m : M)
    (surrounding : State Native.Node) (rest : Algebra Native.Node)
    (key odd : Bool) (i : Typed.Lane (Typed.layout c.shape))
    (live : Typed.active (Typed.layout c.shape) surrounding key i) :
    api.read (Binding.execute api p c m)
      (Binding.key c (Typed.work c.shape surrounding key i,odd))=
      (((Typed.execution rest (Native.ops p) (Typed.layout c.shape)
        (state api c m surrounding)).val.mem (Typed.region (Typed.layout c.shape) key)
          (Typed.offset (Typed.layout c.shape) surrounding key odd i)).read .real).value := by
  have hw := (Typed.active_work c.shape (shape_valid c accepted) surrounding key i).mp live
  rw [Binding.readback api p c accepted m _ hw odd,Typed.execution_memory,Typed.reference_eq_target]
  have hs : (state api c m surrounding).pids=surrounding.pids := rfl
  have hr := Typed.target_readback (Native.ops p) c.shape (shape_valid c accepted)
    (state api c m surrounding) key odd i
      (by simpa only [Typed.active,Typed.token,Typed.head,hs] using live)
  have offsets : Typed.offset (Typed.layout c.shape) (state api c m surrounding) key odd i=
    Typed.offset (Typed.layout c.shape) surrounding key odd i := rfl
  rw [offsets] at hr
  rw [hr,Typed.value_graph,graph_caller]
  rfl

#axiomsClean direct
#axiomsClean graph_caller
#axiomsClean physical_agreement
end
end VeriTile.Bench.Repairs.RotaryEmb.TypedBinding
