import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.TensorValues
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton
noncomputable section
set_option maxHeartbeats 4000000

def inputWords {phase} (r : Request phase) (heap : ByteMemory) : RegionName → Nat → Word :=
  fun rg off => load32 heap (base r rg+4*off)

def torchWord (heap : ByteMemory) (x : Payload) (api : PayloadLaws x) (i : Nat) : Word :=
  (Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩) x.tensor api.tensor i).bits

theorem pointer_word (heap : ByteMemory) (x : Payload) (api : PayloadLaws x)
    (hc : x.tensor.metadata.isContiguous=Bool.true) (hn : x.tensor.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<x.tensor.metadata.numel) :
    load32 heap (x.tensor.dataPtr.toNat+4*i)=torchWord heap x api i := by
  simp only [torchWord,Torch.FP32.logicalValue,hn,Bool.false_eq_true,if_false]
  rw [payload_pointer x api hc i hi]

def IncomingReads : {phase : Phase} → (r : Request phase) → Contract r → ByteMemory → Prop
  | .forward,_,_,_ => True
  | .backward,r,h,heap => ∀ i,i<r.incoming.original.tensor.metadata.numel →
      inputWords r heap "incoming" i=torchWord heap r.incoming.original h.incoming.1.1 i

structure NativeReads {phase} (r : Request phase) (h : Contract r) (heap : ByteMemory) : Prop where
  query : ∀ i,i<r.q.tensor.metadata.numel → inputWords r heap "query" i=torchWord heap r.q h.q.1 i
  key : ∀ i,i<r.k.tensor.metadata.numel → inputWords r heap "key" i=torchWord heap r.k h.k.1 i
  value : ∀ i,i<r.v.tensor.metadata.numel → inputWords r heap "value" i=torchWord heap r.v h.v.1 i
  incoming : IncomingReads r h heap

theorem native_reads {phase} (r : Request phase) (h : Contract r) (heap : ByteMemory)
    (prep : Preparation r (load32 heap)) : NativeReads r h heap := by
  constructor
  · intro i hi
    exact pointer_word heap r.q h.q.1 (h.q.2.2.2.1 rfl) h.q.2.2.2.2.1 i hi
  · intro i hi
    simpa [inputWords,base,argument] using pointer_word heap r.k h.k.1 (h.k.2.2.2.1 rfl) h.k.2.2.2.2.1 i hi
  · intro i hi
    simpa [inputWords,base,argument] using pointer_word heap r.v h.v.1 (h.v.2.2.2.1 rfl) h.v.2.2.2.2.1 i hi
  · cases phase with
    | forward => trivial
    | backward =>
      intro i hi
      have hd := h.incoming.2
      have ho := h.incoming.1
      have bi : i<r.incoming.dense.tensor.metadata.numel := by
        rw [descriptor_numel _ _ _ _ _ hd]
        rwa [descriptor_numel _ _ _ _ _ ho] at hi
      change Prepared r (load32 heap) at prep
      have hp := prep i hi
      rw [logical_direct _ _ hd.1 (hd.2.2.2.1 rfl) i bi] at hp
      change load32 heap (base r "incoming"+4*i)=_
      rw [show base r "incoming"=r.incoming.dense.tensor.dataPtr.toNat by rfl,hp]
      simp [logical,torchWord,Torch.FP32.logicalValue,ho.2.2.2.2.1]

#axiomsClean native_reads
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
