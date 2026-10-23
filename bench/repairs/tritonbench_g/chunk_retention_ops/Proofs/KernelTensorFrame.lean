import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensor
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelPacket (Call)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section

/-- Exact logical and physical frames for the same resulting state. Physical
observations use the ordinary Tensor data pointer; disjoint caller input backing
is preserved, including storage-offset prefixes and tails. -/
def Frame {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (before : KernelPacket.Memory α) (after : State α) : Prop :=
  (∀ r off,¬KernelPacket.Footprint a r off → after.mem r off=before r off) ∧
  (∀ role (v : View a.shape role) addr,¬Written a role v addr →
    project decode read a role v after addr=read addr) ∧
  (∀ role (v : View a.shape role) (input : VeriTile.Torch.FP32.Tensor),
    (v.tensor.storage.base+4*v.tensor.storage.elements ≤ input.storage.base ∨
      input.storage.base+4*input.storage.elements ≤ v.tensor.storage.base) →
    ∀ addr,input.storage.base≤addr ∧ addr < input.storage.base+4*input.storage.elements →
      project decode read a role v after addr=read addr)

theorem frame_of_exec {α β : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (decode : α→β) (read : Nat→β) (a : Call) (m : KernelPacket.Memory α) (t : State α)
    (ht : FP.Structural.exec (engine (withDot M D)) a.kernel (a.state m)=some t) :
    Frame decode read a m t := by
  obtain ⟨u,hu,_,hf⟩:=KernelPacket.complete M D a m
  have he : u=t:=Option.some.inj (hu.symm.trans ht)
  subst u
  exact ⟨hf,fun role v addr h=>exterior decode read a role v t addr h,
    fun role v input apart addr inside=>input_frame decode read a role v t input apart addr inside⟩

#axiomsClean frame_of_exec
#specNonCircular Frame avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
