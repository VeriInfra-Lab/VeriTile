import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.FPValues
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Physical

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Preparation
open VeriTile.Triton FP.Structural FPValues Geometry
noncomputable section
open Classical

/-- Same-call fresh allocation and coordinate-copy facts, before execution.
These fields bind regional source data to typed native-address reads; none
mentions execution success, intermediate values, or computed outputs. -/
structure Prepared (α : Type) (h : Call) (data : Data α) where
  storage : Storage.Placement h
  heap : Nat → Cell α
  input : ∀ off,off<storage.input.extent → heap (storage.input.address off)=
    .mk (dtype h.storage).toTileDType (ofFloat _ (data.input off))
  weights : ∀ off,off<storage.weights.extent → heap (storage.weights.address off)=
    .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights off))
  output : ∀ off,off<storage.output.extent → heap (storage.output.address off)=
    .mk h.outputType.toTileDType (ofFloat _ (data.output off))
  indices : ∀ off,off<storage.indices.extent → heap (storage.indices.address off)=.mk .int (data.indices off)

def Fresh {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (callerOwned : Nat → Prop) : Prop :=
  ∀ addr,callerOwned addr → ¬p.storage.input.Contains addr ∧ ¬p.storage.weights.Contains addr ∧
    ¬p.storage.output.Contains addr ∧ ¬p.storage.indices.Contains addr

theorem source_loads {α : Type} (h : Call) (data : Data α) (ready : Prepared α h data)
    (p : Program h) (slot : Fin h.adapters) (c : Fin h.cycles) (e : Fin 256) (n : Fin h.BN) :
    let k := rank p.split.val c.val e.val
    ready.heap (ready.storage.input.address (inputOffset h p.batch.val k))=
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.input (inputOffset h p.batch.val k))) ∧
    ready.heap (ready.storage.weights.address (weightOffset h slot.val n.val k))=
      .mk (dtype h.storage).toTileDType (ofFloat _ (data.weights (weightOffset h slot.val n.val k))) ∧
    ready.heap (ready.storage.output.address (outputOffset h p.batch.val n.val))=
      .mk h.outputType.toTileDType (ofFloat _ (data.output (outputOffset h p.batch.val n.val))) ∧
    ready.heap (ready.storage.indices.address p.batch.val)=.mk .int (data.indices p.batch.val) := by
  obtain ⟨x,w,y,i⟩ := Storage.all_source_offsets h ready.storage p slot c e n
  exact ⟨ready.input _ x,ready.weights _ w,ready.output _ y,ready.indices _ i⟩

def outputAddress {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (i : Fin h.batches × Fin h.N) := Physical.outputAddress p.storage i

theorem output_injective {α : Type} {h : Call} {data : Data α} (p : Prepared α h data) :
    Function.Injective (outputAddress p) := Physical.output_injective p.storage

def place {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (values : Fin h.batches → Fin h.N → Cell α) : Nat → Cell α :=
  Physical.place p.storage data.indices p.heap values

theorem physical_readback {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (values : Fin h.batches → Fin h.N → Cell α) (b : Fin h.batches) (n : Fin h.N)
    (active : data.indices b.val≠-1) : place p values (outputAddress p (b,n))=values b n :=
  Physical.readback p.storage data.indices p.heap values b n active

theorem caller_frame {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (values : Fin h.batches → Fin h.N → Cell α) (callerOwned : Nat → Prop)
    (fresh : Fresh p callerOwned) (addr : Nat) (owned : callerOwned addr) :
    place p values addr=p.heap addr :=
  Physical.frame p.storage data.indices p.heap values addr (fresh addr owned).2.2.1

#axiomsClean source_loads
#axiomsClean physical_readback
#axiomsClean caller_frame
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Preparation
