import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPValues
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.NativeControl
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ReturnedStorage
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.PhysicalGrid

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Preparation
open VeriTile.Triton FP.Structural FPValues ReturnedStorage

/-- Ordinary local allocation and snapshot facts, all for the actual same-call
padded buffers. These bind source regional cells to typed reads at the native
byte addresses. No execution or final-output equation is a field. -/
structure Prepared (α : Type) (h : Call) (data : Data α) where
  storage : Storage.Placement h
  heap : Nat → Cell α
  input : ∀ off,off<storage.input.extent →
    heap (storage.input.address off)=.mk .real (data.input off)
  weights : ∀ off,off<storage.weights.extent →
    heap (storage.weights.address off)=.mk .fp16 (data.weights off)
  output : ∀ off,off<storage.output.extent →
    heap (storage.output.address off)=.mk .real (data.output off)
  indices : ∀ off,off<storage.indices.extent →
    heap (storage.indices.address off)=.mk .int (data.indices off)

/-- Fresh scratch allocations are disjoint from every caller-owned byte,
including aliased inputs/weights/metadata and the full output backing. -/
def Fresh {α : Type} {h : Call} {data : Data α} (p : Prepared α h data)
    (callerOwned : Nat → Prop) : Prop :=
  ∀ address,callerOwned address →
    ¬p.storage.input.Contains address ∧ ¬p.storage.weights.Contains address ∧
    ¬p.storage.output.Contains address ∧ ¬p.storage.indices.Contains address

def outputHeap {α : Type} [Inhabited α] {h : Call} {data : Data α} (p : Prepared α h data) : Nat → α :=
  fun address => (p.heap address).read .real

theorem output_bound {α : Type} [Inhabited α] {h : Call} {data : Data α} (p : Prepared α h data) :
    ∀ off,off<p.storage.output.extent → outputHeap p (p.storage.output.address off)=data.output off := by
  intro off inside
  simp [outputHeap,p.output off inside]

theorem source_loads {α : Type} {h : Call} {data : Data α} (ready : Prepared α h data)
    (p : Geometry.Program h) (slot : Nat) (selected : slot<h.adapters) (n : Fin 256) (k : Fin h.BK) :
    ready.heap (ready.storage.input.address (Geometry.inputOffset h p.batch k.val))=
      .mk .real (data.input (Geometry.inputOffset h p.batch k.val)) ∧
    ready.heap (ready.storage.weights.address (Geometry.weightOffset h slot p.row n.val k.val))=
      .mk .fp16 (data.weights (Geometry.weightOffset h slot p.row n.val k.val)) ∧
    ready.heap (ready.storage.output.address (Geometry.outputOffset h p.batch p.row n.val))=
      .mk .real (data.output (Geometry.outputOffset h p.batch p.row n.val)) ∧
    ready.heap (ready.storage.indices.address p.batch)=.mk .int (data.indices p.batch) := by
  obtain ⟨x,w,y,i⟩ := Storage.all_source_offsets h ready.storage p slot selected n k
  exact ⟨ready.input _ x,ready.weights _ w,ready.output _ y,ready.indices _ i⟩

/-- Fresh scratch stores preserve every address in every caller backing,
including aliases and bytes outside the output view. -/
theorem caller_frame {α : Type} [Inhabited α] {h : Call} {data : Data α}
    (M : Algebra α) (reduction : Reduction α) (ready : Prepared α h data)
    (callerOwned : Nat → Prop) (fresh : Fresh ready callerOwned)
    (programs : List (Geometry.Program h)) (address : Nat) (owned : callerOwned address) :
    PhysicalGrid.run M reduction h ready.storage programs data (outputHeap ready) address=outputHeap ready address := by
  apply PhysicalGrid.frame
  intro p _ same
  have inside : FPGrid.address h p<ready.storage.output.extent := by
    rw [ready.storage.outputSize]
    simpa [FPGrid.address,Geometry.outputOffset] using Geometry.output_inside h p ⟨0,by decide⟩
  have inOutput := Storage.address_inside ready.storage.output (FPGrid.address h p) inside
  change ready.storage.output.Contains (PhysicalGrid.address h ready.storage p) at inOutput
  rw [←same] at inOutput
  exact (fresh address owned).2.2.1 inOutput


#axiomsClean caller_frame
#axiomsClean output_bound
#axiomsClean source_loads
end VeriTile.Bench.Repairs.BgmvExpandSlice.Preparation
