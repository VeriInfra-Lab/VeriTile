import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Storage

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Physical
open Geometry
noncomputable section
open Classical

def outputAddress {h : Call} (s : Storage.Placement h)
    (i : Fin h.batches × Fin h.N) := s.output.address (outputOffset h i.1.val i.2.val)

theorem output_injective {h : Call} (s : Storage.Placement h) :
    Function.Injective (outputAddress s) := by
  intro i j same
  have off := VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.address_injective s.output _ _ same
  obtain ⟨b,n⟩ := Geometry.output_injective h i.1.val j.1.val
    ⟨i.2.val,i.2.isLt.trans_le h.rowsCovered⟩ ⟨j.2.val,j.2.isLt.trans_le h.rowsCovered⟩ off
  exact Prod.ext (Fin.ext b) (Fin.ext (congrArg (fun n : Fin h.BN => n.val) n))

/-- Address transport of already computed cell values. The local runtime
contract stores a cell at its generated physical address; this definition
does not assume the cell's numerical value or source execution. -/
def place {α : Type} {h : Call} (s : Storage.Placement h) (indices : Nat → Int)
    (before : Nat → α) (values : Fin h.batches → Fin h.N → α) : Nat → α :=
  fun addr => if found : ∃ i : Fin h.batches × Fin h.N,
      indices i.1.val≠-1 ∧ addr=outputAddress s i then
    let i := Classical.choose found
    values i.1 i.2
  else before addr

theorem readback {α : Type} {h : Call} (s : Storage.Placement h) (indices : Nat → Int)
    (before : Nat → α) (values : Fin h.batches → Fin h.N → α)
    (b : Fin h.batches) (n : Fin h.N) (active : indices b.val≠-1) :
    place s indices before values (outputAddress s (b,n))=values b n := by
  let found : ∃ i : Fin h.batches × Fin h.N,indices i.1.val≠-1 ∧
      outputAddress s (b,n)=outputAddress s i := ⟨(b,n),active,rfl⟩
  have eq := output_injective s (Classical.choose_spec found).2
  unfold place
  rw [dif_pos found,←eq]

theorem frame {α : Type} {h : Call} (s : Storage.Placement h) (indices : Nat → Int)
    (before : Nat → α) (values : Fin h.batches → Fin h.N → α) (addr : Nat)
    (outside : ¬s.output.Contains addr) : place s indices before values addr=before addr := by
  unfold place
  rw [dif_neg]
  rintro ⟨i,_,same⟩
  have inside : outputOffset h i.1.val i.2.val<s.output.extent := by
    rw [s.outputSize]
    exact Geometry.output_inside h ⟨i.1,⟨0,by decide⟩⟩ ⟨i.2.val,i.2.isLt.trans_le h.rowsCovered⟩
  have contained := VeriTile.Bench.Repairs.BgmvExpandSlice.Storage.address_inside s.output _ inside
  exact outside (same ▸ contained)

#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Physical
