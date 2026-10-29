import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelReal
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelEndpoints
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

theorem real (interpret : Nat → ℝ) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State (Option ℝ)) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Reference.ideal c order (KernelReal.entry interpret c heap other) = some after ∧
      (∀ b head d,b<Checked.batches c.val → head<Checked.heads c.val → d<Checked.width c.val →
        (after.mem "O" (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d))).read .real =
          some (KernelReal.target interpret c heap (b*Checked.heads c.val+head) d)) ∧
      (∀ r i,r≠("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelReal.entry interpret c heap other).mem r i) := by
  obtain ⟨after,executed,values,frame⟩ := typed Reference.ideal c order all (KernelReal.entry interpret c heap other)
  refine ⟨after,executed,?_,frame⟩
  intro b head d hb hh hd
  have hr := List.mem_range.mp (coverage c b head hb hh).1
  have bound : (b*Checked.heads c.val+head)*Checked.width c.val+d<size c := by unfold size; nlinarith
  rw [output_index c t heap h _ bound]
  let k : TileIndex [Checked.block c.val] := (⟨d,by rw [block_eq c]; exact hd⟩,PUnit.unit)
  have read := values _ k hr
  rw [KernelReal.value interpret c heap other h _ hr k] at read
  change after.mem "O" ((b*Checked.heads c.val+head)*Checked.width c.val+d)=_ at read
  rw [read,Cell.read_same]

theorem native (p : KernelNative.Ops) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State Native.Term) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Native.algebra c order (KernelNative.entry c heap other) = some after ∧
      (∀ b head d,b<Checked.batches c.val → head<Checked.heads c.val → d<Checked.width c.val →
        Native.observe (KernelNative.lower p) (KernelNative.schedule c) after "O"
          (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d)) =
        KernelNative.target p c (lengths c heap) (word c heap "E") (word c heap "M")
          (b*Checked.heads c.val+head) d) ∧
      (∀ r i,r≠("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelNative.entry c heap other).mem r i) := by
  obtain ⟨after,executed,values,frame⟩ := typed Native.algebra c order all (KernelNative.entry c heap other)
  refine ⟨after,executed,?_,frame⟩
  intro b head d hb hh hd
  have hr := List.mem_range.mp (coverage c b head hb hh).1
  have bound : (b*Checked.heads c.val+head)*Checked.width c.val+d<size c := by unfold size; nlinarith
  rw [output_index c t heap h _ bound]
  let k : TileIndex [Checked.block c.val] := (⟨d,by rw [block_eq c]; exact hd⟩,PUnit.unit)
  have read := values _ k hr
  change after.mem "O" ((b*Checked.heads c.val+head)*Checked.width c.val+d)=_ at read
  unfold Native.observe
  rw [read,Cell.read_same]
  exact KernelNative.value p c heap other h _ hr k

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelEndpoints
