import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Correct
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.FPEquiv
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelIdealReadback

/-! Full original checked Tensor and physical-storage applications. The raw
source and recurrence lemmas are shared with the primary kernel interfaces. -/
namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.TensorBridge
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor KernelReadback
noncomputable section

theorem real (realRead : Nat → ℝ) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State (Option ℝ)) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Reference.ideal c order (KernelReal.entry realRead c heap other)=some after ∧
      (∀ b head d,b < Checked.batches c.val → head < Checked.heads c.val → d < Checked.width c.val →
        KernelIdealReadback.output realRead c after (t.output.tensor.storage.address
          (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d)))=
          some (KernelReal.target realRead c heap (b*Checked.heads c.val+head) d)) ∧
      (∀ a,a < base c ∨ base c+4*size c ≤ a → KernelIdealReadback.output realRead c after a=some (realRead a)) ∧
      (∀ o∈[c.val.mid,c.val.log,c.val.lengths],∀ a,
        o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements →
        KernelIdealReadback.output realRead c after a=some (realRead a)) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelReal.entry realRead c heap other).mem r i) :=
  KernelIdealReadback.real realRead c t heap other h order all

theorem native (p : KernelNative.Ops) (S : Native.Schedule) (c : Call) (t : Arguments c)
    (heap : Heap) (other : State Native.Term) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Native.algebra c order (KernelNative.entry c heap other)=some after ∧
      (∀ b head d,b < Checked.batches c.val → head < Checked.heads c.val → d < Checked.width c.val →
        VeriTile.Torch.FP32.logicalValue (readWord (execute p S c heap other h.metadata order))
          t.output.tensor t.output.laws ((b*Checked.heads c.val+head)*Checked.width c.val+d)=
        (⟨KernelNative.target p S c heap (b*Checked.heads c.val+head) d⟩ : VeriTile.Torch.FP32.Word)) ∧
      (∀ a,a < base c ∨ base c+4*size c ≤ a → execute p S c heap other h.metadata order a=heap a) ∧
      (∀ o∈[c.val.mid,c.val.log,c.val.lengths],∀ a,
        o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements →
        execute p S c heap other h.metadata order a=heap a) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ size c ≤ i → after.mem r i=(KernelNative.entry c heap other).mem r i) :=
  KernelReadback.native p S c t heap other h order all

theorem source (c : Call) :
    FlashDecode2.kernel c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) =
    phi (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) := KernelModel.source c

theorem empty {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (before : State α) (h : Checked.batches c.val = 0 ∨ Checked.heads c.val = 0) :
    run A c (List.range (rows c)) before = some before := KernelModel.empty A c before h

#axiomsClean real
#axiomsClean native
#axiomsClean source
#axiomsClean empty
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.TensorBridge
