import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelReadback

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelIdealReadback
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor KernelReadback
noncomputable section
set_option maxHeartbeats 8000000

/-- Physical-address interpretation of the ideal source output cells. This is
a derived output view, never a supplied equality or successful-result premise. -/
def output (realRead : Nat → ℝ) (c : Call) (after : State (Option ℝ)) (a : Nat) : Option ℝ :=
  if base c ≤ a ∧ a < base c+4*size c ∧ (a-base c)%4=0 then
    (after.mem "O" ((a-base c)/4)).read .real else some (realRead a)

theorem exterior (realRead : Nat → ℝ) (c : Call) (after : State (Option ℝ))
    (a : Nat) (outside : a < base c ∨ base c+4*size c ≤ a) :
    output realRead c after a=some (realRead a) := by
  have miss : ¬(base c ≤ a ∧ a < base c+4*size c ∧ (a-base c)%4=0) := by omega
  simp only [output,miss,ite_false]

theorem input_storage (realRead : Nat → ℝ) (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (after : State (Option ℝ))
    (o : Checked.Observation) (member : o∈[c.val.mid,c.val.log,c.val.lengths])
    (a : Nat) (inside : o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements) :
    output realRead c after a=some (realRead a) := by
  apply exterior
  by_cases positive : 0 < size c
  · have active : Checked.active c.val := by
      dsimp [size,rows] at positive
      constructor
      · by_contra bad
        have zero : Checked.batches c.val=0 := by omega
        simp [zero] at positive
      · by_contra bad
        have zero : Checked.heads c.val=0 := by omega
        simp [zero] at positive
    have apart := (h.launch active).2.1 o member
    have cap := (KernelSafety.facts c t h.metadata c.val.output (by simp [Checked.tensors])).capacity
    have shape := h.metadata.2.2.2.2.2.2.1
    have numel : c.val.output.metadata.numel=size c := by
      simp [VeriTile.Torch.TensorMetadata.numel,shape,size,rows,Nat.mul_assoc]
    rw [output_pointer c t h.metadata positive]
    unfold Checked.apart at apart
    omega
  · have empty : size c=0 := by omega
    simp only [empty,Nat.mul_zero,Nat.add_zero]
    omega

theorem real (realRead : Nat → ℝ) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State (Option ℝ)) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Reference.ideal c order (KernelReal.entry realRead c heap other)=some after ∧
      (∀ b head d,b < Checked.batches c.val → head < Checked.heads c.val → d < Checked.width c.val →
        output realRead c after (t.output.tensor.storage.address
          (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d)))=
          some (KernelReal.target realRead c heap (b*Checked.heads c.val+head) d)) ∧
      (∀ a,a < base c ∨ base c+4*size c ≤ a → output realRead c after a=some (realRead a)) ∧
      (∀ o∈[c.val.mid,c.val.log,c.val.lengths],∀ a,
        o.storage.base ≤ a ∧ a < o.storage.base+4*o.storage.elements →
        output realRead c after a=some (realRead a)) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelReal.entry realRead c heap other).mem r i) := by
  obtain ⟨after,ran,values,frame⟩ := KernelEndpoints.real realRead c t heap other h order all
  refine ⟨after,ran,?_,exterior realRead c after,input_storage realRead c t heap h after,frame⟩
  intro b head d hb hh hd
  have hr := List.mem_range.mp (coverage c b head hb hh).1
  have bound : (b*Checked.heads c.val+head)*Checked.width c.val+d < size c := by unfold size; nlinarith
  have pointer := output_pointer c t h.metadata (by omega)
  have storage : t.output.tensor.storage.base=base c := by
    rw [←t.outputObserved] at pointer
    exact pointer.symm
  rw [output_index c t heap h _ bound]
  simp only [VeriTile.Torch.FP32.Storage.address,storage]
  have inside : base c+4*((b*Checked.heads c.val+head)*Checked.width c.val+d) < base c+4*size c := by omega
  have div : 4*((b*Checked.heads c.val+head)*Checked.width c.val+d)/4=
      (b*Checked.heads c.val+head)*Checked.width c.val+d := by omega
  simp only [output,Nat.le_add_right,inside,Nat.add_sub_cancel_left,
    Nat.mul_mod_right,div,and_self,ite_true]
  have result := values b head d hb hh hd
  rwa [output_index c t heap h _ bound] at result

#axiomsClean real
#axiomsClean exterior
#specNonCircular output avoiding [KernelReal.target,KernelEndpoints.real]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelIdealReadback
