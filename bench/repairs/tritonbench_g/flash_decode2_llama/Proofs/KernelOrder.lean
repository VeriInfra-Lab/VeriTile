import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelModel

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.KernelModel
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

theorem order {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (first second : List Nat) (firstAll : first.Perm (List.range (rows c)))
    (secondAll : second.Perm (List.range (rows c))) (before : State α) :
    ∃ a b,run A c first before=some a ∧ run A c second before=some b ∧ a.mem=b.mem := by
  obtain ⟨a,ha,va,fa⟩ := typed A c first firstAll before
  obtain ⟨b,hb,vb,fb⟩ := typed A c second secondAll before
  refine ⟨a,b,ha,hb,?_⟩
  funext r i
  by_cases region : r=("O" : RegionName)
  · subst r
    by_cases outside : size c ≤ i
    · rw [fa _ _ (Or.inr outside),fb _ _ (Or.inr outside)]
    · have inside : i<size c := by omega
      have pos : 0<Checked.width c.val := by
        apply Nat.pos_of_ne_zero
        intro zero
        simp [size,zero] at inside
      have row : i/Checked.width c.val<rows c := (Nat.div_lt_iff_lt_mul pos).mpr inside
      have laneBound := Nat.mod_lt i pos
      let k : TileIndex [Checked.block c.val] :=
        (⟨i%Checked.width c.val,by rw [block_eq c]; exact laneBound⟩,PUnit.unit)
      have index : i/Checked.width c.val*Checked.width c.val+lane k=i := by
        change i/Checked.width c.val*Checked.width c.val+i%Checked.width c.val=i
        simpa only [Nat.mul_comm] using Nat.div_add_mod i (Checked.width c.val)
      rw [←index,va _ k row,vb _ k row]
  · rw [fa _ _ (Or.inl region),fb _ _ (Or.inl region)]

#axiomsClean order
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.KernelModel
