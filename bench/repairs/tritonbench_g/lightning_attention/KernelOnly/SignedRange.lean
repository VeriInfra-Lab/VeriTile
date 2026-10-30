import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Parts
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
/-- Mirrored signed source ranges execute in their original descending order.
The invariant is indexed by the number of completed iterations. -/
theorem signed_range_invariant (M : Algebra α) (idx : RegName)
    (body : List ComputeStmt) (count : Nat) (I : Nat → State α → Prop)
    (bodyStep : ∀ n, n<count → ∀ s, I n s → ∃ t,
      run M body (s.setReg idx .int [] (fun _ => (n : Int)+1-count))=some t ∧ I (n+1) t)
    (n : Nat) (hn : n≤count) (s : State α) (hs : I n s) :
    ∃ t, rangeInt M idx ((n : Int)+1-count) 1 1 body s=some t ∧ I count t := by
  have go : ∀ remaining n, count-n=remaining → n≤count → ∀ s, I n s →
      ∃ t, rangeInt M idx ((n : Int)+1-count) 1 1 body s=some t ∧ I count t := by
    intro remaining
    induction remaining with
    | zero =>
      intro n he hn s hs
      have en : n=count := by omega
      subst n
      refine ⟨s,?_,hs⟩
      rw [rangeInt]
      simp
    | succ r ih =>
      intro n he hn s hs
      have hb : n<count := by omega
      obtain ⟨u,hu,hi⟩ := bodyStep n hb s hs
      obtain ⟨t,ht,hf⟩ := ih (n+1) (by omega) (by omega) u hi
      refine ⟨t,?_,hf⟩
      rw [rangeInt]
      have hc : (n : Int)+1-count<1 := by omega
      simp only [Nat.one_ne_zero,if_false,hc,if_true,hu,bind,Option.bind_some]
      convert ht using 1 <;> push_cast <;> ring
  exact go (count-n) n rfl hn s hs

theorem run_single_reverse (M : Algebra α) (idx : RegName) (n : Nat)
    (body : List ComputeStmt) (s t : State α)
    (h : rangeInt M idx (1-(n : Int)) 1 1 body s=some t) :
    run M [.forRangeInt idx (reverseStart n) (.constInt 1) (.constNat 1) body] s=some t := by
  simpa [run,step,reverseStart,evalOp_unfold,numeric,bop,Tile.bop,
    Broadcast.leftIndex,Broadcast.rightIndex] using h

#axiomsClean run_single_reverse
#axiomsClean signed_range_invariant
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
