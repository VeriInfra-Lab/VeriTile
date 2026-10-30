import bench.tritonbench_g.lightning_attention.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem append (M : Algebra α) (bounds : RegionName → Nat) (first rest : List ComputeStmt)
    (s t : State α) (safe : Safety.traceSafeList M bounds first s)
    (ran : run M first s = some t) (tail : Safety.traceSafeList M bounds rest t) :
    Safety.traceSafeList M bounds (first ++ rest) s := by
  rw [Safety.traceSafeList_append]
  refine ⟨safe,?_⟩
  intro u hu
  have same : u = t := Option.some.inj (hu.symm.trans ran)
  subst u
  exact tail

theorem single_range (M : Algebra α) (bounds : RegionName → Nat) (idx : RegName)
    (stop : Nat) (body : List ComputeStmt) (s : State α)
    (safe : Safety.rangeSafe M bounds idx 0 stop 1 body s) :
    Safety.traceSafeList M bounds
      [.forRangeDyn idx (.constNat 0) (.constNat stop) (.constNat 1) body] s := by
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    evalOp_unfold,true_and]
  refine ⟨safe,?_⟩
  cases step M (.forRangeDyn idx (.constNat 0) (.constNat stop) (.constNat 1) body) s <;> trivial

theorem single_reverse (M : Algebra α) (bounds : RegionName → Nat) (idx : RegName)
    (count : Nat) (body : List ComputeStmt) (s : State α)
    (safe : Safety.rangeIntSafe M bounds idx (1-(count : Int)) 1 1 body s) :
    Safety.traceSafeList M bounds
      [.forRangeInt idx (reverseStart count) (.constInt 1) (.constNat 1) body] s := by
  apply Safety.traceSafeList_cons_intro
  · have index : (0 : Int) - ((count : Int)-1) = 1-count := by omega
    simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,reverseStart,evalOp_unfold,
      numeric,bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,index] using safe
  · intro t ran
    exact Safety.traceSafeList_nil _ _ _

/-- The original mirrored signed iterator, indexed by completed iterations.
This checks each body at its actual signed register value and successor state. -/
theorem signed_range (M : Algebra α) (bounds : RegionName → Nat) (idx : RegName)
    (body : List ComputeStmt) (count : Nat) (I : Nat → State α → Prop)
    (safeBody : ∀ n, n<count → ∀ s, I n s →
      Safety.traceSafeList M bounds body (s.setReg idx .int [] (fun _ => (n : Int)+1-count)))
    (preserved : ∀ n, n<count → ∀ s, I n s → ∀ t,
      run M body (s.setReg idx .int [] (fun _ => (n : Int)+1-count))=some t → I (n+1) t)
    (n : Nat) (hn : n≤count) (s : State α) (hs : I n s) :
    Safety.rangeIntSafe M bounds idx ((n : Int)+1-count) 1 1 body s := by
  have go : ∀ remaining n, count-n=remaining → n≤count → ∀ s, I n s →
      Safety.rangeIntSafe M bounds idx ((n : Int)+1-count) 1 1 body s := by
    intro remaining
    induction remaining with
    | zero =>
      intro n he hn s hs
      have en : n=count := by omega
      subst n
      rw [Safety.rangeIntSafe_unfold]
      simp
    | succ r ih =>
      intro n he hn s hs
      have hb : n<count := by omega
      have hc : (n : Int)+1-count<1 := by omega
      rw [Safety.rangeIntSafe_unfold]
      simp only [Nat.one_ne_zero,if_false,hc,if_true]
      refine ⟨safeBody n hb s hs,?_⟩
      cases ran : run M body (s.setReg idx .int [] (fun _ => (n : Int)+1-count)) with
      | none => trivial
      | some t =>
        have next := ih (n+1) (by omega) (by omega) t (preserved n hb s hs t ran)
        convert next using 1 <;> push_cast <;> ring
  exact go (count-n) n rfl hn s hs
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
