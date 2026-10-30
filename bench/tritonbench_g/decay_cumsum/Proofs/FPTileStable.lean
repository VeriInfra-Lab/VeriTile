import bench.tritonbench_g.decay_cumsum.Proofs.FPTile

namespace VeriTile.Bench.Repairs.DecayCumsum.FPTileStable
open VeriTile.Triton FP.Structural FPOps FPPrefix Geometry FPTile
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

theorem forward_congr (M : Algebra α) (g g' : Nat → α) (n : Nat)
    (eq : ∀ t, t<n → g t=g' t) : FPReference.forwardCarry M g n=FPReference.forwardCarry M g' n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [FPReference.forwardCarry]
    rw [ih (fun t ht => eq t (by omega)),eq n (by omega)]

theorem backward_congr (M : Algebra α) (span n : Nat) (bound : n≤span)
    (dqi dqo dki dko q k g dqi' dqo' dki' dko' q' k' g' : Nat → α)
    (ei : ∀ t, t<span → dqi t=dqi' t) (eo : ∀ t, t<span → dqo t=dqo' t)
    (ki : ∀ t, t<span → dki t=dki' t) (ko : ∀ t, t<span → dko t=dko' t)
    (eq : ∀ t, t<span → q t=q' t) (ek : ∀ t, t<span → k t=k' t) (eg : ∀ t, t<span → g t=g' t) :
    FPReference.backwardCarry M span dqi dqo dki dko q k g n=
      FPReference.backwardCarry M span dqi' dqo' dki' dko' q' k' g' n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    have tInside : span-1-n<span := by omega
    have lastInside : span-1<span := by omega
    simp only [FPReference.backwardCarry]
    rw [ih (by omega),ei _ tInside,eo _ tInside,ki _ tInside,ko _ tInside,
      eq _ tInside,ek _ tInside,eg _ tInside,eg _ lastInside]

theorem expected_stable (M : Algebra α) (kind : Kind) (h : Call) (p : Phase) (positive : 0<p.span)
    (prog : Program h p) (scale : ℝ) (u v : State α)
    (reads : ∀ r∈inputs kind, ∀ t, t<p.span → ∀ j : Fin h.BK,
      u.mem r (off h p prog t j)=v.mem r (off h p prog t j))
    (r : RegionName) (t : Fin p.span) (j : Fin h.BK) :
    expected M kind h p prog scale u r t.val j=expected M kind h p prog scale v r t.val j := by
  have rawEq (r : RegionName) (member : r∈inputs kind) (s : Nat) (hs : s<p.span) :
      raw h.BK r (off h p prog s) u (j,PUnit.unit)=raw h.BK r (off h p prog s) v (j,PUnit.unit) :=
    congrArg (Cell.read .real) (reads r member s hs j)
  have endpoint : p.span-1<p.span := by omega
  cases kind
  · apply congrArg (FPReference.stored M)
    apply forward_congr
    intro n hn
    exact rawEq "g" (by simp [inputs]) n (by omega)
  · unfold expected FPPrepare.qValue FPPrepare.kValue
    rw [rawEq "q" (by simp [inputs]) t.val t.isLt,rawEq "k" (by simp [inputs]) t.val t.isLt,
      rawEq "g" (by simp [inputs]) t.val t.isLt,rawEq "g" (by simp [inputs]) (p.span-1) endpoint]
  · unfold expected FPBackwardBase.output
    apply congrArg (FPReference.stored M)
    split
    · unfold FPBackwardBase.qValue
      rw [rawEq "dqi" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt,
        rawEq "dqo" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt,
        rawEq "g" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt]
    · split
      · unfold FPBackwardBase.kValue
        rw [rawEq "dki" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt,
          rawEq "dko" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt,
          rawEq "g" (by simp [inputs,FPBackwardBase.inputRegions]) (p.span-1) endpoint,
          rawEq "g" (by simp [inputs,FPBackwardBase.inputRegions]) t.val t.isLt]
      · apply backward_congr M p.span (p.span-t.val) (by omega)
        all_goals intro s hs; exact rawEq _ (by simp [inputs,FPBackwardBase.inputRegions]) s hs

#axiomsClean expected_stable
end
end VeriTile.Bench.Repairs.DecayCumsum.FPTileStable
