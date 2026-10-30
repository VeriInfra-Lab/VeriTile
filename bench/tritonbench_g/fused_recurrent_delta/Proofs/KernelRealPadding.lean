import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
open VeriTile.Triton FP.Structural KernelRegisters KernelBackwardRegisters
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 14000000

def padVec {K L : Nat} (x : Math.Vec K) (i : Fin L) : ℝ := if hi : i.val<K then x ⟨i.val,hi⟩ else 0
def padInput {K : Nat} (L : Nat) (a : Math.Input K) : Math.Input L :=
  ⟨fun t => padVec (a.q t),fun t => padVec (a.k t),a.x,a.beta,padVec a.h0⟩
def padGrad {K : Nat} (L : Nat) (z : Math.Grad K) : Math.Grad L :=
  ⟨padVec z.q,padVec z.k,z.x,z.beta,padVec z.h⟩
theorem pad_sum {K L : Nat} (covered : K≤L) (x : Math.Vec K) :
    (∑ i : Fin L,padVec x i)=∑ i,x i := by
  rw [Finset.sum_fin_eq_sum_range]
  calc
    _ = ∑ i∈Finset.range L,if hi : i<K then x ⟨i,hi⟩ else 0 := by
      apply Finset.sum_congr rfl
      intro i hi
      simp [Finset.mem_range.mp hi,padVec]
    _ = ∑ i∈Finset.range K,if hi : i<K then x ⟨i,hi⟩ else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono covered)
      intro i _ outside
      simp only [Finset.mem_range] at outside
      simp [outside]
    _ = _ := (Finset.sum_fin_eq_sum_range x).symm
theorem pad_dot {K L : Nat} (covered : K≤L) (x y : Math.Vec K) :
    Math.dot (padVec (L:=L) x) (padVec y)=Math.dot x y := by
  have terms : (fun i : Fin L => padVec x i*padVec y i)=padVec (fun i => x i*y i) := by
    funext i
    by_cases hi : i.val<K <;> simp [padVec,hi]
  simp only [Math.dot]
  rw [terms,pad_sum covered]
theorem pad_next {K L : Nat} (covered : K≤L) (h k : Math.Vec K) (x beta : ℝ) :
    Math.next (padVec (L:=L) h) (padVec k) x beta=padVec (Math.next h k x beta) := by
  unfold Math.next Math.residual
  rw [pad_dot covered]
  funext i
  by_cases hi : i.val<K <;> simp [padVec,hi]
theorem pad_state {K L : Nat} (covered : K≤L) (a : Math.Input K) (n : Nat) :
    Math.state (padInput L a) n=padVec (Math.state a n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Math.state,ih]
    simp only [padInput,Math.state]
    exact pad_next covered _ _ _ _
theorem pad_output {K L : Nat} (covered : K≤L) (scale : ℝ) (a : Math.Input K) (t : Nat) :
    Math.output scale (padInput L a) t=Math.output scale a t := by
  unfold Math.output
  rw [pad_state covered]
  change scale*Math.dot (padVec (Math.state a (t+1))) (padVec (a.q t))=_
  rw [pad_dot covered]

theorem pad_pullback {K L : Nat} (covered : K≤L) (scale : ℝ) (h q k : Math.Vec K)
    (x beta g : ℝ) (G : Math.Vec K) :
    Math.pullback scale (padVec (L:=L) h) (padVec q) (padVec k) x beta g (padVec G)=
      padGrad L (Math.pullback scale h q k x beta g G) := by
  have e : (fun i : Fin L => padVec G i+scale*g*padVec q i)=padVec (fun i => G i+scale*g*q i) := by
    funext i
    by_cases hi : i.val<K <;> simp [padVec,hi]
  simp only [Math.pullback,Math.residual,e,pad_dot covered,pad_next covered,padGrad]
  congr 1 <;> funext i <;> by_cases hi : i.val<K <;> simp [padVec,hi]
theorem pad_reverse {K L : Nat} (covered : K≤L) (scale : ℝ) (a : Math.Input K)
    (g : Nat → ℝ) (terminal : Math.Vec K) (T n : Nat) :
    Math.reverse scale (padInput L a) g (padVec terminal) T n=padVec (Math.reverse scale a g terminal T n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [Math.reverse,pad_state covered,ih]
    change (Math.pullback scale (padVec (Math.state a (T-1-n))) (padVec (a.q (T-1-n)))
      (padVec (a.k (T-1-n))) (a.x (T-1-n)) (a.beta (T-1-n)) (g (T-1-n))
      (padVec (Math.reverse scale a g terminal T n))).h=_
    rw [pad_pullback covered]
    rfl
theorem pad_gradient {K L : Nat} (covered : K≤L) (scale : ℝ) (a : Math.Input K)
    (g : Nat → ℝ) (terminal : Math.Vec K) (T t : Nat) :
    Math.gradient scale (padInput L a) g (padVec terminal) T t=padGrad L (Math.gradient scale a g terminal T t) := by
  unfold Math.gradient Math.carry
  rw [pad_state covered,pad_reverse covered]
  exact pad_pullback covered scale _ _ _ _ _ _ _
theorem pad_loss {K L : Nat} (covered : K≤L) (scale : ℝ) (a : Math.Input K)
    (g : Nat → ℝ) (terminal : Math.Vec K) (T : Nat) :
    Math.loss scale (padInput L a) g (padVec terminal) T=Math.loss scale a g terminal T := by
  simp only [Math.loss,pad_output covered,pad_state covered,pad_dot covered]

/-- The padded source column is exactly the zero extension of the K-dimensional
mathematical input. No key-padding terms are omitted from the native model. -/
def unpaddedColumn (s : Shape) (f : Flags) (bh v : Nat) (u : State ℝ) : Math.Input s.K where
  q := fun t i => (u.mem "q" (keyAddress s bh t i.val)).read .real
  k := fun t i => (u.mem "k" (keyAddress s bh t i.val)).read .real
  x := fun t => (u.mem "v" (bh*(s.T*s.V)+v+t*s.V)).read .real
  beta := fun t => if f.vectorBeta then (u.mem "beta" (bh*(s.T*s.V)+v+t*s.V)).read .real
    else (u.mem "beta" (betaAddress s bh t)).read .real
  h0 := fun i => if f.initial then (u.mem "h0" (bh*s.K*s.V+i.val*s.V+v)).read .real else 0
theorem column_padding (s : Shape) (f : Flags) (bh iv : Nat) (u : State ℝ) (j : Fin s.FBV)
    (active : iv*s.FBV+j.val<s.V) :
    column s f bh iv u j=padInput s.BK (unpaddedColumn s f bh (iv*s.FBV+j.val) u) := by
  unfold column padInput unpaddedColumn
  congr 1
  · simp [active,valAddress,Nat.add_assoc]
  · simp [active,valAddress,Nat.add_assoc]
  · funext i
    cases hf : f.initial <;> by_cases hi : i.val<s.K <;>
      simp [padVec,hf,hi,active,stateAddress,Nat.add_assoc]

#axiomsClean pad_dot
#axiomsClean pad_state
#axiomsClean pad_gradient
#axiomsClean pad_loss
#axiomsClean column_padding
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelReal
