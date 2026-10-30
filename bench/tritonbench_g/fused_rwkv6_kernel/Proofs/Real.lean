import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Host
import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.RealReference
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 8000000
attribute [local instance] Classical.propDecidable

theorem tree_real_list (t : Tree) (f : Nat → ℝ) :
    t.eval 0 (·+·) f=(t.leaves.map f).sum := by
  induction t with
  | zero => rfl
  | leaf i => simp [Tree.eval,Tree.leaves]
  | add a b ha hb => simp [Tree.eval,Tree.leaves,ha,hb,List.sum_append]

theorem list_range_sum (n : Nat) (f : Nat → ℝ) :
    ((List.range n).map f).sum=∑ i∈Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,List.map_append,List.sum_append,Finset.sum_range_succ,ih]

theorem tree_real (t : Tree) (n : Nat) (h : t.legal n) (f : Nat → ℝ) :
    t.eval 0 (·+·) f=∑ i∈Finset.range n, f i := by
  rw [tree_real_list]
  exact (List.Perm.sum_eq (h.map f)).trans (list_range_sum n f)

theorem sum_blocks (I J : Nat) (f : Nat → ℝ) :
    (∑ i∈Finset.range I, ∑ j∈Finset.range J, f (i*J+j))=
      ∑ e∈Finset.range (I*J),f e := by
  induction I with
  | zero => simp
  | succ I ih =>
    rw [Finset.sum_range_succ,ih]
    simp only [Nat.succ_mul,Finset.sum_range_add]

theorem sum_pad (K N : Nat) (h : K≤N) (f : Nat → ℝ) :
    (∑ e∈Finset.range N, if e<K then f e else 0)=∑ e∈Finset.range K,f e := by
  have hz : ∀ e∈Finset.range N, e∉Finset.range K → (if e<K then f e else 0)=0 := by
    intro e _ he
    simp only [Finset.mem_range] at he
    simp [he]
  symm
  calc
    (∑ e∈Finset.range K,f e) = ∑ e∈Finset.range K,if e<K then f e else 0 := by
      apply Finset.sum_congr rfl
      intro e he
      simp [Finset.mem_range.1 he]
    _ = _ := Finset.sum_subset (Finset.range_mono h) hz

theorem scalarOutput_real (c : Shape) (hc : Sizes c) (plan : Plans c) (x : Input ℝ)
    (scale : ℝ) (z t f : Nat) (hf : f<c.V) :
    scalarOutput realOps c plan x scale z t f=realOutput c x scale z (visit c t) f := by
  unfold scalarOutput
  change (plan.host z t f).eval 0 (·+·) (fun i =>
    scalarPartial realOps c plan x scale z i (visit c t) f) = _
  rw [tree_real _ c.NK (plan.hostLegal z t f)]
  have hk : ∀ i, scalarPartial realOps c plan x scale z i (visit c t) f=
      ∑ j∈Finset.range c.BK,scalarTerm realOps c x scale z (visit c t) (i*c.BK+j) f := by
    intro i
    exact tree_real _ c.BK (plan.keyLegal z i (visit c t) f) _
  simp only [hk]
  rw [sum_blocks c.NK c.BK (fun e => scalarTerm realOps c x scale z (visit c t) e f)]
  simp_rw [scalar_real_term c x scale z (visit c t) _ f hf]
  rw [sum_pad c.K (c.NK*c.BK) (ceil_cover c.K c.BK hc.BK)]
  rfl

/-- Complete mathematical returned tensor, independent dense sum and recurrence. -/
theorem real_returned (c : Shape) (hc : Sizes c) (plan : Plans c) (scale : ℝ)
    (m : Memory ℝ) (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    forward realOps c plan scale m .result (resultAddress c z t f)=
      realOutput c (inputs c m) scale z (visit c t) f := by
  rw [returned_value realOps c hc plan scale m z t f hz ht hf]
  exact scalarOutput_real c hc plan (inputs c m) scale z t f hf

theorem real_terminal (c : Shape) (hc : Sizes c) (plan : Plans c) (scale : ℝ)
    (m : Memory ℝ) (hs : c.final=true) (z e f : Nat)
    (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) :
    forward realOps c plan scale m .ht (stateAddress c z e f)=realState c (inputs c m) z e f c.T := by
  rw [terminal_value realOps c hc plan scale m hs z e f hz he hf]
  exact scalar_real_state c (inputs c m) z e f c.T he hf

/-- An explicit closed-state invariant, independent of tiles and imperative execution. -/
def stateClosed (c : Shape) (x : Input ℝ) (z e f n : Nat) : ℝ :=
  (if c.initial then x.h0 z e f else 0) *
    (∏ j∈Finset.range n, Real.exp (x.w z (visit c j) e)) +
  ∑ i∈Finset.range n, (x.k z (visit c i) e * x.v z (visit c i) f) *
    (∏ j∈Finset.Ico (i+1) n,Real.exp (x.w z (visit c j) e))

theorem zero_time_state (c : Shape) (x : Input ℝ) (z e f : Nat) :
    realState c x z e f 0=if c.initial then x.h0 z e f else 0 := rfl

#axiomsClean real_returned
#axiomsClean real_terminal
#specNonCircular realState avoiding [runLoop,runProgram,runGrid,forward]
#specNonCircular realOutput avoiding [runLoop,runProgram,runGrid,forward]
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
