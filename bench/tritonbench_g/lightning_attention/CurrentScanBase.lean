import bench.tritonbench_g.lightning_attention.CurrentReadback
import bench.tritonbench_g.lightning_attention.RealPartitions
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem fold_invariant {β : Type} (f : Nat → β → β) (P : β → Prop) (n : Nat) (a : β)
    (init : P a) (step : ∀ k,k<n → ∀ b,P b → P (f k b)) : P (foldRange f 0 n 1 a) := by
  induction n with
  | zero => simpa only [foldRange_zero] using init
  | succ n ih => rw [foldRange_succ];exact step n (by omega) _ (ih (fun k hk => step k (by omega)))

/-- A source fold updates this observed cell exactly once. Both the preserved
input invariant and the original cell value are propagated through the fold. -/
theorem fold_once {β γ : Type} (f : Nat → β → β) (get : β → γ) (P : β → Prop)
    (H : γ → γ) (stop hit : Nat) (a : β) (init : P a)
    (step : ∀ k,k<stop → ∀ b,P b → P (f k b))
    (atHit : ∀ b,P b → get (f hit b)=H (get b))
    (away : ∀ k,k<stop → k≠hit → ∀ b,P b → get (f k b)=get b) :
    get (foldRange f 0 stop 1 a)=(if hit<stop then H (get a) else get a) := by
  have aux : ∀ n,n≤stop → get (foldRange f 0 n 1 a)=(if hit<n then H (get a) else get a) := by
    intro n
    induction n with
    | zero => intro hn;simp only [foldRange_zero,Nat.not_lt_zero,ite_false]
    | succ n ih =>
      intro hn
      have hP := fold_invariant f P n a init (fun k hk => step k (by omega))
      rw [foldRange_succ]
      by_cases he : n=hit
      · subst n
        rw [atHit _ hP,ih (by omega)]
        simp
      · rw [away n (by omega) he _ hP,ih (by omega)]
        have hc : hit<n+1 ↔ hit<n := by omega
        simp only [hc]
  exact aux stop le_rfl

theorem subG_inputs (M : Algebra α) (p : Parameters) (head row : Nat)
    (m original : Memory α) (h : Inputs m original) : subG M p head row m=subG M p head row original := by
  unfold subG read
  rw [h "incoming" (by decide)]
theorem subK_inputs (M : Algebra α) (p : Parameters) (head row : Nat)
    (m original : Memory α) (h : Inputs m original) : subK M p head row m=subK M p head row original := by
  unfold subK read
  rw [h "key" (by decide)]
theorem subVT_inputs (M : Algebra α) (p : Parameters) (head row : Nat)
    (m original : Memory α) (h : Inputs m original) : subVT M p head row m=subVT M p head row original := by
  unfold subVT read
  rw [h "value" (by decide)]

theorem query_region_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.E p.D) (m : Memory α) (region : RegionName)
    (address : Nat) (hr : region≠"queryGradient") :
    queryUpdate M D p head row carry m region address=m region address := by
  apply writeRows_frame
  exact Or.inl hr

theorem keyValue_region_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α) (region : RegionName)
    (address : Nat) (hk : region≠"keyGradient") (hv : region≠"valueGradient") :
    keyValueUpdate M D p head row carry m region address=m region address := by
  unfold keyValueUpdate
  rw [writeRows_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl hv),
    writeTrans_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl hk)]

theorem query_time_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.E p.D) (m : Memory α) (time : Nat) (a : Fin p.D)
    (safe : ∀ lane : Fin p.CB,row+lane.val≠time) :
    queryUpdate M D p head row carry m "queryGradient" (head*p.N*p.D+time*p.D+a.val)=
      m "queryGradient" (head*p.N*p.D+time*p.D+a.val) :=
  writeRows_time_frame _ _ _ _ _ _ _ _ _ _ _ safe

theorem key_time_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α) (time : Nat) (a : Fin p.D)
    (safe : ∀ lane : Fin p.CB,row+lane.val≠time) :
    keyValueUpdate M D p head row carry m "keyGradient" (head*p.N*p.D+time*p.D+a.val)=
      m "keyGradient" (head*p.N*p.D+time*p.D+a.val) := by
  unfold keyValueUpdate
  rw [writeRows_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact writeTrans_time_frame _ _ _ _ _ _ _ _ _ _ _ safe

theorem value_time_frame (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row : Nat) (carry : Matrix α p.D p.E) (m : Memory α) (time : Nat) (c : Fin p.E)
    (safe : ∀ lane : Fin p.CB,row+lane.val≠time) :
    keyValueUpdate M D p head row carry m "valueGradient" (head*p.N*p.E+time*p.E+c.val)=
      m "valueGradient" (head*p.N*p.E+time*p.E+c.val) := by
  unfold keyValueUpdate
  rw [writeRows_time_frame _ _ _ _ _ _ _ _ _ _ _ safe]
  exact writeTrans_frame _ _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))

theorem forward_sub_collision (NC CB n s l b j k : Nat)
    (hs : s<NC) (hj : j<NC) (hl : l<CB) (hk : k<CB)
    (he : (n*NC+s)*CB+l=(b*NC+j)*CB+k) : n=b ∧ s=j ∧ l=k := by
  have h := row_feature_collision CB _ _ _ _ hl hk he
  have q := row_feature_collision NC _ _ _ _ hs hj h.1
  exact ⟨q.1,q.2,h.2⟩

theorem reverse_sub_collision (p : Parameters) (htile : p.NC*p.CB=p.BT)
    (n s l b j k : Nat) (hn : n<p.NT) (hb : b<p.NT)
    (hs : s<p.NC) (hj : j<p.NC) (hl : l<p.CB) (hk : k<p.CB)
    (he : p.NT*p.BT-(n*p.NC+s+1)*p.CB+l=p.NT*p.BT-(b*p.NC+j+1)*p.CB+k) :
    n=b ∧ s=j ∧ l=k := by
  rw [← htile,Real.reverse_row _ _ _ n s hn hs,Real.reverse_row _ _ _ b j hb hj] at he
  have he' : ((p.NT-1-n)*p.NC+(p.NC-1-s))*p.CB+l=
      ((p.NT-1-b)*p.NC+(p.NC-1-j))*p.CB+k := by
    simpa only [Nat.add_mul,Nat.mul_assoc] using he
  have h := forward_sub_collision p.NC p.CB _ _ _ _ _ _ (by omega) (by omega) hl hk he'
  exact ⟨by omega,by omega,h.2.2⟩

#axiomsClean fold_once
#axiomsClean forward_sub_collision
#axiomsClean reverse_sub_collision
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Current
