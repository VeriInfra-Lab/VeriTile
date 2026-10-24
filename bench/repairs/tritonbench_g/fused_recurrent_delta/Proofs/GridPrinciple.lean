import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Memory
import Mathlib.Data.List.Nodup
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.GridPrinciple
noncomputable section
set_option maxHeartbeats 5000000
def Agree {α : Type} (S : Addr → Prop) (m n : Memory α) := ∀ a,S a → m a=n a
theorem agree_trans {α : Type} {S : Addr → Prop} {a b c : Memory α}
    (h : Agree S a b) (k : Agree S b c) : Agree S a c := by
  intro p hp
  rw [h p hp,k p hp]
theorem fold_agree {α J : Type} (step : J → Memory α → Memory α) (S : Addr → Prop)
    (xs : List J) (m : Memory α) (h : ∀ j∈xs,∀ m,Agree S (step j m) m) :
    Agree S (xs.foldl (fun m j => step j m) m) m := by
  induction xs generalizing m with
  | nil => intro a _;rfl
  | cons j js ih =>
    exact agree_trans (ih _ (fun k hk => h k (by simp [hk]))) (h j (by simp) m)

/-- Unique writers may consume current memory, including their own scratch.
The local congruence and cross-program preservation premises are separately
proved from the concrete loads/stores before this lemma is instantiated. -/
theorem unique_writer {α J : Type} [DecidableEq J] (step : J → Memory α → Memory α)
    (S : J → Addr → Prop) (target : J) (a : Addr) (ha : S target a)
    (preserve : ∀ j,j≠target → ∀ m,Agree (S target) (step j m) m)
    (congruent : ∀ m n,Agree (S target) m n → step target m a=step target n a)
    (xs : List J) (nd : xs.Nodup) (mem : target∈xs) (initial : Memory α) :
    xs.foldl (fun m j => step j m) initial a=step target initial a := by
  have run : ∀ ys : List J,ys.Nodup → target∈ys → ∀ m,Agree (S target) m initial →
      ys.foldl (fun m j => step j m) m a=step target initial a := by
    intro ys
    induction ys with
    | nil => intro _ hm;simpa using hm
    | cons j ys ih =>
      intro hn hm m hc
      obtain ⟨hjn,hnd⟩:=List.nodup_cons.mp hn
      by_cases hj : j=target
      · subst j
        have hs:=fold_agree step (S target) ys (step target m)
          (fun k hk cur => preserve k (by intro he;subst k;exact hjn hk) cur)
        change (ys.foldl (fun m j => step j m) (step target m)) a=_
        rw [hs a ha,congruent m initial hc]
      · apply ih hnd (by simpa [List.mem_cons,Ne.symm hj] using hm) (step j m)
        exact agree_trans (preserve j hj m) hc
  exact run xs nd mem initial (fun _ _ => rfl)
theorem unique_writer_valid {α J : Type} [DecidableEq J] (step : J → Memory α → Memory α)
    (S : J → Addr → Prop) (valid : J → Prop) (target : J) (a : Addr) (ha : S target a)
    (preserve : ∀ j,valid j → j≠target → ∀ m,Agree (S target) (step j m) m)
    (congruent : ∀ m n,Agree (S target) m n → step target m a=step target n a)
    (xs : List J) (nd : xs.Nodup) (mem : target∈xs) (hv : ∀ j∈xs,valid j) (initial : Memory α) :
    xs.foldl (fun m j => step j m) initial a=step target initial a := by
  have run : ∀ ys : List J,ys.Nodup → target∈ys → (∀ j∈ys,valid j) → ∀ m,Agree (S target) m initial →
      ys.foldl (fun m j => step j m) m a=step target initial a := by
    intro ys
    induction ys with
    | nil => intro _ hm;simp at hm
    | cons j ys ih =>
      intro hn hm hys m hc
      obtain ⟨hjn,hnd⟩:=List.nodup_cons.mp hn
      by_cases hj : j=target
      · subst j
        have hs:=fold_agree step (S target) ys (step target m)
          (fun k hk cur => preserve k (hys k (by simp [hk])) (by intro he;subst k;exact hjn hk) cur)
        change (ys.foldl (fun m j => step j m) (step target m)) a=_
        rw [hs a ha,congruent m initial hc]
      · apply ih hnd (by simpa [List.mem_cons,Ne.symm hj] using hm)
          (fun k hk => hys k (by simp [hk])) (step j m)
        exact agree_trans (preserve j (hys j (by simp)) hj m) hc
  exact run xs nd mem hv initial (fun _ _ => rfl)

theorem jobs_nodup (BH NV : Nat) : (jobs BH NV).Nodup := by
  exact (List.nodup_range).product (List.nodup_range)

#axiomsClean unique_writer
#axiomsClean jobs_nodup
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.GridPrinciple
