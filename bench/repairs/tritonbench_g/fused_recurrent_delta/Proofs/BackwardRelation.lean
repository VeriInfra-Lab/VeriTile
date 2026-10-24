import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.WholeSource
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.GridPrinciple
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Tree
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardRelation
open GridPrinciple
noncomputable section
set_option maxHeartbeats 10000000
def ReadOnly : Region → Prop
  | .q | .k | .v | .beta | .h0 | .do_ | .dht => True
  | _ => False
def Protected (s : Shape) (job : Nat × Nat) (a : Addr) :=
  match a.region with
  | .q | .k | .v | .beta | .h0 | .do_ | .dht => True
  | .dq | .dk | .dbeta => a.plane=job.2*s.BH+job.1
  | .dv | .dh0 => a.plane=job.1 ∧ job.2*s.BBV≤a.col ∧ a.col<(job.2+1)*s.BBV
  | _ => False
theorem ro_protected (s : Shape) (job : Nat × Nat) (a : Addr) (hr : ReadOnly a.region) : Protected s job a := by
  cases h:a.region <;> simp_all [ReadOnly,Protected]
theorem row_agree {α : Type} (S : Addr → Prop) (p : Ops α) (m n : Memory α) (hc : Agree S m n)
    (r : Region) (plane t lo hi : Nat) (value : Nat → α) :
    Agree S (rowStore p m r plane t lo hi value) (rowStore p n r plane t lo hi value) := by
  intro a ha
  unfold rowStore write
  split
  · rfl
  · exact hc a ha
theorem state_agree {α : Type} (S : Addr → Prop) (s : Shape) (p : Ops α) (m n : Memory α)
    (hc : Agree S m n) (r : Region) (bh iv BV : Nat) (h : Nat → Nat → α) :
    Agree S (stateStore s p m r bh iv BV h) (stateStore s p n r bh iv BV h) := by
  intro a ha
  unfold stateStore write
  split
  · rfl
  · exact hc a ha
theorem key_ro {α : Type} (s : Shape) (p : Ops α) (job : Nat × Nat) (m n : Memory α)
    (hc : Agree (Protected s job) m n) (r : Region) (hr : ReadOnly r) (plane t : Nat) :
    keyLoad s p m r plane t=keyLoad s p n r plane t := by
  funext i
  unfold keyLoad load
  split
  · exact hc _ (ro_protected s job _ hr)
  · rfl
theorem val_ro {α : Type} (s : Shape) (p : Ops α) (job : Nat × Nat) (m n : Memory α)
    (hc : Agree (Protected s job) m n) (r : Region) (hr : ReadOnly r) (plane t iv BV : Nat) :
    valLoad s p m r plane t iv BV=valLoad s p n r plane t iv BV := by
  funext j
  unfold valLoad load
  split
  · exact hc _ (ro_protected s job _ hr)
  · rfl
theorem beta_ro {α : Type} (s : Shape) (f : Flags) (p : Ops α) (job : Nat × Nat) (m n : Memory α)
    (hc : Agree (Protected s job) m n) (plane t iv BV : Nat) :
    betaLoad s f p m plane t iv BV=betaLoad s f p n plane t iv BV := by
  funext j
  have hb : m ⟨.beta,plane,t,0⟩=n ⟨.beta,plane,t,0⟩:=hc _ (by trivial)
  simp only [betaLoad,val_ro s p job m n hc .beta (by trivial),hb]
theorem seed_ro {α : Type} (s : Shape) (p : Ops α) (job : Nat × Nat) (m n : Memory α)
    (hc : Agree (Protected s job) m n) (r : Region) (hr : ReadOnly r) (use : Bool) (bh iv BV : Nat) :
    seed s p m r use bh iv BV=seed s p n r use bh iv BV := by
  funext i j
  have hm : m ⟨r,bh,i,iv*BV+j⟩=n ⟨r,bh,i,iv*BV+j⟩:=hc _ (ro_protected s job _ hr)
  simp only [seed,load,hm]
theorem dk_agree {α : Type} (s : Shape) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (t : Nat) :
    keyLoad s p m .dk (iv*s.BH+bh) t=keyLoad s p n .dk (iv*s.BH+bh) t := by
  funext i
  unfold keyLoad load
  split
  · exact hc _ rfl
  · rfl
theorem dv_agree {α : Type} (s : Shape) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (t j : Nat) (hj : j<s.BBV) :
    valLoad s p m .dv bh t iv s.BBV j=valLoad s p n .dv bh t iv s.BBV j := by
  unfold valLoad load
  split
  · apply hc
    dsimp only [Protected]
    exact ⟨rfl,by omega,by rw [Nat.add_mul];omega⟩
  · rfl
def Rel {α : Type} (s : Shape) (job : Nat × Nat) (a b : Machine α) :=
  a.h=b.h ∧ Agree (Protected s job) a.mem b.mem

theorem reverse_step {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv t : Nat) (a b : Machine α) (hc : Rel s (bh,iv) a b) :
    Rel s (bh,iv) (reverseStep s f p g scale bh iv t a) (reverseStep s f p g scale bh iv t b) := by
  rcases a with ⟨ha,ma⟩
  rcases b with ⟨hb,mb⟩
  obtain ⟨hh,hc⟩:=hc
  dsimp at hh
  subst hb
  dsimp only [Rel,reverseStep]
  simp only [key_ro s p (bh,iv) ma mb hc .q (by trivial),key_ro s p (bh,iv) ma mb hc .k (by trivial),
    val_ro s p (bh,iv) ma mb hc .v (by trivial),val_ro s p (bh,iv) ma mb hc .do_ (by trivial),
    beta_ro s f p (bh,iv) ma mb hc]
  refine ⟨True.intro,?_⟩
  cases hbeta:f.vectorBeta <;> simp only [hbeta,Bool.false_eq_true,ite_false,ite_true]
    <;> apply row_agree <;> apply row_agree <;> apply row_agree <;> exact hc
theorem reverse_loop {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv n : Nat) (a b : Machine α) (hc : Rel s (bh,iv) a b) :
    Rel s (bh,iv) (reverseLoop s f p g scale bh iv n a) (reverseLoop s f p g scale bh iv n b) := by
  induction n with
  | zero => exact hc
  | succ n ih => exact reverse_step s f p g scale bh iv n _ _ ih

theorem recompute_step {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (bh iv t : Nat)
    (a b : Machine α) (hc : Rel s (bh,iv) a b) :
    Rel s (bh,iv) (recomputeStep s f p g scale bh iv t a) (recomputeStep s f p g scale bh iv t b) := by
  rcases a with ⟨ha,ma⟩
  rcases b with ⟨hb,mb⟩
  obtain ⟨hh,hc⟩:=hc
  dsimp at hh
  subst hb
  have corr (i : Nat) :
      p.sub (keyLoad s p ma .dk (iv*s.BH+bh) t i)
        ((g.correction s.BBV).eval p (fun j => p.mul (valLoad s p ma .dv bh t iv s.BBV j) (ha i j)))=
      p.sub (keyLoad s p mb .dk (iv*s.BH+bh) t i)
        ((g.correction s.BBV).eval p (fun j => p.mul (valLoad s p mb .dv bh t iv s.BBV j) (ha i j))) := by
    rw [dk_agree s p ma mb bh iv hc t]
    congr 1
    apply Tree.eval_congr_legal p _ _ hg
    intro j hj
    rw [dv_agree s p ma mb bh iv hc t j hj]
  dsimp only [Rel,recomputeStep]
  simp only [Recompute.key_after_store,Recompute.val_after_store,Recompute.beta_after_store,
    ne_eq,reduceCtorEq,not_false_eq_true]
  simp only [key_ro s p (bh,iv) ma mb hc .k (by trivial),val_ro s p (bh,iv) ma mb hc .v (by trivial),
    val_ro s p (bh,iv) ma mb hc .do_ (by trivial),beta_ro s f p (bh,iv) ma mb hc,corr]
  refine ⟨True.intro,?_⟩
  apply row_agree
  apply row_agree
  exact hc
theorem recompute_loop {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (bh iv n : Nat)
    (a b : Machine α) (hc : Rel s (bh,iv) a b) :
    Rel s (bh,iv) (recomputeLoop s f p g scale bh iv n a) (recomputeLoop s f p g scale bh iv n b) := by
  induction n with
  | zero => exact hc
  | succ n ih => exact recompute_step s f p g hg scale bh iv n _ _ ih

theorem program_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees)
    (hg : (g.correction s.BBV).Legal s.BBV) (scale : α) (bh iv : Nat) (m n : Memory α)
    (hc : Agree (Protected s (bh,iv)) m n) :
    Agree (Protected s (bh,iv)) (backwardProgram s f p g scale bh iv m) (backwardProgram s f p g scale bh iv n) := by
  let am : Machine α:=⟨seed s p m .dht f.terminalGrad bh iv s.BBV,m⟩
  let an : Machine α:=⟨seed s p n .dht f.terminalGrad bh iv s.BBV,n⟩
  have hr:=reverse_loop s f p g scale bh iv s.T am an
    ⟨seed_ro s p (bh,iv) m n hc .dht (by trivial) f.terminalGrad bh iv s.BBV,hc⟩
  let mid (a : Machine α) := if f.initialGrad then stateStore s p a.mem .dh0 bh iv s.BBV a.h else a.mem
  have hm : Agree (Protected s (bh,iv)) (mid (reverseLoop s f p g scale bh iv s.T am))
      (mid (reverseLoop s f p g scale bh iv s.T an)) := by
    dsimp only [mid]
    rw [hr.1]
    cases hf:f.initialGrad <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    · exact hr.2
    · exact state_agree _ s p _ _ hr.2 .dh0 bh iv s.BBV _
  exact (recompute_loop s f p g hg scale bh iv s.T _ _
    ⟨seed_ro s p (bh,iv) _ _ hm .h0 (by trivial) f.initial bh iv s.BBV,hm⟩).2

#axiomsClean reverse_loop
#axiomsClean recompute_loop
#axiomsClean program_agree
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.BackwardRelation
