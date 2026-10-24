import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Forward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Geometry
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.GridPrinciple
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.ForwardGrid
open GridPrinciple
noncomputable section
set_option maxHeartbeats 10000000
def Protected (s : Shape) (job : Nat × Nat) (a : Addr) :=
  match a.region with
  | .q | .k | .beta | .h0 => True
  | .v | .o | .ht => a.plane=job.1 ∧ job.2*s.FBV≤a.col ∧ a.col<(job.2+1)*s.FBV
  | _ => False
theorem other_program {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (target job : Nat × Nat) (hne : job≠target) (m : Memory α) :
    Agree (Protected s target) (forwardProgram s f p g scale job.1 job.2 m) m := by
  intro a ha
  rcases target with ⟨tb,tv⟩
  rcases job with ⟨bh,iv⟩
  rcases a with ⟨r,plane,t,col⟩
  have outside (hp : plane=tb) (hlo : tv*s.FBV≤col) (hhi : col<(tv+1)*s.FBV) :
      ¬(plane=bh ∧ iv*s.FBV≤col ∧ col<min ((iv+1)*s.FBV) s.V) := by
    rintro ⟨he,lo,hi⟩
    have same:= (Geometry.owner_unique s.FBV iv col lo (lt_of_lt_of_le hi (Nat.min_le_left _ _))).trans
      (Geometry.owner_unique s.FBV tv col hlo hhi).symm
    apply hne
    have hb : bh=tb := by omega
    simp [hb,same]
  cases r <;> simp only [Protected] at ha <;> try contradiction
  all_goals simp only [Forward.program_full]
  all_goals try (simp [stateStore,Forward.memory,write,ite_apply];done)
  all_goals
    obtain ⟨hp,hlo,hhi⟩:=ha
    have hout:=outside hp hlo hhi
    have hout' : ¬(plane=bh ∧ iv*s.FBV≤col ∧ col<(iv+1)*s.FBV ∧ col<s.V) := by simpa using hout
    cases hf : f.final <;> simp [stateStore,Forward.memory,write,hf]
      <;> aesop

theorem key_agree {α : Type} (s : Shape) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (r : Region) (hr : r=.q ∨ r=.k) (t i : Nat) :
    keyLoad s p m r bh t i=keyLoad s p n r bh t i := by
  unfold keyLoad load
  split
  · apply hc
    rcases hr with rfl|rfl <;> trivial
  · rfl
theorem v_agree {α : Type} (s : Shape) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (t j : Nat) (hj : j<s.FBV) :
    valLoad s p m .v bh t iv s.FBV j=valLoad s p n .v bh t iv s.FBV j := by
  unfold valLoad load
  split
  · apply hc
    dsimp only [Protected]
    exact ⟨rfl,by omega,by rw [Nat.add_mul];omega⟩
  · rfl
theorem beta_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (t j : Nat) :
    betaLoad s f p m bh t iv s.FBV j=betaLoad s f p n bh t iv s.FBV j := by
  have hb (plane row col : Nat) : m ⟨.beta,plane,row,col⟩=n ⟨.beta,plane,row,col⟩ := hc _ (by trivial)
  simp only [betaLoad,valLoad,load,hb]
theorem seed_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (i j : Nat) :
    seed s p m .h0 f.initial bh iv s.FBV i j=seed s p n .h0 f.initial bh iv s.FBV i j := by
  have hs : m ⟨.h0,bh,i,iv*s.FBV+j⟩=n ⟨.h0,bh,i,iv*s.FBV+j⟩ := hc _ (by trivial)
  simp only [seed,load,hs]
theorem h_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (j : Nat) (hj : j<s.FBV) (t i : Nat) :
    Forward.h s f p g m bh iv t i j=Forward.h s f p g n bh iv t i j := by
  induction t generalizing i with
  | zero => exact seed_agree s f p m n bh iv hc i j
  | succ t ih =>
    simp only [Forward.h,ih,key_agree s p m n bh iv hc .k (Or.inr rfl),
      v_agree s p m n bh iv hc t j hj,beta_agree s f p m n bh iv hc]
theorem residual_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (j : Nat) (hj : j<s.FBV) (t : Nat) :
    Forward.residual s f p g m bh iv t j=Forward.residual s f p g n bh iv t j := by
  simp only [Forward.residual,h_agree s f p g m n bh iv hc j hj,
    key_agree s p m n bh iv hc .k (Or.inr rfl),v_agree s p m n bh iv hc t j hj]
theorem output_agree {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α) (m n : Memory α) (bh iv : Nat)
    (hc : Agree (Protected s (bh,iv)) m n) (j : Nat) (hj : j<s.FBV) (t : Nat) :
    Forward.output s f p g scale m bh iv t j=Forward.output s f p g scale n bh iv t j := by
  simp only [Forward.output,h_agree s f p g m n bh iv hc j hj,
    key_agree s p m n bh iv hc .q (Or.inl rfl)]
theorem program_congr {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (job : Nat × Nat) (m n : Memory α) (hc : Agree (Protected s job) m n)
    (a : Addr) (ha : Protected s job a) :
    forwardProgram s f p g scale job.1 job.2 m a=forwardProgram s f p g scale job.1 job.2 n a := by
  have hmem:=hc a ha
  rcases job with ⟨bh,iv⟩
  rcases a with ⟨r,plane,t,col⟩
  cases r <;> simp only [Protected] at ha <;> try contradiction
  all_goals simp only [Forward.program_full]
  all_goals try (simpa [stateStore,Forward.memory,write,ite_apply] using hmem)
  all_goals
    obtain ⟨hp,hlo,hhi⟩:=ha
    have hj : col-iv*s.FBV<s.FBV := by rw [Nat.add_mul] at hhi;omega
    have hh:=h_agree s f p g m n bh iv hc (col-iv*s.FBV) hj
    have hr:=residual_agree s f p g m n bh iv hc (col-iv*s.FBV) hj
    have ho:=output_agree s f p g scale m n bh iv hc (col-iv*s.FBV) hj
    cases hf : f.final <;> simp [stateStore,Forward.memory,write,hf,hh,hr,ho,hmem]

theorem grid_readback {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) (bh iv : Nat) (hb : bh<s.BH) (hv : iv<s.FNV) (a : Addr)
    (ha : Protected s (bh,iv) a) :
    forwardGrid s f p g scale m a=forwardProgram s f p g scale bh iv m a := by
  exact unique_writer (fun job mem => forwardProgram s f p g scale job.1 job.2 mem)
    (Protected s) (bh,iv) a ha (fun job hj mem => other_program s f p g scale (bh,iv) job hj mem)
    (fun x y hc => program_congr s f p g scale (bh,iv) x y hc a ha)
    (jobs s.BH s.FNV) (jobs_nodup _ _) ((MemoryProof.job_member _ _ _ _).mpr ⟨hb,hv⟩) m

#axiomsClean other_program
#axiomsClean program_congr
#axiomsClean grid_readback
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.ForwardGrid
