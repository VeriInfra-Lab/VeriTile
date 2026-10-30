import bench.repairs.tritonbench_g.rotary_emb.Proofs.Geometry

namespace VeriTile.Bench.Repairs.RotaryEmb

variable {α : Type}

def pairValues (s : Shape) (p : Ops α) (w : Work) (m : Memory α) : α × α :=
  sourcePair p (m (dataRegion w.key) (s.cell w false))
    (m (dataRegion w.key) (s.cell w true))
    (m .cos (s.coefficient false w.token (2*w.pair)))
    (m .sin (s.coefficient true w.token (2*w.pair)))
    (m .cos (s.coefficient false w.token (2*w.pair+1)))
    (m .sin (s.coefficient true w.token (2*w.pair+1)))

def component (v : α × α) (odd : Bool) : α := if odd then v.2 else v.1

def step (s : Shape) (p : Ops α) (w : Work) (m : Memory α) : Memory α :=
  sourceStep p (dataRegion w.key) (s.cell w false) (s.cell w true)
    (s.coefficient false w.token (2*w.pair))
    (s.coefficient false w.token (2*w.pair+1))
    (s.coefficient true w.token (2*w.pair))
    (s.coefficient true w.token (2*w.pair+1)) m

def Writes (s : Shape) (w : Work) (r : Region) (a : Nat) : Prop :=
  r=dataRegion w.key ∧ (a=s.cell w false ∨ a=s.cell w true)

instance (s : Shape) (w : Work) (r : Region) (a : Nat) : Decidable (Writes s w r a) :=
  inferInstanceAs (Decidable (_ ∧ (_ ∨ _)))

theorem cells_distinct (s : Shape) (w : Work) : s.cell w false ≠ s.cell w true := by
  simp [Shape.cell,Shape.address]

theorem step_readback (s : Shape) (p : Ops α) (w : Work) (m : Memory α) (odd : Bool) :
    step s p w m (dataRegion w.key) (s.cell w odd)=component (pairValues s p w m) odd := by
  have hn := cells_distinct s w
  cases odd <;> simp [step,sourceStep,component,pairValues,hn]

theorem step_frame (s : Shape) (p : Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ¬Writes s w r a) : step s p w m r a=m r a := by
  simp only [Writes,not_and,not_or] at outside
  unfold step sourceStep
  by_cases hr : r=dataRegion w.key
  · simp [hr,outside hr]
  · simp [hr]

theorem disjoint_cell (s : Shape) (valid : s.Valid) (w v : Work)
    (hw : s.Active w) (hv : s.Active v) (different : w≠v) (odd : Bool) :
    ¬Writes s w (dataRegion v.key) (s.cell v odd) := by
  rintro ⟨hr,ha|ha⟩
  · exact different (cell_injective s valid w v false odd hw hv hr.symm ha.symm).1
  · exact different (cell_injective s valid w v true odd hw hv hr.symm ha.symm).1

theorem step_coefficients (s : Shape) (p : Ops α) (w : Work) (m : Memory α)
    (sine : Bool) (a : Nat) :
    step s p w m (if sine then .sin else .cos) a=m (if sine then .sin else .cos) a := by
  apply step_frame
  cases sine <;> cases hk : w.key <;> simp [Writes,dataRegion,hk]

theorem step_preserves_pair (s : Shape) (valid : s.Valid) (p : Ops α) (w v : Work)
    (hw : s.Active w) (hv : s.Active v) (different : w≠v) (m : Memory α) :
    pairValues s p v (step s p w m)=pairValues s p v m := by
  have x0 := step_frame s p w m _ _ (disjoint_cell s valid w v hw hv different false)
  have x1 := step_frame s p w m _ _ (disjoint_cell s valid w v hw hv different true)
  have c0 := step_coefficients s p w m false (s.coefficient false v.token (2*v.pair))
  have c1 := step_coefficients s p w m false (s.coefficient false v.token (2*v.pair+1))
  have s0 := step_coefficients s p w m true (s.coefficient true v.token (2*v.pair))
  have s1 := step_coefficients s p w m true (s.coefficient true v.token (2*v.pair+1))
  simp only [Bool.false_eq_true,↓reduceIte] at c0 c1 s0 s1
  simp only [pairValues,x0,x1,c0,c1,s0,s1]

/-- Each work item reads the memory returned by the preceding item. This is
in-place execution, not evaluation from an assumed immutable input snapshot. -/
def execute (s : Shape) (p : Ops α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => execute s p ws (step s p w m)

theorem execute_frame (s : Shape) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws, ¬Writes s w r a) :
    execute s p ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change execute s p ws (step s p w m) r a=m r a
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact step_frame s p w m r a (outside w (by simp))

theorem execute_readback (s : Shape) (valid : s.Valid) (p : Ops α)
    (ws : List Work) (m : Memory α) (active : ∀ w∈ws,s.Active w)
    (unique : ws.Nodup) (v : Work) (member : v∈ws) (odd : Bool) :
    execute s p ws m (dataRegion v.key) (s.cell v odd)=
      component (pairValues s p v m) odd := by
  induction ws generalizing m with
  | nil => simp at member
  | cons w ws ih =>
    have nd := List.nodup_cons.mp unique
    have hw := active w (by simp)
    have hv := active v member
    rcases List.mem_cons.mp member with same|tail
    · subst v
      change execute s p ws (step s p w m) _ _=_
      rw [execute_frame s p ws (step s p w m) _ _ (by
        intro v hv'
        exact disjoint_cell s valid v w (active v (by simp [hv'])) hw
          (by intro eq; subst v; exact nd.1 hv') odd)]
      exact step_readback s p w m odd
    · have different : w≠v := by intro eq; subst v; exact nd.1 tail
      change execute s p ws (step s p w m) _ _=_
      rw [ih (step s p w m) (fun x hx => active x (by simp [hx])) nd.2 tail]
      rw [step_preserves_pair s valid p w v hw hv different m]

#axiomsClean execute_frame
#axiomsClean execute_readback
end VeriTile.Bench.Repairs.RotaryEmb
