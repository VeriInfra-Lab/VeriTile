import bench.tritonbench_g.rotary_transform.Proofs.Geometry
namespace VeriTile.Bench.Repairs.RotaryTransform
variable {α : Type}
theorem step_readback (c : Config) (p : Ops α) (w : Work) (hw : c.Active w)
    (m : Memory α) (o : Bool) :
    step c p w m c.shape.output (c.cell w o)=component (pairValues c p w m) o := by
  have hn := cells_distinct c w hw
  cases o <;> simp [step,component,hn]
theorem step_frame (c : Config) (p : Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ¬Writes c w r a) : step c p w m r a=m r a := by
  simp only [Writes,not_and,not_or] at outside
  unfold step
  by_cases hr : r=c.shape.output
  · simp [hr,outside hr]
  · simp [hr]
theorem disjoint_cell (c : Config) (hc : c.Valid) (w v : Work)
    (hw : c.Active w) (hv : c.Active v) (different : w≠v) (o : Bool) :
    ¬Writes c w c.shape.output (c.cell v o) := by
  rintro ⟨_,ha|ha⟩
  · exact different (cell_injective c hc w v false o hw hv ha.symm).1
  · exact different (cell_injective c hc w v true o hw hv ha.symm).1
theorem disjoint_input (c : Config) (hc : c.Valid) (w v : Work)
    (hw : c.Active w) (hv : c.Active v) (different : w≠v) (o : Bool) :
    ¬Writes c w .input (c.cell v o) := by
  intro h
  exact disjoint_cell c hc w v hw hv different o ⟨rfl,h.2⟩
theorem step_readonly (c : Config) (p : Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (h : r≠.input ∧ r≠.output) : step c p w m r a=m r a := by
  apply step_frame
  cases hi : c.shape.inplace <;> simp_all [Writes,Shape.output]
theorem step_preserves_pair (c : Config) (hc : c.Valid) (p : Ops α) (w v : Work)
    (hw : c.Active w) (hv : c.Active v) (different : w≠v) (m : Memory α) :
    pairValues c p v (step c p w m)=pairValues c p v m := by
  have x0 := step_frame c p w m _ _ (disjoint_input c hc w v hw hv different false)
  have x1 := step_frame c p w m _ _ (disjoint_input c hc w v hw hv different true)
  have co := step_readonly c p w m .cosine (c.coeff v) (by decide)
  have si := step_readonly c p w m .sine (c.coeff v) (by decide)
  simp only [pairValues,x0,x1,co,si]
def execute (c : Config) (p : Ops α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => execute c p ws (step c p w m)
theorem execute_frame (c : Config) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes c w r a) : execute c p ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change execute c p ws (step c p w m) r a=_
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact step_frame c p w m r a (outside w (by simp))
theorem execute_readback (c : Config) (hc : c.Valid) (p : Ops α)
    (ws : List Work) (m : Memory α) (active : ∀ w∈ws,c.Active w) (unique : ws.Nodup)
    (v : Work) (member : v∈ws) (o : Bool) :
    execute c p ws m c.shape.output (c.cell v o)=component (pairValues c p v m) o := by
  induction ws generalizing m with
  | nil => simp at member
  | cons w ws ih =>
    have nd := List.nodup_cons.mp unique
    have hw := active w (by simp)
    have hv := active v member
    rcases List.mem_cons.mp member with same|tail
    · subst v
      change execute c p ws (step c p w m) _ _=_
      rw [execute_frame c p ws (step c p w m) _ _ (by
        intro v hv'
        exact disjoint_cell c hc v w (active v (by simp [hv'])) hw
          (by intro eq;subst v;exact nd.1 hv') o)]
      exact step_readback c p w hw m o
    · have different : w≠v := by intro eq;subst v;exact nd.1 tail
      change execute c p ws (step c p w m) _ _=_
      rw [ih (step c p w m) (fun x hx => active x (by simp [hx])) nd.2 tail]
      rw [step_preserves_pair c hc p w v hw hv different m]
theorem execute_append (c : Config) (p : Ops α) (a b : List Work) (m : Memory α) :
    execute c p (a++b) m=execute c p b (execute c p a m) := by
  induction a generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih (step c p w m)
#axiomsClean execute_readback
#axiomsClean execute_frame
end VeriTile.Bench.Repairs.RotaryTransform

