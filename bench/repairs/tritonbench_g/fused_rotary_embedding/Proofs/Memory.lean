import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding
variable {α : Type}

/-- Every pair is computed from current memory. V takes the copy branch and
does not pass through any arithmetic instruction. -/
def pairValues (s : Shape) (p : Kind → Ops α) (w : Work) (m : Memory α) : α × α :=
  let x0 := m w.kind.input (s.input w false)
  let x1 := m w.kind.input (s.input w true)
  if w.kind=.v then (x0,x1)
  else sourcePair (p w.kind) x0 x1 (m .cos (s.coeff false w.token w.pair))
    (m .sin (s.coeff true w.token w.pair))

/-- Pair-local source order: all operands loaded before low and high stores. -/
def step (c : Config) (p : Kind → Ops α) (w : Work) (m : Memory α) : Memory α :=
  let values := pairValues c.shape p w m
  Stores.store (Stores.store m ((w.kind.output,c.output w false),values.1))
    ((w.kind.output,c.output w true),values.2)

def Writes (c : Config) (w : Work) (r : Region) (a : Nat) :=
  r=w.kind.output ∧ (a=c.output w false ∨ a=c.output w true)
instance (c : Config) (w : Work) (r : Region) (a : Nat) : Decidable (Writes c w r a) :=
  inferInstanceAs (Decidable (_∧(_∨_)))

theorem cells_distinct (c : Config) (hc : c.Valid) (w : Work) (hw : c.shape.Active w) :
    c.output w false≠c.output w true := by
  intro eq
  have := (output_injective c hc w w false true hw hw rfl eq).2
  contradiction

theorem step_readback (c : Config) (hc : c.Valid) (p : Kind → Ops α) (w : Work)
    (hw : c.shape.Active w) (m : Memory α) (o : Bool) :
    step c p w m w.kind.output (c.output w o)=component (pairValues c.shape p w m) o := by
  have hn := cells_distinct c hc w hw
  cases o <;> simp [step,Stores.store,component,hn]

theorem step_frame (c : Config) (p : Kind → Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ¬Writes c w r a) : step c p w m r a=m r a := by
  simp only [Writes,not_and,not_or] at outside
  unfold step Stores.store
  by_cases hr : r=w.kind.output
  · simp [hr,outside hr]
  · simp [hr]

theorem disjoint_cell (c : Config) (hc : c.Valid) (w v : Work)
    (hw : c.shape.Active w) (hv : c.shape.Active v) (different : w≠v) (o : Bool) :
    ¬Writes c w v.kind.output (c.output v o) := by
  rintro ⟨hr,ha|ha⟩
  · exact different (output_injective c hc w v false o hw hv hr.symm ha.symm).1
  · exact different (output_injective c hc w v true o hw hv hr.symm ha.symm).1

theorem input_stable (c : Config) (hc : c.Valid) (p : Kind → Ops α) (w v : Work)
    (hw : c.shape.Active w) (hv : c.shape.Active v) (different : w≠v) (m : Memory α) (o : Bool) :
    step c p w m v.kind.input (c.shape.input v o)=m v.kind.input (c.shape.input v o) := by
  apply step_frame
  cases kv : v.kind
  · have eq : c.output v o=c.shape.input v o := by simp [Config.output,kv]
    have h := disjoint_cell c hc w v hw hv different o
    simpa [eq,kv,Kind.input,Kind.output] using h
  · cases kw : w.kind <;> simp [Writes,kv,kw,Kind.input,Kind.output]
  · cases kw : w.kind <;> simp [Writes,kv,kw,Kind.input,Kind.output]

theorem step_readonly (c : Config) (p : Kind → Ops α) (w : Work) (m : Memory α)
    (r : Region) (a : Nat) (hr : r≠.q ∧ r≠.kcache ∧ r≠.vcache) : step c p w m r a=m r a := by
  apply step_frame
  cases kw : w.kind <;> simp_all [Writes,Kind.output]

theorem step_preserves_pair (c : Config) (hc : c.Valid) (p : Kind → Ops α) (w v : Work)
    (hw : c.shape.Active w) (hv : c.shape.Active v) (different : w≠v) (m : Memory α) :
    pairValues c.shape p v (step c p w m)=pairValues c.shape p v m := by
  have x0 := input_stable c hc p w v hw hv different m false
  have x1 := input_stable c hc p w v hw hv different m true
  have co := step_readonly c p w m .cos (c.shape.coeff false v.token v.pair) (by decide)
  have si := step_readonly c p w m .sin (c.shape.coeff true v.token v.pair) (by decide)
  simp only [pairValues,x0,x1,co,si]

/-- Current memory is threaded through all stages and programs. -/
def execute (c : Config) (p : Kind → Ops α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => execute c p ws (step c p w m)

theorem execute_frame (c : Config) (p : Kind → Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes c w r a) : execute c p ws m r a=m r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change execute c p ws (step c p w m) r a=m r a
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact step_frame c p w m r a (outside w (by simp))

theorem execute_readback (c : Config) (hc : c.Valid) (p : Kind → Ops α)
    (ws : List Work) (m : Memory α) (active : ∀ w∈ws,c.shape.Active w)
    (unique : ws.Nodup) (v : Work) (member : v∈ws) (o : Bool) :
    execute c p ws m v.kind.output (c.output v o)=component (pairValues c.shape p v m) o := by
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
      exact step_readback c hc p w hw m o
    · have different : w≠v := by intro eq;subst v;exact nd.1 tail
      change execute c p ws (step c p w m) _ _=_
      rw [ih (step c p w m) (fun x hx => active x (by simp [hx])) nd.2 tail]
      rw [step_preserves_pair c hc p w v hw hv different m]

theorem execute_append (c : Config) (p : Kind → Ops α) (a b : List Work) (m : Memory α) :
    execute c p (a++b) m=execute c p b (execute c p a m) := by
  induction a generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih (step c p w m)

#axiomsClean execute_readback
#axiomsClean execute_frame
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
