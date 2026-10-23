import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.GraphTransport

/-! Compatibility statements retain their original raw-state domains. Their
outputs/frame are obtained from the shared finite IO theorem, then transported
to the original source grid. Tensor consumers use this same connection. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.LegacyIO
open VeriTile.Triton FP.Structural BmmOptimized SourceGraph Interface
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

def values (s : Shape) (m : Memory α) : ChannelValues α (Interface.inputs s) := by
  intro i
  refine Fin.cases (fun j=>m .a j.1.val) ?_ i
  intro k
  have hk : k=0 := Fin.eq_zero k
  subst k
  exact fun j=>m .b j.1.val

def capacity (s : Shape) (r : RegionName) : Nat :=
  if r="A" then s.batch*s.m*s.k else if r="B" then s.batch*s.k*s.n
  else if r="O" then s.size else 0

theorem bounds (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (m : Memory α) :
    (io s c p rest ps).InBounds (capacity s) (values s m) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    simp only [io,List.mem_cons,List.mem_singleton,not_or] at outside
    simp [capacity,outside.1,outside.2.1,outside.2.2]
  · intro i j _
    fin_cases i
    · change j.1.val<s.batch*s.m*s.k
      exact j.1.isLt
    · change j.1.val<s.batch*s.k*s.n
      exact j.1.isLt
  · intro o j _
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    change j.1.val<s.size
    exact j.1.isLt
  · intro k
    exact k.elim0

theorem binding (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (m : Memory α) (initial : State α)
    (h : Bridge.Related m initial) :
    (io s c p rest ps).InputBinding (values s m) (liftState initial) := by
  intro i j _
  fin_cases i
  · change (mapCell Node.value (initial.mem "A" j.1.val)).read .real=Node.value (m .a j.1.val)
    rw [h.a]
    rfl
  · change (mapCell Node.value (initial.mem "B" j.1.val)).read .real=Node.value (m .b j.1.val)
    rw [h.b]
    rfl

theorem array_reference (s : Shape) (p : Scalar α) (m : Memory α)
    (x : BmmOptimized.Cell) (hx : s.Active x) (count : Nat) :
    reference s p (arrays p.zero s (values s m)) x count=reference s p m x count := by
  induction count with
  | zero=>rfl
  | succ count ih=>
    simp only [reference,ih]
    by_cases inside : count<s.k
    · have ha:=active_a_bound s x.batch x.row count hx.1 hx.2.1 inside
      have hb:=active_b_bound s x.batch count x.column hx.1 inside hx.2.2
      simp only [inside,ite_true,arrays,dif_pos ha,dif_pos hb]
      rfl
    · simp only [inside,ite_false]

theorem output_inverse (s : Shape) (x : BmmOptimized.Cell) (hx : s.Active x) :
    outputCell s ⟨s.o x.batch x.row x.column,address_bound s x hx⟩=x := by
  have h:=output_valid s ⟨s.o x.batch x.row x.column,address_bound s x hx⟩
  exact address_injective s _ x h.1 hx h.2

/-- Explicit use of a shared IO completeness proof; the original grid result
is connected by the fixed graph observer, rather than an assumed FP fusion law. -/
theorem from_io (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c))
    (proved : (io s c p rest ps).Implements (scalarFormula s p))
    (m : Memory α) (initial : State α) (related : Bridge.Related m initial) :
    ∃ final,Bridge.grid s c p rest ps initial=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (ordered s p m x)) ∧ Bridge.Frame s initial final := by
  obtain ⟨_,graph,ran,result,frame⟩:=proved (values s m) trivial (capacity s)
    (bounds s c p rest ps m) (liftState initial) (binding s c p rest ps m initial related)
  obtain ⟨final,oldRun,memory⟩:=transport s c p rest ps order (liftState initial) initial
    (lift_observe p initial) graph ran
  refine ⟨final,oldRun,?_,?_⟩
  · intro x hx
    have output:=result (0 : Fin 1) (⟨s.o x.batch x.row x.column,address_bound s x hx⟩,PUnit.unit) trivial
    change mapCell (observe p) (graph.mem "O" (s.o x.batch x.row x.column))=
      .mk .real (ordered s p (arrays p.zero s (values s m))
        (outputCell s ⟨s.o x.batch x.row x.column,address_bound s x hx⟩)) at output
    rw [memory]
    dsimp only
    rw [output,output_inverse s x hx]
    exact congrArg (FP.Structural.Cell.mk .real) (array_reference s p m x hx _)
  · intro r off outside
    have out : (io s c p rest ps).Outside (values s m) r off := by
      constructor
      · intro o j _
        have ho : o=(0 : Fin 1) := Fin.eq_zero o
        subst o
        change r≠"O" ∨ off≠j.1.val
        rcases outside with region|bound
        · exact Or.inl region
        · exact Or.inr (by have hi:=j.1.isLt;omega)
      · intro k
        exact k.elim0
    rw [memory]
    dsimp only
    rw [frame r off out]
    exact (congrFun (congrFun (lift_observe p initial) r) off).symm

theorem complete (s : Shape) (c : Config) (p : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c))
    (m : Memory α) (initial : State α) (related : Bridge.Related m initial) :
    ∃ final,Bridge.grid s c p rest ps initial=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (ordered s p m x)) ∧ Bridge.Frame s initial final :=
  from_io s c p rest ps order (Interface.complete s c p rest ps order) m initial related

#axiomsClean from_io
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.LegacyIO
