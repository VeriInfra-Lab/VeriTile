import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.Grid

/-! Reuse the existing source execution proof on arbitrary raw cells. Only
typed input reads are related to arrays; exterior tags and initial output cells
remain unrestricted. This closes the finite-channel IO binding boundary. -/
namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.ReadGrid
open VeriTile.Triton FP.Structural BmmOptimized Source Execution
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

structure Inputs (m : Memory α) (s : State α) : Prop where
  a : ∀ i,(s.mem "A" i).read .real=m .a i
  b : ∀ i,(s.mem "B" i).read .real=m .b i

def canonical (m : Memory α) (s : State α) : State α :=
  {s with mem:=fun r i=>if r="A" then .mk .real (m .a i)
    else if r="B" then .mk .real (m .b i)
    else if r="O" then .mk .real (m .output i) else s.mem r i}

theorem canonical_related (m : Memory α) (s : State α) :
    Bridge.Related m (canonical m s) := by
  constructor <;> intro i <;> simp [canonical]

theorem input_values (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (m : Memory α) (initial : State α) (h : Inputs m initial) (k : Nat) :
    aValue o (parameters s c) (Bridge.start s c pid initial) k=
      aValue o (parameters s c) (Bridge.start s c pid (canonical m initial)) k ∧
    bValue o (parameters s c) (Bridge.start s c pid initial) k=
      bValue o (parameters s c) (Bridge.start s c pid (canonical m initial)) k := by
  constructor <;> funext i
  · unfold aValue
    change (if _ then (initial.mem "A" _).read .real else o.zero)=
      (if _ then (FP.Structural.Cell.mk .real (m .a _)).read .real else o.zero)
    simp only [Cell.read_same,h.a]
    rfl
  · unfold bValue
    change (if _ then (initial.mem "B" _).read .real else o.zero)=
      (if _ then (FP.Structural.Cell.mk .real (m .b _)).read .real else o.zero)
    simp only [Cell.read_same,h.b]
    rfl

theorem accumulation_equal (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (m : Memory α) (initial : State α) (h : Inputs m initial) (k : Nat) :
    accumulation o (parameters s c) (Bridge.start s c pid initial) k=
      accumulation o (parameters s c) (Bridge.start s c pid (canonical m initial)) k := by
  induction k with
  | zero=>rfl
  | succ k ih=>
    funext i
    simp only [accumulation,(input_values s c o pid m initial h k).1,
      (input_values s c o pid m initial h k).2,ih]

theorem writes_equal (s : Shape) (c : Config) (o : Scalar α) (pid : Program)
    (hp : pid∈programs s c) (m : Memory α) (initial : State α) (h : Inputs m initial) :
    Bridge.sourceWrites o (parameters s c) (Bridge.start s c pid initial)=
      programWrites s c o pid m := by
  rw [←Bridge.writes_equal s c o pid hp m (canonical m initial) (canonical_related m initial)]
  unfold Bridge.sourceWrites
  simp only [accumulation_equal s c o pid m initial h]
  rfl

theorem commit_mem_congr (writes : List (Nat×α)) (left right : State α)
    (h : left.mem=right.mem) :
    (Bridge.commitState writes left).mem=(Bridge.commitState writes right).mem := by
  induction writes generalizing left right with
  | nil=>exact h
  | cons w ws ih=>
    apply ih
    simp [State.write,h]

theorem commit_inputs (writes : List (Nat×α)) (m : Memory α) (initial final : State α)
    (h : Inputs m initial) (eq : final.mem=(Bridge.commitState writes initial).mem) :
    Inputs m final := by
  constructor
  · intro i
    rw [eq,Bridge.commit_frame writes initial "A" i (by intros;exact Or.inl (by decide))]
    exact h.a i
  · intro i
    rw [eq,Bridge.commit_frame writes initial "B" i (by intros;exact Or.inl (by decide))]
    exact h.b i

theorem program_execution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (pid : Program) (hp : pid∈programs s c) (m : Memory α) (initial : State α)
    (h : Inputs m initial) :
    ∃ final,Execution.exec o rest (parameters s c) (Bridge.start s c pid initial)=some final ∧
      final.mem=(Bridge.commitState (programWrites s c o pid m) initial).mem := by
  obtain ⟨final,run,eq⟩:=Execution.correctness o rest (parameters s c) (Bridge.start s c pid initial)
  rw [Bridge.scatter_commit,writes_equal s c o pid hp m initial h] at eq
  exact ⟨final,run,eq.trans (commit_mem_congr _ _ _ rfl)⟩

theorem grid_execution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (ps : List Program) (legal : ∀ pid∈ps,pid∈programs s c)
    (m : Memory α) (initial : State α) (h : Inputs m initial) :
    ∃ final,Bridge.grid s c o rest ps initial=some final ∧
      final.mem=(Bridge.commitState (ps.flatMap (fun pid=>programWrites s c o pid m)) initial).mem := by
  induction ps generalizing initial with
  | nil=>exact ⟨initial,rfl,rfl⟩
  | cons pid ps ih=>
    obtain ⟨next,ran,eq⟩:=program_execution s c o rest pid (legal pid (by simp)) m initial h
    obtain ⟨final,run,finish⟩:=ih (fun q hq=>legal q (by simp [hq])) next
      (commit_inputs _ m initial next h eq)
    refine ⟨final,by simp [Bridge.grid,ran,bind,Option.bind,run],?_⟩
    rw [finish,commit_mem_congr _ next _ eq]
    simp only [List.flatMap_cons,Bridge.commitState,List.foldl_append]

theorem commit_read (writes : List (Nat×α)) (initial : State α) (address : Nat) (value : α)
    (present : ∃ w∈writes,w.1=address)
    (agree : ∀ w∈writes,w.1=address → w.2=value) :
    (Bridge.commitState writes initial).mem "O" address=.mk .real value := by
  induction writes generalizing initial with
  | nil=>simpa using present
  | cons w ws ih=>
    change (Bridge.commitState ws (initial.write "O" w.1 (.mk .real w.2))).mem "O" address=_
    by_cases tail : ∃ v∈ws,v.1=address
    · exact ih _ tail (fun v hv=>agree v (by simp [hv]))
    · have here : w.1=address := by
        rcases present with ⟨v,hv,eq⟩
        rcases List.mem_cons.mp hv with same|inside
        · simpa [same] using eq
        · exact False.elim (tail ⟨v,inside,eq⟩)
      rw [Bridge.commit_frame _ _ _ _ (by
        intro v hv
        exact Or.inr (fun h=>tail ⟨v,hv,h.symm⟩))]
      simp only [State.write,here,and_self,↓reduceIte]
      exact congrArg (FP.Structural.Cell.mk .real) (agree w (by simp) here)

/-- Complete ordered source result, using decoded input cells without a raw
tag assumption. The original complete execution and ownership proofs are reused. -/
theorem ordered_execution (s : Shape) (c : Config) (o : Scalar α) (rest : Algebra α)
    (ps : List Program) (order : ps.Perm (programs s c))
    (m : Memory α) (initial : State α) (h : Inputs m initial) :
    ∃ final,Bridge.grid s c o rest ps initial=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (ordered s o m x)) ∧ Bridge.Frame s initial final := by
  obtain ⟨final,run,eq⟩:=grid_execution s c o rest ps
    (fun pid hp=>order.mem_iff.mp hp) m initial h
  have members : ∀ w,w∈ps.flatMap (fun pid=>programWrites s c o pid m) ↔ w∈allWrites s c o m := by
    intro w
    simp only [allWrites,List.mem_flatMap,order.mem_iff]
  refine ⟨final,run,?_,?_⟩
  · intro x hx
    rw [eq,←source_ordered s o m x hx]
    apply commit_read
    · exact ⟨_,(members _).mpr (allWrites_covers s c o m x hx),rfl⟩
    · intro w hw address
      obtain ⟨y,hy,wy⟩:=allWrites_member s c o m w ((members _).mp hw)
      subst w
      have same:=address_injective s y x hy hx address
      simp [same]
  · intro r off outside
    rw [eq]
    apply Bridge.commit_frame
    intro w hw
    rcases outside with other|bound
    · exact Or.inl other
    · obtain ⟨x,hx,eq⟩:=allWrites_member s c o m w ((members _).mp hw)
      have upper:=address_bound s x hx
      subst w
      exact Or.inr (by simp only;omega)

#axiomsClean ordered_execution
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.ReadGrid
