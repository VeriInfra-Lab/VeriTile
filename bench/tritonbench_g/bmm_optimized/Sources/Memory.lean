import bench.tritonbench_g.bmm_optimized.Sources.Arithmetic

namespace VeriTile.Bench.Repairs.BmmOptimized
variable {α : Type}

theorem store_frame (m : Memory α) (address : Nat) (value : α) (r : Region) (a : Nat)
    (outside : r≠.output ∨ a≠address) : store m address value r a=m r a := by
  rcases outside with h|h <;> simp [store,h]
theorem store_inputs (m : Memory α) (address : Nat) (value : α) :
    (store m address value .a=m .a) ∧ (store m address value .b=m .b) := by
  constructor <;> funext n <;> simp [store]

theorem laneLoop_inputs (s : Shape) (p : Scalar α) (a b : Memory α)
    (ha : a .a=b .a) (hb : a .b=b .b) (x : Cell) (start n : Nat) (carry : α) :
    laneLoop s p a x start n carry=laneLoop s p b x start n carry := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [laneLoop,aLoad,bLoad,ha,hb,ih]
theorem sourceValue_inputs (s : Shape) (p : Scalar α) (a b : Memory α)
    (ha : a .a=b .a) (hb : a .b=b .b) (x : Cell) :
    sourceValue s p a x=sourceValue s p b x := by
  simp only [sourceValue,blockLoop_flat]
  exact laneLoop_inputs s p a b ha hb x 0 _ _

def commit (writes : List (Nat×α)) (m : Memory α) :=
  writes.foldl (fun state w => store state w.1 w.2) m
theorem commit_frame (writes : List (Nat×α)) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w∈writes,r≠.output ∨ a≠w.1) : commit writes m r a=m r a := by
  induction writes generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change commit ws (store m w.1 w.2) r a=m r a
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    exact store_frame m w.1 w.2 r a (outside w (by simp))
theorem commit_inputs (writes : List (Nat×α)) (m : Memory α) :
    (commit writes m .a=m .a) ∧ (commit writes m .b=m .b) := by
  constructor <;> funext a <;> apply commit_frame <;> intros <;> exact Or.inl (by decide)
theorem commit_append (a b : List (Nat×α)) (m : Memory α) :
    commit (a++b) m=commit b (commit a m) := by
  exact List.foldl_append

theorem programWrites_inputs (s : Shape) (c : Config) (p : Scalar α) (pid : Program)
    (a b : Memory α) (ha : a .a=b .a) (hb : a .b=b .b) :
    programWrites s c p pid a=programWrites s c p pid b := by
  unfold programWrites
  congr 1
  funext x
  split_ifs
  · rw [sourceValue_inputs s p a b ha hb]
  · rfl
theorem sourceProgram_inputs (s : Shape) (c : Config) (p : Scalar α) (pid : Program)
    (m : Memory α) : (sourceProgram s c p pid m .a=m .a) ∧
      (sourceProgram s c p pid m .b=m .b) :=
  commit_inputs _ m

theorem sourcePrograms_inputs (s : Shape) (c : Config) (p : Scalar α)
    (ps : List Program) (m : Memory α) :
    (sourcePrograms s c p ps m .a=m .a) ∧ (sourcePrograms s c p ps m .b=m .b) := by
  induction ps generalizing m with
  | nil => exact ⟨rfl,rfl⟩
  | cons pid ps ih =>
    have step := sourceProgram_inputs s c p pid m
    have tail := ih (sourceProgram s c p pid m)
    exact ⟨tail.1.trans step.1,tail.2.trans step.2⟩

/-- Flattening snapshots is a theorem of current-memory input preservation;
the executor itself computes each CTA from its current state. -/
theorem sourcePrograms_commit (s : Shape) (c : Config) (p : Scalar α)
    (ps : List Program) (m : Memory α) :
    sourcePrograms s c p ps m =
      commit (ps.flatMap (fun pid => programWrites s c p pid m)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid ps ih =>
    rw [sourcePrograms,ih,List.flatMap_cons,commit_append]
    have inputs := sourceProgram_inputs s c p pid m
    have same : ps.flatMap (fun q => programWrites s c p q (sourceProgram s c p pid m))=
        ps.flatMap (fun q => programWrites s c p q m) := by
      congr 1
      funext q
      exact programWrites_inputs s c p q _ m inputs.1 inputs.2
    rw [same]
    rfl

theorem address_injective (s : Shape) (x y : Cell) (hx : s.Active x) (hy : s.Active y)
    (eq : s.o x.batch x.row x.column=s.o y.batch y.row y.column) : x=y := by
  unfold Shape.Active at hx hy
  have hn : 0<s.n := by omega
  have hm : 0<s.m := by omega
  have divN (b row col : Nat) (hc : col<s.n) :
      ((b*s.m+row)*s.n+col)/s.n=b*s.m+row := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hn,Nat.div_eq_of_lt hc];simp
  have divM (b row : Nat) (hr : row<s.m) : (b*s.m+row)/s.m=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hm,Nat.div_eq_of_lt hr];simp
  have rows := congrArg (fun n => n/s.n) eq
  simp only [Shape.o,divN _ _ _ hx.2.2,divN _ _ _ hy.2.2] at rows
  have batches := congrArg (fun n => n/s.m) rows
  simp only [divM _ _ hx.2.1,divM _ _ hy.2.1] at batches
  rw [batches] at rows
  have rr : x.row=y.row := by omega
  have cc : x.column=y.column := by
    unfold Shape.o at eq
    rw [batches,rr] at eq
    omega
  cases x;cases y;simp_all

theorem address_bound (s : Shape) (x : Cell) (hx : s.Active x) :
    s.o x.batch x.row x.column<s.size := by
  unfold Shape.Active at hx
  have row : x.batch*s.m+x.row<s.batch*s.m := by
    calc
      _ < x.batch*s.m+s.m := Nat.add_lt_add_left hx.2.1 _
      _ = (x.batch+1)*s.m := by ring
      _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
  unfold Shape.o Shape.size
  calc
    _ < (x.batch*s.m+x.row)*s.n+s.n := Nat.add_lt_add_left hx.2.2 _
    _ = (x.batch*s.m+x.row+1)*s.n := by ring
    _ ≤ _ := Nat.mul_le_mul_right _ (by omega)

#axiomsClean sourceValue_inputs
#axiomsClean sourcePrograms_inputs
#axiomsClean sourcePrograms_commit
#axiomsClean address_injective
#axiomsClean address_bound
end VeriTile.Bench.Repairs.BmmOptimized
