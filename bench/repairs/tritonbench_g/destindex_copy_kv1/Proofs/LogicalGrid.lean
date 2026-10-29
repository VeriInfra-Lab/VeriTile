import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.LogicalGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- Finite active source reads, with signed metadata interpreted as the
nonnegative logical destination. No global typed-memory assumption is needed. -/
structure Bound (c : Config) (s : State α) (source : Nat → Nat → Nat → α)
    (dest : Nat → Nat) : Prop where
  input : ∀ row,row<c.count → ∀ head dim,head<c.heads → dim<c.dims →
    (s.mem inputRegion (flat c row head dim)).read .real=source row head dim
  metadata : ∀ row,row<c.count → (s.mem metadataRegion.cast row).read .int=(dest row : Int)

def launch (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0}

def step (rest : Algebra α) (c : Config) (s : State α) (pid : Nat) : State α :=
  (Tile.execution rest (layout c) (launch pid s)).val

def run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) : State α :=
  ps.foldl (step rest c) s

theorem step_bound (rest : Algebra α) (c : Config) (s : State α) (pid : Nat)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest) :
    Bound c (step rest c s pid) source dest := by
  constructor
  · intro row hr head dim hh hd
    rw [step,Tile.frame _ _ _ _ _ (Or.inl (by change inputRegion≠outputRegion;decide))]
    exact h.input row hr head dim hh hd
  · intro row hr
    rw [step,Tile.frame _ _ _ _ _ (Or.inl (by change metadataRegion.cast≠outputRegion;decide))]
    exact h.metadata row hr

theorem offset (c : Config) (s : State α) (pid : Nat) (hp : pid<c.count)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (i : TileIndex [block c.heads,block c.dims]) :
    Tile.outputOffset (layout c) (launch pid s) i=flat c (dest pid) i.1.val i.2.1.val := by
  have metadata : Tile.destination (layout c) (launch pid s)=(dest pid : Int) := by
    simpa [Tile.destination,layout,launch] using h.metadata pid hp
  unfold Tile.outputOffset
  rw [metadata]
  simp only [layout,←Nat.cast_mul,←Nat.cast_add,Int.toNat_natCast]
  simp [flat,Nat.mul_comm]

theorem step_read (rest : Algebra α) (c : Config) (s : State α) (pid head dim : Nat)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (hp : pid<c.count) (hh : head<c.heads) (hd : dim<c.dims) :
    (step rest c s pid).mem outputRegion (flat c (dest pid) head dim)=
      .mk .real (source pid head dim) := by
  have hb := Nat.le_pow_clog (by decide : 1<2) c.heads
  have db := Nat.le_pow_clog (by decide : 1<2) c.dims
  let i : TileIndex [block c.heads,block c.dims] :=
    (⟨head,by unfold block;omega⟩,⟨dim,by unfold block;omega⟩,())
  have hi : Tile.active (layout c) i := ⟨hh,hd⟩
  have injective : ∀ a b,Tile.active (layout c) a → Tile.active (layout c) b →
      Tile.outputOffset (layout c) (launch pid s) a=Tile.outputOffset (layout c) (launch pid s) b → a=b := by
    intro a b ha hb same
    rw [offset c s pid hp source dest h a,offset c s pid hp source dest h b] at same
    have parts := Grid.flat_injective c _ _ _ _ _ _ ha.1 hb.1 ha.2 hb.2 same
    apply Prod.ext
    · exact Fin.ext parts.2.1
    · apply Prod.ext
      · exact Fin.ext parts.2.2
      · rfl
  have value := Tile.readback rest (layout c) (launch pid s) i hi injective
  rw [offset c s pid hp source dest h i] at value
  have input : (s.mem inputRegion (flat c pid head dim)).read .real=source pid head dim :=
    h.input pid hp head dim hh hd
  have value' : (step rest c s pid).mem outputRegion (flat c (dest pid) head dim)=
      .mk .real ((s.mem inputRegion (flat c pid head dim)).read .real) := by
    simpa [step,Tile.loaded,Tile.inputOffset,layout,launch,i,flat,Nat.mul_comm] using value
  exact value'.trans (congrArg (fun x : α => (Cell.mk .real x : Cell α)) input)

theorem step_frame (rest : Algebra α) (c : Config) (s : State α) (pid x : Nat)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (hp : pid<c.count)
    (miss : ∀ head dim,head<c.heads → dim<c.dims → x≠flat c (dest pid) head dim) :
    (step rest c s pid).mem outputRegion x=s.mem outputRegion x := by
  apply Tile.frame
  apply Or.inr
  intro i hi
  rw [offset c s pid hp source dest h i]
  exact miss _ _ hi.1 hi.2

theorem other_frame (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (r : RegionName) (o : Nat) (outside : r≠outputRegion) :
    (run rest c ps s).mem r o=s.mem r o := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    change (run rest c ps (step rest c s p)).mem r o=s.mem r o
    rw [ih]
    exact Tile.frame rest (layout c) (launch p s) r o (Or.inl outside)

theorem run_frame (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (members : ∀ p∈ps,p<c.count) (x : Nat)
    (miss : ∀ p∈ps,∀ head dim,head<c.heads → dim<c.dims → x≠flat c (dest p) head dim) :
    (run rest c ps s).mem outputRegion x=s.mem outputRegion x := by
  induction ps generalizing s with
  | nil => rfl
  | cons p tail ih =>
    change (run rest c tail (step rest c s p)).mem outputRegion x=s.mem outputRegion x
    rw [ih _ (step_bound rest c s p source dest h) (fun q hq => members q (by simp [hq]))
      (fun q hq => miss q (by simp [hq]))]
    exact step_frame rest c s p x source dest h (members p (by simp)) (miss p (by simp))

theorem stable (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (unique : ∀ p q,p<c.count → q<c.count → dest p=dest q → p=q)
    (members : ∀ p∈ps,p<c.count) (row head dim : Nat)
    (hr : row<c.count) (hh : head<c.heads) (hd : dim<c.dims)
    (initial : s.mem outputRegion (flat c (dest row) head dim)=.mk .real (source row head dim)) :
    (run rest c ps s).mem outputRegion (flat c (dest row) head dim)=.mk .real (source row head dim) := by
  induction ps generalizing s with
  | nil => exact initial
  | cons p tail ih =>
    apply ih _ (step_bound rest c s p source dest h) (fun q hq => members q (by simp [hq]))
    by_cases eq : p=row
    · subst p;exact step_read rest c s row head dim source dest h hr hh hd
    · rw [step_frame rest c s p _ source dest h (members p (by simp))]
      · exact initial
      · intro a b ha hb same
        have parts := Grid.flat_injective c _ _ _ _ _ _ hh ha hd hb same
        exact eq (unique p row (members p (by simp)) hr parts.1.symm)

theorem run_read (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (unique : ∀ p q,p<c.count → q<c.count → dest p=dest q → p=q)
    (members : ∀ p∈ps,p<c.count) (row head dim : Nat)
    (member : row∈ps) (hh : head<c.heads) (hd : dim<c.dims) :
    (run rest c ps s).mem outputRegion (flat c (dest row) head dim)=.mk .real (source row head dim) := by
  induction ps generalizing s with
  | nil => simp at member
  | cons p tail ih =>
    change (run rest c tail (step rest c s p)).mem outputRegion _=_
    rcases List.mem_cons.mp member with eq|hm
    · subst p
      exact stable rest c tail _ source dest (step_bound rest c s row source dest h)
        unique (fun q hq => members q (by simp [hq])) row head dim (members row (by simp)) hh hd
        (step_read rest c s row head dim source dest h (members row (by simp)) hh hd)
    · exact ih _ (step_bound rest c s p source dest h) (fun q hq => members q (by simp [hq])) hm

#axiomsClean step_read
#axiomsClean run_read
#axiomsClean run_frame
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.LogicalGrid
