import bench.tritonbench_g.destindex_copy_kv1.Proofs.Tile
import bench.tritonbench_g.destindex_copy_kv1.Contract
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

structure Bound (s : State α) (source : Nat → α) (dest : Nat → Nat) : Prop where
  input : ∀ x, s.mem inputRegion x = .mk .real (source x)
  metadata : ∀ x, s.mem metadataRegion.cast x = .mk .int (dest x)

def step (rest : Algebra α) (c : Config) (s : State α) (pid : Nat) : State α :=
  (Tile.execution rest (layout c) {s with pids := fun d => if d=0 then pid else s.pids d}).val

def run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) : State α :=
  ps.foldl (step rest c) s

theorem step_bound (rest : Algebra α) (c : Config) (s : State α) (pid : Nat)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest) :
    Bound (step rest c s pid) source dest := by
  constructor
  · intro x
    rw [step,Tile.frame _ _ _ _ _ (Or.inl (by change inputRegion ≠ outputRegion; decide))]
    exact h.input x
  · intro x
    rw [step,Tile.frame _ _ _ _ _ (Or.inl (by change metadataRegion.cast ≠ outputRegion; decide))]
    exact h.metadata x

theorem destination (c : Config) (s : State α) (pid : Nat)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest) :
    Tile.destination (layout c) {s with pids := fun d => if d=0 then pid else s.pids d}=dest pid := by
  simp [Tile.destination,layout,h.metadata]

theorem offset (c : Config) (s : State α) (pid : Nat)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (i : TileIndex [block c.heads,block c.dims]) :
    Tile.outputOffset (layout c) {s with pids := fun d => if d=0 then pid else s.pids d} i=
      flat c (dest pid) i.1.val i.2.1.val := by
  unfold Tile.outputOffset
  rw [destination c s pid source dest h]
  simp only [layout,←Nat.cast_mul,←Nat.cast_add,Int.toNat_natCast]
  simp [flat,Nat.mul_comm]

theorem flat_injective (c : Config) (r h d r' h' d' : Nat)
    (hh : h<c.heads) (hh' : h'<c.heads) (hd : d<c.dims) (hd' : d'<c.dims)
    (same : flat c r h d=flat c r' h' d') : r=r' ∧ h=h' ∧ d=d' := by
  have grouped (r h d : Nat) : flat c r h d=(r*c.heads+h)*c.dims+d := by
    simp only [flat,Nat.add_mul,Nat.mul_assoc]
  have dd := congrArg (fun x => x%c.dims) same
  simp only [grouped,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd'] at dd
  have rh : r*c.heads+h=r'*c.heads+h' := by rw [grouped,grouped] at same; nlinarith
  have hh2 := congrArg (fun x => x%c.heads) rh
  simp only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh'] at hh2
  exact ⟨by nlinarith,hh2,dd⟩

theorem flat_bound (c : Config) (n r h d : Nat)
    (hr : r<n) (hh : h<c.heads) (hd : d<c.dims) : flat c r h d<n*c.heads*c.dims := by
  have first : r*c.heads+h<n*c.heads := by nlinarith
  have second : (r*c.heads+h)*c.dims+d<(n*c.heads)*c.dims := by nlinarith
  simpa only [flat,Nat.add_mul,Nat.mul_assoc] using second

theorem step_read (rest : Algebra α) (c : Config) (s : State α) (pid head dim : Nat)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (hh : head<c.heads) (hd : dim<c.dims) :
    (step rest c s pid).mem outputRegion (flat c (dest pid) head dim)=
      .mk .real (source (flat c pid head dim)) := by
  have hb := Nat.le_pow_clog (by decide : 1<2) c.heads
  have db := Nat.le_pow_clog (by decide : 1<2) c.dims
  let i : TileIndex [block c.heads,block c.dims] :=
    (⟨head,by unfold block; omega⟩,⟨dim,by unfold block; omega⟩,())
  have hi : Tile.active (layout c) i := ⟨hh,hd⟩
  have injective : ∀ a b, Tile.active (layout c) a → Tile.active (layout c) b →
      Tile.outputOffset (layout c) {s with pids := fun d => if d=0 then pid else s.pids d} a=
      Tile.outputOffset (layout c) {s with pids := fun d => if d=0 then pid else s.pids d} b → a=b := by
    intro a b ha hb eq
    rw [offset c s pid source dest h a,offset c s pid source dest h b] at eq
    have parts := flat_injective c _ _ _ _ _ _ ha.1 hb.1 ha.2 hb.2 eq
    apply Prod.ext
    · exact Fin.ext parts.2.1
    · apply Prod.ext
      · exact Fin.ext parts.2.2
      · rfl
  have read := Tile.readback rest (layout c)
    {s with pids := fun d => if d=0 then pid else s.pids d} i hi injective
  rw [offset c s pid source dest h i] at read
  simpa [step,Tile.loaded,Tile.inputOffset,layout,i,h.input,flat,Nat.mul_comm] using read

theorem step_frame (rest : Algebra α) (c : Config) (s : State α) (pid x : Nat)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (miss : ∀ head dim, head<c.heads → dim<c.dims → x≠flat c (dest pid) head dim) :
    (step rest c s pid).mem outputRegion x=s.mem outputRegion x := by
  apply Tile.frame
  apply Or.inr
  intro i hi
  rw [offset c s pid source dest h i]
  exact miss _ _ hi.1 hi.2

theorem run_bound (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest) :
    Bound (run rest c ps s) source dest := by
  induction ps generalizing s with
  | nil => exact h
  | cons p tail ih => exact ih _ (step_bound rest c s p source dest h)

theorem run_frame (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest) (x : Nat)
    (miss : ∀ p∈ps, ∀ head dim, head<c.heads → dim<c.dims → x≠flat c (dest p) head dim) :
    (run rest c ps s).mem outputRegion x=s.mem outputRegion x := by
  induction ps generalizing s with
  | nil => rfl
  | cons p tail ih =>
    change (run rest c tail (step rest c s p)).mem outputRegion x=s.mem outputRegion x
    rw [ih _ (step_bound rest c s p source dest h) (fun q hq => miss q (by simp [hq]))]
    exact step_frame rest c s p x source dest h (miss p (by simp))

theorem stable (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q)
    (members : ∀ p∈ps, p<c.count) (row head dim : Nat)
    (hr : row<c.count) (hh : head<c.heads) (hd : dim<c.dims)
    (initial : s.mem outputRegion (flat c (dest row) head dim)=.mk .real (source (flat c row head dim))) :
    (run rest c ps s).mem outputRegion (flat c (dest row) head dim)=.mk .real (source (flat c row head dim)) := by
  induction ps generalizing s with
  | nil => exact initial
  | cons p tail ih =>
    apply ih _ (step_bound rest c s p source dest h) (fun q hq => members q (by simp [hq]))
    by_cases eq : p=row
    · subst p; exact step_read rest c s row head dim source dest h hh hd
    · rw [step_frame rest c s p _ source dest h]
      · exact initial
      · intro a b ha hb same
        have parts := flat_injective c _ _ _ _ _ _ hh ha hd hb same
        exact eq (unique p row (members p (by simp)) hr parts.1.symm)

theorem run_read (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q)
    (members : ∀ p∈ps, p<c.count) (row head dim : Nat)
    (member : row∈ps) (hh : head<c.heads) (hd : dim<c.dims) :
    (run rest c ps s).mem outputRegion (flat c (dest row) head dim)=.mk .real (source (flat c row head dim)) := by
  induction ps generalizing s with
  | nil => simp at member
  | cons p tail ih =>
    change (run rest c tail (step rest c s p)).mem outputRegion _=_
    rcases List.mem_cons.mp member with eq | hm
    · subst p
      exact stable rest c tail _ source dest (step_bound rest c s row source dest h)
        unique (fun q hq => members q (by simp [hq])) row head dim (members row (by simp)) hh hd
        (step_read rest c s row head dim source dest h hh hd)
    · exact ih _ (step_bound rest c s p source dest h) (fun q hq => members q (by simp [hq])) hm

/-- Every complete program order is covered by the same source theorem. -/
theorem correct (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (h : Bound s source dest)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q)
    (complete : ∀ p, p∈ps ↔ p<c.count) :
    RowCopy c dest source (fun x => (s.mem outputRegion x).read .real)
      (fun x => ((run rest c ps s).mem outputRegion x).read .real) := by
  constructor
  · intro row head dim hr hh hd
    dsimp only
    rw [run_read rest c ps s source dest h unique (fun p hp => (complete p).mp hp)
      row head dim ((complete row).mpr hr) hh hd,Cell.read_same]
  · intro x miss
    dsimp only
    rw [run_frame rest c ps s source dest h x]
    intro p hp head dim hh hd same
    exact miss ⟨p,head,dim,(complete p).mp hp,hh,hd,same⟩

#axiomsClean step_read
#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Grid
