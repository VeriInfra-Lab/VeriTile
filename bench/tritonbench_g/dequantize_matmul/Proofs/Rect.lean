import bench.tritonbench_g.cross_entropy1.Proofs.Grid

namespace VeriTile.Bench.Repairs.DequantizeMatmul
open CrossEntropy1
noncomputable section
set_option maxHeartbeats 3000000

abbrev Address := Nat × Nat
abbrev Memory (α : Type) := Address → α
abbrev Task := Nat × Nat

structure Window where
  storage : Nat
  offset : Nat
  rows : Nat
  columns : Nat
  deriving DecidableEq

def Window.addr (w : Window) (row col : Nat) : Address :=
  (w.storage,w.offset+row*w.columns+col)
def Window.inside (w : Window) (p : Address) : Prop :=
  p.1=w.storage ∧ w.offset≤p.2 ∧ p.2<w.offset+w.rows*w.columns
def Window.row (w : Window) (p : Address) : Nat := (p.2-w.offset)/w.columns
def Window.col (w : Window) (p : Address) : Nat := (p.2-w.offset)%w.columns

def ceilDiv (n block : Nat) : Nat := n/block + if n%block=0 then 0 else 1

theorem ceil_cover (n block : Nat) (positive : 0<block) : n≤block*ceilDiv n block := by
  have split := Nat.mod_add_div n block
  have remainder := Nat.mod_lt n positive
  unfold ceilDiv
  split_ifs with h
  · simp only [Nat.add_zero]; omega
  · simp only [Nat.mul_add,Nat.mul_one]; omega

theorem owner_bound {n block col : Nat} (positive : 0<block) (active : col<n) :
    col/block<ceilDiv n block := by
  apply (Nat.div_lt_iff_lt_mul positive).mpr
  simpa only [Nat.mul_comm] using lt_of_lt_of_le active (ceil_cover n block positive)

theorem Window.address_inside (w : Window) (r c : Nat) (hr : r<w.rows) (hc : c<w.columns) :
    w.inside (w.addr r c) := by
  refine ⟨rfl,by dsimp [addr]; omega,?_⟩
  have h := Nat.mul_le_mul_right w.columns (Nat.succ_le_of_lt hr)
  simp only [addr]
  nlinarith

theorem Window.coordinates (w : Window) (r c : Nat) (hc : c<w.columns) :
    w.row (w.addr r c)=r ∧ w.col (w.addr r c)=c := by
  have hn : 0<w.columns := Nat.zero_lt_of_lt hc
  have sub : w.offset+r*w.columns+c-w.offset=r*w.columns+c := by omega
  have hd : (r*w.columns+c)/w.columns=r := by
    rw [Nat.mul_comm,Nat.mul_add_div hn,Nat.div_eq_of_lt hc,Nat.add_zero]
  have hm : (r*w.columns+c)%w.columns=c := by
    rw [Nat.mul_comm,Nat.mul_add_mod,Nat.mod_eq_of_lt hc]
  simpa only [row,col,addr,sub] using And.intro hd hm

structure Rect where
  window : Window
  blockRows : Nat
  blockColumns : Nat

def Rect.tasks (g : Rect) : List Task :=
  CrossEntropy1.Grid.tasks (ceilDiv g.window.rows g.blockRows)
    (ceilDiv g.window.columns g.blockColumns)
def Rect.owner (g : Rect) (p : Address) : Task :=
  (g.window.row p/g.blockRows,g.window.col p/g.blockColumns)
def Rect.writes (g : Rect) (q : Task) (p : Address) : Prop :=
  g.window.inside p ∧ g.owner p=q
instance (g : Rect) (q : Task) : DecidablePred (g.writes q) := by
  intro p
  unfold Rect.writes Window.inside
  infer_instance

def Rect.step (g : Rect) (value : Memory α → Nat → Nat → α)
    (q : Task) (m : Memory α) : Memory α :=
  Schedule.put g.writes (fun _ p => value m (g.window.row p) (g.window.col p)) q m
def Rect.run (g : Rect) (value : Memory α → Nat → Nat → α) (m : Memory α) : Memory α :=
  Schedule.run (g.step value) g.tasks m

def Rect.ReadsOutside (g : Rect) (value : Memory α → Nat → Nat → α) : Prop :=
  ∀ entry current, (∀ p,p.1≠g.window.storage → current p=entry p) →
    ∀ r c,value current r c=value entry r c

theorem Rect.frozen (g : Rect) (value : Memory α → Nat → Nat → α)
    (stable : g.ReadsOutside value) (entry : Memory α) :
    g.run value entry=Schedule.run
      (Schedule.put g.writes (fun _ p => value entry (g.window.row p) (g.window.col p))) g.tasks entry := by
  apply Schedule.freeze_reads (reads:=fun _ p => p.1≠g.window.storage) (writes:=g.writes) (entry:=entry)
  · intro q current same
    funext point
    simp only [step,Schedule.put,stable entry current same]
  · intro q current point outside
    exact if_neg outside
  · exact CrossEntropy1.Grid.task_nodup _ _
  · intro q _ other _ _ point read write
    exact read write.1.1
  · intro _ _ _ _; rfl

theorem Rect.output (g : Rect) (value : Memory α → Nat → Nat → α)
    (stable : g.ReadsOutside value) (entry : Memory α)
    (br : 0<g.blockRows) (bc : 0<g.blockColumns)
    (r c : Nat) (hr : r<g.window.rows) (hc : c<g.window.columns) :
    g.run value entry (g.window.addr r c)=value entry r c := by
  have coordinates := g.window.coordinates r c hc
  have present : g.owner (g.window.addr r c)∈g.tasks := by
    apply CrossEntropy1.Grid.task_mem _ _ _ |>.mpr
    simpa only [owner,coordinates.1,coordinates.2] using
      And.intro (owner_bound br hr) (owner_bound bc hc)
  have active : g.writes (g.owner (g.window.addr r c)) (g.window.addr r c) :=
    ⟨g.window.address_inside r c hr hc,rfl⟩
  rw [g.frozen value stable entry,Schedule.fixed_output g.writes
    (fun _ p => value entry (g.window.row p) (g.window.col p))
    g.tasks entry (g.owner (g.window.addr r c)) _ active]
  · simp only [if_pos present,coordinates.1,coordinates.2]
  · intro q _ writes; exact writes.2.symm

theorem Rect.frame (g : Rect) (value : Memory α → Nat → Nat → α)
    (stable : g.ReadsOutside value) (entry : Memory α) (point : Address)
    (outside : ¬g.window.inside point) : g.run value entry point=entry point := by
  rw [g.frozen value stable entry]
  apply Schedule.fixed_frame
  intro q _ writes
  exact outside writes.1

#axiomsClean Window.coordinates
#axiomsClean Rect.frozen
#axiomsClean Rect.output
#axiomsClean Rect.frame
end
end VeriTile.Bench.Repairs.DequantizeMatmul
