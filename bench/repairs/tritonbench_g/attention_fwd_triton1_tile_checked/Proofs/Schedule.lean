import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Schedule
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001 TermBody SharedStep
noncomputable section
set_option maxHeartbeats 8000000

theorem main_disjoint (c : Shape) (head other row otherRow j otherJ : Nat)
    (different : head≠other) (hr : row<c.T) (hor : otherRow<c.T) (hj : j<c.D) (hoj : otherJ<c.D) :
    c.mainOffset head row j≠c.mainOffset other otherRow otherJ := by
  intro equal
  have digits := digit_injective c.headSize head (row*c.D+j) other (otherRow*c.D+otherJ)
    (digit_bound _ _ _ _ hr hj) (digit_bound _ _ _ _ hor hoj)
    (by simpa [Shape.mainOffset,Nat.add_assoc] using equal)
  exact different digits.1

theorem history_disjoint (c : Shape) (head other chunk otherChunk i otherI j otherJ : Nat)
    (different : head≠other) (hc : chunk<c.chunks) (hoc : otherChunk<c.chunks)
    (hi : i<c.D) (hoi : otherI<c.D) (hj : j<c.D) (hoj : otherJ<c.D) :
    c.historyOffset head chunk i j≠c.historyOffset other otherChunk otherI otherJ := by
  intro equal
  have a := digit_bound _ _ _ _ (digit_bound _ _ _ _ hc hi) hj
  have b := digit_bound _ _ _ _ (digit_bound _ _ _ _ hoc hoi) hoj
  have digits := digit_injective c.historyHead head ((chunk*c.D+i)*c.D+j)
    other ((otherChunk*c.D+otherI)*c.D+otherJ) a b
    (by simpa [Shape.historyOffset,Nat.add_assoc] using equal)
  exact different digits.1

theorem step_output_other [Inhabited A] (c : Shape) (scale : ℝ) (head chunk other row : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix A c.D c.D) (s : State (Term A)) (j : Fin c.D)
    (different : other≠head) (hr : row<c.T) :
    (finished c scale head chunk storeFlag ifcond carry s).mem "o" (c.mainOffset other row j.val)=
      s.mem "o" (c.mainOffset other row j.val) := by
  rw [finished,scatter_frame,updated_memory]
  · exact saved_input c head chunk storeFlag carry s "o" _ (by decide)
  · right
    rintro ⟨i,k,⟨⟩⟩ active equal
    simp only [TileShape.indexToList,main_bound,decide_eq_true_eq] at active
    simp only [TileShape.indexToList,main_address] at equal
    exact main_disjoint c other head row (chunk*32+i.val) j.val k.val different hr active j.isLt k.isLt equal

theorem step_history_other [Inhabited A] (c : Shape) (scale : ℝ) (head chunk other otherChunk : Nat)
    (storeFlag ifcond : Bool) (carry : Matrix A c.D c.D) (s : State (Term A)) (i j : Fin c.D)
    (different : other≠head) (hc : chunk<c.chunks) (hoc : otherChunk<c.chunks) :
    (finished c scale head chunk storeFlag ifcond carry s).mem "h" (c.historyOffset other otherChunk i.val j.val)=
      s.mem "h" (c.historyOffset other otherChunk i.val j.val) := by
  rw [finished,scatter_other _ _ _ _ _ _ "h" _ (by change ("h" : RegionName)≠"o"; decide),updated_memory]
  cases storeFlag with
  | «false» => rfl
  | «true» =>
    apply scatter_frame
    right
    rintro ⟨a,b,⟨⟩⟩ _ equal
    simp only [TileShape.indexToList,history_address] at equal
    exact history_disjoint c other head otherChunk chunk i.val a.val j.val b.val
      different hoc hc i.isLt a.isLt j.isLt b.isLt equal

theorem loop_output_other [Inhabited A] (c : Shape) (scale : ℝ) (head other row : Nat)
    (storeFlag ifcond : Bool) (s : State (Term A)) (n : Nat) (j : Fin c.D)
    (different : other≠head) (hr : row<c.T) :
    (SharedLoop.state c scale head storeFlag ifcond s n).mem "o" (c.mainOffset other row j.val)=
      s.mem "o" (c.mainOffset other row j.val) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [SharedLoop.state,step_output_other _ _ _ _ _ _ _ _ _ _ _ different hr]; exact ih

theorem loop_history_other [Inhabited A] (c : Shape) (scale : ℝ) (head other chunk : Nat)
    (storeFlag ifcond : Bool) (s : State (Term A)) (n : Nat) (hn : n≤c.chunks) (i j : Fin c.D)
    (different : other≠head) (hc : chunk<c.chunks) :
    (SharedLoop.state c scale head storeFlag ifcond s n).mem "h" (c.historyOffset other chunk i.val j.val)=
      s.mem "h" (c.historyOffset other chunk i.val j.val) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [SharedLoop.state,step_history_other _ _ _ _ _ _ _ _ _ _ _ _ different (by omega) hc]
    exact ih (by omega)

/-- A schedule is a finite list of program IDs. The recursion executes the
tail before the head; arbitrary permutations of the valid grid are included. -/
def state [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) : List Nat → State (Term A)
  | [] => s
  | head::order => SharedLoop.state c scale head storeFlag ifcond
      (SharedGrid.atProgram c head (state c scale storeFlag ifcond s order)) c.chunks

def execute [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) : List Nat → Option (State (Term A))
  | [] => some s
  | head::order => do
      let before ← execute c scale storeFlag ifcond s order
      FP.Structural.exec algebra (Program.kernel c scale storeFlag ifcond) (SharedGrid.atProgram c head before)

theorem success [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) (order : List Nat) :
    execute c scale storeFlag ifcond s order=some (state c scale storeFlag ifcond s order) := by
  induction order with
  | nil => rfl
  | cons head order ih =>
    simp only [execute,ih]
    exact SharedLoop.kernel_run c scale head storeFlag ifcond _ (by simp [SharedGrid.atProgram])

theorem input [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) (order : List Nat) (r : RegionName) (off : Nat) (ho : r≠"o") (hh : r≠"h") :
    (state c scale storeFlag ifcond s order).mem r off=s.mem r off := by
  induction order with
  | nil => rfl
  | cons head order ih => rw [state,SharedLoop.input _ _ _ _ _ _ _ r off ho hh]; exact ih

theorem reads [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) (order : List Nat) (r : RegionName) (ho : r≠"o") (hh : r≠"h") :
    read (state c scale storeFlag ifcond s order) r=read s r := by
  funext off
  unfold SharedStep.read
  rw [input c scale storeFlag ifcond s order r off ho hh]

theorem output [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) (order : List Nat) (head : Nat) (member : head∈order)
    (row : Nat) (hr : row<c.T) (j : Fin c.D) :
    (state c scale storeFlag ifcond s order).mem "o" (c.mainOffset head row j.val)=
      .mk .real (TermBody.cast (Graph.output c (read s "q") (read s "k") (read s "v") scale head
        (row/32) ifcond (⟨row%32,Nat.mod_lt _ (by decide)⟩,j,PUnit.unit))) := by
  induction order with
  | nil => simp at member
  | cons current order ih =>
    by_cases same : head=current
    · subst current
      have decomposition : row/32*32+row%32=row := by omega
      have out := SharedLoop.output c scale head storeFlag ifcond
        (SharedGrid.atProgram c head (state c scale storeFlag ifcond s order)) c.chunks (row/32)
        (c.chunk_of_row row hr) ⟨row%32,Nat.mod_lt _ (by decide)⟩ j (by simpa [decomposition] using hr)
      change (state c scale storeFlag ifcond s (head::order)).mem "o"
        (mainAddress c head (row/32) (row%32) j.val)=_ at out
      simp only [mainAddress,decomposition] at out
      change _=Cell.mk .real (TermBody.cast (Graph.output c
        (read (state c scale storeFlag ifcond s order) "q")
        (read (state c scale storeFlag ifcond s order) "k")
        (read (state c scale storeFlag ifcond s order) "v") scale head (row/32) ifcond _)) at out
      rw [reads c scale storeFlag ifcond s order "q" (by decide) (by decide),
        reads c scale storeFlag ifcond s order "k" (by decide) (by decide),
        reads c scale storeFlag ifcond s order "v" (by decide) (by decide)] at out
      exact out
    · rw [state,loop_output_other _ _ _ _ _ _ _ _ _ _ same hr]
      exact ih (by simpa [same] using member)

theorem history [Inhabited A] (c : Shape) (scale : ℝ) (ifcond : Bool)
    (s : State (Term A)) (order : List Nat) (head : Nat) (member : head∈order)
    (chunk : Nat) (hc : chunk<c.chunks) (i j : Fin c.D) :
    (state c scale Bool.true ifcond s order).mem "h" (c.historyOffset head chunk i.val j.val)=
      .mk .real (TermBody.cast (Graph.carry c (read s "k") (read s "v") head ifcond chunk (i,j,PUnit.unit))) := by
  induction order with
  | nil => simp at member
  | cons current order ih =>
    by_cases same : head=current
    · subst current
      have out := SharedLoop.history c scale head ifcond
        (SharedGrid.atProgram c head (state c scale Bool.true ifcond s order)) c.chunks chunk hc i j
      change _=Cell.mk .real (TermBody.cast (Graph.carry c
        (read (state c scale Bool.true ifcond s order) "k")
        (read (state c scale Bool.true ifcond s order) "v") head ifcond chunk (i,j,PUnit.unit))) at out
      rw [reads c scale Bool.true ifcond s order "k" (by decide) (by decide),
        reads c scale Bool.true ifcond s order "v" (by decide) (by decide)] at out
      exact out
    · rw [state,loop_history_other _ _ _ _ _ _ _ _ _ (by rfl) _ _ same hc]
      exact ih (by simpa [same] using member)

theorem frame [Inhabited A] (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool)
    (s : State (Term A)) (order : List Nat) (valid : ∀ head∈order, head<c.heads)
    (r : RegionName) (off : Nat) (ho : r≠"o" ∨ c.mainSize≤off)
    (hh : storeFlag=Bool.false ∨ r≠"h" ∨ c.historySize≤off) :
    (state c scale storeFlag ifcond s order).mem r off=s.mem r off := by
  induction order with
  | nil => rfl
  | cons head order ih =>
    rw [state,SharedLoop.frame _ _ _ _ _ _ (valid head (by simp)) c.chunks (by rfl) r off ho hh]
    exact ih (fun h hm => valid h (by simp [hm]))

#axiomsClean success
#axiomsClean output
#axiomsClean history
#axiomsClean frame
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Schedule
