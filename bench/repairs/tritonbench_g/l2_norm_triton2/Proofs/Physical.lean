import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Physical

namespace VeriTile.Bench.Repairs.L2NormTriton2.Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Actual independent X, DY and DX allocations, with two read roles. -/
structure Layout extends L2NormTriton1.Physical.Layout where
  gradient : Allocation

def Layout.inputFor (p : Layout) (gradient : Bool) : Allocation :=
  if gradient then p.gradient else p.input

def Layout.unary (p : Layout) (gradient : Bool) : L2NormTriton1.Physical.Layout :=
  {p.toLayout with input := p.inputFor gradient}
abbrev Layout.count (p : Layout) := p.toLayout.count
def Layout.xkey (p : Layout) (gradient : Bool) (row col : Nat) : Key :=
  (p.inputFor gradient).key (row*p.cols+col)
abbrev Layout.ykey (p : Layout) := p.toLayout.ykey

structure Valid (p : Layout) : Prop where
  main : L2NormTriton1.Physical.Valid p.toLayout
  gradientCapacity : p.count≤p.gradient.words
  gradientSeparate : p.gradient.separate p.output

theorem Valid.validFor (p : Layout) (v : Valid p) (gradient : Bool) :
    L2NormTriton1.Physical.Valid (p.unary gradient) := by
  cases gradient
  · exact v.main
  · exact ⟨v.main.positive,v.main.blockCovers,v.main.blockLimit,v.gradientCapacity,
      v.main.outputCapacity,v.gradientSeparate⟩

theorem index_bound (p : Layout) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    row*p.cols+col<p.count := L2NormTriton1.Physical.index_bound p.toLayout row col hr hc

theorem xbacked (p : Layout) (v : Valid p) (gradient : Bool) (row col : Nat)
    (hr : row<p.rows) (hc : col<p.cols) :
    (p.inputFor gradient).contains (p.xkey gradient row col) :=
  L2NormTriton1.Physical.xbacked (p.unary gradient) (Valid.validFor p v gradient) row col hr hc

theorem ybacked (p : Layout) (v : Valid p) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    p.output.contains (p.ykey row col) := L2NormTriton1.Physical.ybacked p.toLayout v.main row col hr hc

/-- Reuse allocation-derived native bounds independently for both reads. -/
theorem active_address (p : Layout) (v : Valid p) (gradient : Bool) (row col : Nat)
    (hr : row<p.rows) (hc : col<p.cols) :
    row*p.cols+col<2^62 ∧ (p.xkey gradient row col).address+4≤2^64 ∧
      (p.ykey row col).address+4≤2^64 :=
  L2NormTriton1.Physical.active_address (p.unary gradient) (Valid.validFor p v gradient) row col hr hc

theorem padded_index (p : Layout) (v : Valid p) (row lane : Nat)
    (hr : row<p.rows) (hl : lane<p.block) : row*p.cols+lane<2^63 :=
  L2NormTriton1.Physical.padded_index p.toLayout v.main row lane hr hl

def rowInputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (p : Layout) (s : M) (row : Nat) (gradient : Bool) : Nat → α :=
  fun col => if col<p.cols then ops.read s (p.xkey gradient row col) else default

def tile {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M) : M :=
  stores ops (p.ykey row) (compute (rowInputs ops p s row Bool.false) (rowInputs ops p s row Bool.true)) (List.range p.cols) s

theorem tile_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (owner address : Nat) (outside : ∀ col,col<p.cols → ¬(p.ykey row col).covers owner address) :
    ops.observe (tile ops compute p row s) owner address=ops.observe s owner address :=
  bytes_frame ops _ _ _ _ _ _ (fun col hc => outside col (List.mem_range.mp hc))

theorem tile_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (observed : Key) (outside : ∀ col,col<p.cols → observed.disjoint (p.ykey row col)) :
    ops.read (tile ops compute p row s) observed=ops.read s observed :=
  read_frame ops _ _ _ _ _ (fun col hc => outside col (List.mem_range.mp hc))

theorem tile_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (col : Nat) (hc : col<p.cols) :
    ops.read (tile ops compute p row s) (p.ykey row col)=compute (rowInputs ops p s row Bool.false) (rowInputs ops p s row Bool.true) col := by
  apply readback ops _ _ _ _ col (List.mem_range.mpr hc)
  · intro j hj same
    have h := congrArg Key.address same
    simp only [Layout.ykey,L2NormTriton1.Physical.Layout.ykey,Allocation.key] at h
    have : j=col := by omega
    rw [this]
  · intro j hj different
    apply Allocation.indices_disjoint
    intro equal
    exact different (congrArg p.output.key equal.symm)

theorem tile_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (row other : Nat) (hr : row<p.rows) (ho : other<p.rows) (s : M) (gradient : Bool) :
    rowInputs ops p (tile ops compute p row s) other gradient=rowInputs ops p s other gradient := by
  funext col
  unfold rowInputs
  split
  · rename_i hc
    apply tile_frame
    intro j hj
    exact Allocation.spans_separate _ _ (Valid.validFor p v gradient).separate _ _
      (xbacked p v gradient other col ho hc) (ybacked p v row j hr hj)
  · rfl

def run {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) : List Nat → M → M
  | [],s => s
  | row::rest,s => run ops compute p rest (tile ops compute p row s)

theorem run_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (schedule : List Nat) (bounded : ∀ row∈schedule,row<p.rows) (row : Nat)
    (hr : row<p.rows) (s : M) (gradient : Bool) :
    rowInputs ops p (run ops compute p schedule s) row gradient=rowInputs ops p s row gradient := by
  induction schedule generalizing s with
  | nil => rfl
  | cons head rest ih =>
    rw [run,ih (fun j hj => bounded j (by simp [hj])) _]
    exact tile_inputs ops compute p v head row (bounded head (by simp)) hr s gradient

theorem run_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (schedule : List Nat) (s : M)
    (owner address : Nat)
    (outside : ∀ row∈schedule,∀ col,col<p.cols → ¬(p.ykey row col).covers owner address) :
    ops.observe (run ops compute p schedule s) owner address=ops.observe s owner address := by
  induction schedule generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_bytes ops compute p row s owner address (outside row (by simp))

theorem run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (schedule : List Nat) (s : M)
    (observed : Key)
    (outside : ∀ row∈schedule,∀ col,col<p.cols → observed.disjoint (p.ykey row col)) :
    ops.read (run ops compute p schedule s) observed=ops.read s observed := by
  induction schedule generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_frame ops compute p row s observed (outside row (by simp))

theorem run_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (schedule : List Nat) (bounded : ∀ row∈schedule,row<p.rows) (unique : schedule.Nodup)
    (row col : Nat) (member : row∈schedule) (hc : col<p.cols) (s : M) :
    ops.read (run ops compute p schedule s) (p.ykey row col)=compute (rowInputs ops p s row Bool.false) (rowInputs ops p s row Bool.true) col := by
  induction schedule generalizing s with
  | nil => simp at member
  | cons head rest ih =>
    obtain ⟨absent,nodup⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp member with equal | tail
    · subst head
      rw [run,run_frame ops compute p rest _ _]
      · exact tile_values ops compute p row s col hc
      · intro other ho j hj
        apply Allocation.indices_disjoint
        intro same
        have : other=row := by
          have equal := congrArg (fun k => k/p.cols) same
          dsimp only at equal
          rw [Nat.mul_comm row p.cols,Nat.mul_add_div v.main.positive,
            Nat.mul_comm other p.cols,Nat.mul_add_div v.main.positive,
            Nat.div_eq_of_lt hc,Nat.div_eq_of_lt hj] at equal
          omega
        exact absent (this ▸ ho)
    · rw [run,ih (fun j hj => bounded j (by simp [hj])) nodup tail]
      rw [tile_inputs ops compute p v head row (bounded head (by simp))
        (bounded row (by simp [tail])) s Bool.false,
        tile_inputs ops compute p v head row (bounded head (by simp)) (bounded row (by simp [tail])) s Bool.true]

/-- A failed launch may complete any subset of valid stores from interrupted
programs, after any completed programs. These effects remain private. -/
def interrupted {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout)
    (completed : List Nat) (storesDone : List (Nat × Nat)) (s : M) : M :=
  let prior := run ops compute p completed s
  stores ops (fun ij => p.ykey ij.1 ij.2)
    (fun ij => compute (rowInputs ops p prior ij.1 Bool.false) (rowInputs ops p prior ij.1 Bool.true) ij.2) storesDone prior

theorem interrupted_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → (Nat → α) → Nat → α) (p : Layout)
    (completed : List Nat) (storesDone : List (Nat × Nat)) (s : M) (owner address : Nat)
    (whole : ∀ row∈completed,∀ col,col<p.cols → ¬(p.ykey row col).covers owner address)
    (part : ∀ ij∈storesDone,¬(p.ykey ij.1 ij.2).covers owner address) :
    ops.observe (interrupted ops compute p completed storesDone s) owner address=ops.observe s owner address := by
  unfold interrupted
  rw [bytes_frame ops _ _ _ _ _ _ part]
  exact run_bytes ops compute p completed s owner address whole

#axiomsClean run_values
#axiomsClean interrupted_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Physical
