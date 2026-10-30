import bench.repairs.tritonbench_g.l2_norm_triton1.Kernels
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.L2NormTriton1.Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 4000000

structure Layout where
  input : Allocation
  output : Allocation
  rows : Nat
  cols : Nat
  block : Nat

def Layout.count (p : Layout) : Nat := p.rows*p.cols

def Layout.xkey (p : Layout) (row col : Nat) : Key := p.input.key (row*p.cols+col)
def Layout.ykey (p : Layout) (row col : Nat) : Key := p.output.key (row*p.cols+col)

structure Valid (p : Layout) : Prop where
  positive : 0<p.cols
  blockCovers : p.cols≤p.block
  blockLimit : p.block≤16384
  inputCapacity : p.count≤p.input.words
  outputCapacity : p.count≤p.output.words
  separate : p.input.separate p.output

theorem index_bound (p : Layout) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    row*p.cols+col<p.count := by
  unfold Layout.count
  have h := Nat.mul_le_mul_right p.cols (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at h
  omega

theorem xbacked (p : Layout) (v : Valid p) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    p.input.contains (p.xkey row col) :=
  Allocation.key_contains _ _ (lt_of_lt_of_le (index_bound p row col hr hc) v.inputCapacity)

theorem ybacked (p : Layout) (v : Valid p) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    p.output.contains (p.ykey row col) :=
  Allocation.key_contains _ _ (lt_of_lt_of_le (index_bound p row col hr hc) v.outputCapacity)

/-- Bounds derive from successful byte allocation; no new input-size guard is
introduced. Every active scalar occupies four valid representable bytes. -/
theorem active_address (p : Layout) (v : Valid p) (row col : Nat)
    (hr : row<p.rows) (hc : col<p.cols) :
    row*p.cols+col<2^62 ∧
    (p.xkey row col).address+4≤2^64 ∧ (p.ykey row col).address+4≤2^64 := by
  have hx := xbacked p v row col hr hc
  have hy := ybacked p v row col hr hc
  have ib := index_bound p row col hr hc
  have capacity := v.inputCapacity
  have representable := p.input.representable
  have representableY := p.output.representable
  simp only [Allocation.contains,Layout.xkey,Layout.ykey,Allocation.key] at hx hy
  simp only [Layout.count,Allocation.words] at ib capacity
  constructor
  · omega
  · exact ⟨le_trans hx.2.2 representable,le_trans hy.2.2 representableY⟩

/-- Even padded lane offsets remain signed64 integers. Masked pointer values
need not lie in the allocation; only active accesses dereference them. -/
theorem padded_index (p : Layout) (v : Valid p) (row lane : Nat)
    (hr : row<p.rows) (hl : lane<p.block) : row*p.cols+lane<2^63 := by
  have active := (active_address p v row 0 hr v.positive).1
  have block := v.blockLimit
  omega

def rowInputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (p : Layout) (s : M) (row : Nat) : Nat → α :=
  fun col => if col<p.cols then ops.read s (p.xkey row col) else default

def tile {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M) : M :=
  stores ops (p.ykey row) (compute (rowInputs ops p s row)) (List.range p.cols) s

theorem tile_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (owner address : Nat) (outside : ∀ col,col<p.cols → ¬(p.ykey row col).covers owner address) :
    ops.observe (tile ops compute p row s) owner address=ops.observe s owner address :=
  bytes_frame ops _ _ _ _ _ _ (fun col hc => outside col (List.mem_range.mp hc))

theorem tile_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (observed : Key) (outside : ∀ col,col<p.cols → observed.disjoint (p.ykey row col)) :
    ops.read (tile ops compute p row s) observed=ops.read s observed :=
  read_frame ops _ _ _ _ _ (fun col hc => outside col (List.mem_range.mp hc))

theorem tile_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (row : Nat) (s : M)
    (col : Nat) (hc : col<p.cols) :
    ops.read (tile ops compute p row s) (p.ykey row col)=compute (rowInputs ops p s row) col := by
  apply readback ops _ _ _ _ col (List.mem_range.mpr hc)
  · intro j hj same
    have h := congrArg Key.address same
    simp only [Layout.ykey,Allocation.key] at h
    have : j=col := by omega
    rw [this]
  · intro j hj different
    apply Allocation.indices_disjoint
    intro equal
    exact different (congrArg p.output.key equal.symm)

theorem tile_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (row other : Nat) (hr : row<p.rows) (ho : other<p.rows) (s : M) :
    rowInputs ops p (tile ops compute p row s) other=rowInputs ops p s other := by
  funext col
  unfold rowInputs
  split
  · rename_i hc
    apply tile_frame
    intro j hj
    exact Allocation.spans_separate _ _ v.separate _ _
      (xbacked p v other col ho hc) (ybacked p v row j hr hj)
  · rfl

def run {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) : List Nat → M → M
  | [],s => s
  | row::rest,s => run ops compute p rest (tile ops compute p row s)

theorem run_inputs {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (schedule : List Nat) (bounded : ∀ row∈schedule,row<p.rows) (row : Nat)
    (hr : row<p.rows) (s : M) :
    rowInputs ops p (run ops compute p schedule s) row=rowInputs ops p s row := by
  induction schedule generalizing s with
  | nil => rfl
  | cons head rest ih =>
    rw [run,ih (fun j hj => bounded j (by simp [hj])) _]
    exact tile_inputs ops compute p v head row (bounded head (by simp)) hr s

theorem run_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (schedule : List Nat) (s : M)
    (owner address : Nat)
    (outside : ∀ row∈schedule,∀ col,col<p.cols → ¬(p.ykey row col).covers owner address) :
    ops.observe (run ops compute p schedule s) owner address=ops.observe s owner address := by
  induction schedule generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_bytes ops compute p row s owner address (outside row (by simp))

theorem run_frame {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (schedule : List Nat) (s : M)
    (observed : Key)
    (outside : ∀ row∈schedule,∀ col,col<p.cols → observed.disjoint (p.ykey row col)) :
    ops.read (run ops compute p schedule s) observed=ops.read s observed := by
  induction schedule generalizing s with
  | nil => rfl
  | cons row rest ih =>
    rw [run,ih _ (fun j hj => outside j (by simp [hj]))]
    exact tile_frame ops compute p row s observed (outside row (by simp))

theorem run_values {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout) (v : Valid p)
    (schedule : List Nat) (bounded : ∀ row∈schedule,row<p.rows) (unique : schedule.Nodup)
    (row col : Nat) (member : row∈schedule) (hc : col<p.cols) (s : M) :
    ops.read (run ops compute p schedule s) (p.ykey row col)=compute (rowInputs ops p s row) col := by
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
          rw [Nat.mul_comm row p.cols,Nat.mul_add_div v.positive,
            Nat.mul_comm other p.cols,Nat.mul_add_div v.positive,
            Nat.div_eq_of_lt hc,Nat.div_eq_of_lt hj] at equal
          omega
        exact absent (this ▸ ho)
    · rw [run,ih (fun j hj => bounded j (by simp [hj])) nodup tail]
      rw [tile_inputs ops compute p v head row (bounded head (by simp))
        (bounded row (by simp [tail])) s]

/-- A failed launch may complete any subset of valid stores from interrupted
programs, after any completed programs. These effects remain private. -/
def interrupted {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout)
    (completed : List Nat) (storesDone : List (Nat × Nat)) (s : M) : M :=
  let prior := run ops compute p completed s
  stores ops (fun ij => p.ykey ij.1 ij.2)
    (fun ij => compute (rowInputs ops p prior ij.1) ij.2) storesDone prior

theorem interrupted_bytes {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (compute : (Nat → α) → Nat → α) (p : Layout)
    (completed : List Nat) (storesDone : List (Nat × Nat)) (s : M) (owner address : Nat)
    (whole : ∀ row∈completed,∀ col,col<p.cols → ¬(p.ykey row col).covers owner address)
    (part : ∀ ij∈storesDone,¬(p.ykey ij.1 ij.2).covers owner address) :
    ops.observe (interrupted ops compute p completed storesDone s) owner address=ops.observe s owner address := by
  unfold interrupted
  rw [bytes_frame ops _ _ _ _ _ _ part]
  exact run_bytes ops compute p completed s owner address whole

#axiomsClean active_address
#axiomsClean padded_index
#axiomsClean run_values
#axiomsClean interrupted_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Physical
