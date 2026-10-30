import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Coverage

namespace VeriTile.Bench.Repairs.KldivOps.FPGraph
open VeriTile.Triton FP.Structural
open Program Tile
noncomputable section
set_option maxHeartbeats 6000000

/-- Ordered FP32 vector reduction, with an explicit zero in every inactive
lane. This is a local reduction port, not a reassociated scalar sum. -/
def block (M : Algebra α) (cols width : Nat) (flag : Bool) (eps : α)
    (prediction target : Nat→α) (start : Nat) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
    (fun j : TileIndex [width] => if start+j.1.val<cols then
      scalar M flag eps (prediction (start+j.1.val)) (target (start+j.1.val)) else zero M) PUnit.unit

def accumulate (M : Algebra α) (cols width : Nat) (flag : Bool) (eps : α)
    (prediction target : Nat→α) (start : Nat) (carry : α) : α :=
  if h : 0<width ∧ start<cols then
    accumulate M cols width flag eps prediction target (start+width)
      (M.binary (some .fp32) .real .add carry (block M cols width flag eps prediction target start))
  else carry
termination_by cols-start
decreasing_by omega

def row (M : Algebra α) (cols width : Nat) (flag : Bool) (eps : α)
    (prediction target : Nat→α) : α := accumulate M cols width flag eps prediction target 0 (zero M)

theorem block_binding {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row start : Nat) (mem : Memory α) :
    chunkSum M p flag eps row start (memoryState mem)=block M p.cols p.block flag eps
      (fun i => (mem p.Y (row*p.cols+i)).read .real)
      (fun i => (mem p.T (row*p.cols+i)).read .real) start := by
  unfold chunkSum block
  congr 1
  funext j
  by_cases active : start+j.1.val<p.cols <;> simp [active,lane,memoryState]

theorem loop_binding {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (start : Nat) (mem : Memory α) (carry : α) (hr : red≠0) :
    loopModel M p flag red row eps start (mem,carry)=
      (mem,accumulate M p.cols p.block flag eps
        (fun i => (mem p.Y (row*p.cols+i)).read .real)
        (fun i => (mem p.T (row*p.cols+i)).read .real) start carry) := by
  rw [loopModel,accumulate]
  split
  next active =>
    simp only [advance,chunkMemory,hr,if_false,nextCarry]
    rw [show chunkSum M p flag eps row start (memoryState mem)=_ from block_binding M p flag eps row start mem]
    exact loop_binding M p flag red row eps (start+p.block) mem _ hr
  next inactive => rfl
termination_by p.cols-start
decreasing_by omega

theorem row_binding {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red rowId : Nat) (eps : α) (mem : Memory α) (hr : red≠0) :
    rowModel M p flag red rowId eps mem p.O rowId=
      .mk .real (row M p.cols p.block flag eps
        (fun i => (mem p.Y (rowId*p.cols+i)).read .real)
        (fun i => (mem p.T (rowId*p.cols+i)).read .real)) := by
  unfold rowModel
  rw [if_neg hr,loop_binding M p flag red rowId eps 0 mem (zero M) hr]
  simp [writeMemory,row]

theorem grid_binding {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (jobs : List Nat) (mem : Memory α) (rowId : Nat) (member : rowId∈jobs)
    (hr : red≠0) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    gridModel M p flag red eps jobs mem p.O rowId=
      .mk .real (row M p.cols p.block flag eps
        (fun i => (mem p.Y (rowId*p.cols+i)).read .real)
        (fun i => (mem p.T (rowId*p.cols+i)).read .real)) := by
  induction jobs generalizing mem with
  | nil => simp at member
  | cons first rest ih =>
    change gridModel M p flag red eps rest (rowModel M p flag red first eps mem) p.O rowId=_
    by_cases later : rowId∈rest
    · rw [ih _ later]
      congr 2
      · funext i
        rw [row_frame M p flag red first eps mem p.Y (rowId*p.cols+i) (by simp [Writes,hy])]
      · funext i
        rw [row_frame M p flag red first eps mem p.T (rowId*p.cols+i) (by simp [Writes,ht])]
    · have eq : rowId=first := by simpa only [List.mem_cons,later,or_false] using member
      subst first
      rw [grid_frame M p flag red eps rest _ p.O rowId (by
        intro other ho hw
        have eq : rowId=other := by simpa [Writes,hr] using hw
        exact later (eq ▸ ho))]
      exact row_binding M p flag red rowId eps mem hr

theorem block_congr (M : Algebra α) (cols width : Nat) (flag : Bool) (eps : α)
    (y₁ t₁ y₂ t₂ : Nat→α) (start : Nat)
    (hy : ∀ i,i<cols → y₁ i=y₂ i) (ht : ∀ i,i<cols → t₁ i=t₂ i) :
    block M cols width flag eps y₁ t₁ start=block M cols width flag eps y₂ t₂ start := by
  unfold block
  congr 1
  funext j
  split
  next active => rw [hy _ active,ht _ active]
  next => rfl

theorem accumulate_congr (M : Algebra α) (cols width : Nat) (flag : Bool) (eps : α)
    (y₁ t₁ y₂ t₂ : Nat→α) (start : Nat) (carry : α)
    (hy : ∀ i,i<cols → y₁ i=y₂ i) (ht : ∀ i,i<cols → t₁ i=t₂ i) :
    accumulate M cols width flag eps y₁ t₁ start carry=accumulate M cols width flag eps y₂ t₂ start carry := by
  conv_lhs => rw [accumulate]
  conv_rhs => rw [accumulate]
  split
  next active =>
    rw [block_congr M cols width flag eps y₁ t₁ y₂ t₂ start hy ht]
    exact accumulate_congr M cols width flag eps y₁ t₁ y₂ t₂ (start+width) _ hy ht
  next inactive => rfl
termination_by cols-start
decreasing_by omega

#specNonCircular row avoiding [Program.grid,Program.gridModel,Program.loopModel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.FPGraph
