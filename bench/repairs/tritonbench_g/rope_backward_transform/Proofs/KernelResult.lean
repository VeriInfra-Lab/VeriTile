import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
open scoped Classical
set_option maxHeartbeats 5000000

/-- Independent scalar result using actual Tensor logical coordinates. -/
def tensorScalar (op : Operations α) (embed : β→α) (c : Call) (before : Nat→β)
    (key : Bool) (row head col : Nat) : α :=
  if col<2*c.geometry.half then
    let j:=if col<c.geometry.half then col else col-c.geometry.half
    scalarPair op (decide (c.geometry.half≤col))
      (embed (logicalRead before (c.q key) (c.geometry.offset key row head j)))
      (embed (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half))))
      (embed (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j)))
      (embed (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j)))
  else embed (logicalRead before (c.q key) (c.geometry.offset key row head col))

theorem scalar_binding {α β : Type} [Inhabited α] (op : Operations α) (embed : β→α)
    (c : Call) (admit : Admitted c) (before : Nat→β) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) :
    scalarResult op c.geometry (memory embed before c) key row head col=
      tensorScalar op embed c before key row head col := by
  have qr (j : Nat) (hj : j<c.geometry.dimension) :=
    q_read embed before c admit key _ (offset_bound c.geometry key row head j hr hh hj)
  have seq : 0<c.geometry.sequence := by
    by_contra hn
    have hz : c.geometry.sequence=0 := by omega
    simp [Request.rows,hz] at hr
  have table (sine : Bool) (j : Nat) (hj : j<c.geometry.half) :=
    table_read embed before c admit sine (row%c.geometry.sequence) j
      (lt_of_lt_of_le (Nat.mod_lt _ seq) (admit.rows sine)) hj
  unfold scalarResult tensorScalar
  split
  · rename_i active
    let j:=if col<c.geometry.half then col else col-c.geometry.half
    have hj : j<c.geometry.half := by dsimp [j]; split <;> omega
    have dim : 2*c.geometry.half≤c.geometry.dimension := by unfold Request.half; omega
    have a:=qr j (by omega)
    have b:=qr (j+c.geometry.half) (by omega)
    have cos:=table Bool.false j hj
    have sin:=table Bool.true j hj
    simp only [Request.rowStride,Request.colStride,Bool.false_eq_true,if_false,if_true] at cos sin
    dsimp only
    rw [a,b,cos,sin]
  · exact qr col hc

/-- Actual typed JIT execution for each completed row, with shared memory
threaded between rows. This is a kernel grid relation, without host dispatch. -/
inductive Execution {α : Type} [Inhabited α] (M : Algebra α) (c : Call) :
    List Nat→Source.Memory α→Source.Memory α→Prop
  | nil (m) : Execution M c [] m m
  | cons {xs m current after} (row : Nat) : Execution M c xs m current →
      (∃ final,FP.Structural.exec M c.source (state current row)=some final ∧ final.mem=after) →
      Execution M c (row::xs) m after

theorem grid_success {α : Type} [Inhabited α] (M : Algebra α) (c : Call)
    (m : Source.Memory α) (xs : List Nat) : Execution M c xs m (grid M c.geometry m xs) := by
  induction xs with
  | nil => exact .nil m
  | cons row xs ih =>
    apply Execution.cons row ih
    rw [source_binding]
    exact row_execution M c.geometry row _

theorem launch_member (c : Call) (row : Nat) (h : row∈launchRows c.geometry) : row<c.geometry.rows := by
  unfold launchRows at h
  split at h
  · exact List.mem_range.mp h
  · simp at h

theorem grid_values {α β : Type} [Inhabited α] (op : Operations α) (embed : β→α)
    (c : Call) (admit : Admitted c) (before : Nat→β) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) :
    ((grid op.backend c.geometry (memory embed before c) (launchRows c.geometry))
      (region key) (c.geometry.offset key row head col)).read .fp32=
      tensorScalar op embed c before key row head col := by
  by_cases on : launched c.geometry
  · rw [launchRows,if_pos on,grid_readback op _ _ _ List.nodup_range row head col key
      (List.mem_range.mpr hr) hh hc]
    exact scalar_binding op embed c admit before key row head col hr hh hc
  · have half : c.geometry.half=0 := by
      by_contra positive
      apply on
      refine ⟨by omega,by omega,?_⟩
      cases key <;> simp only [Request.heads,Bool.false_eq_true,↓reduceIte] at hh
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    rw [launchRows,if_neg on,grid]
    simp only [tensorScalar,half,Nat.mul_zero,Nat.not_lt_zero,↓reduceIte]
    exact q_read embed before c admit key _ (offset_bound _ _ _ _ _ hr hh hc)

/-- Complete word footprint of the selected grid, including both Q and K. -/
def Written (r : Request) (key : Bool) (offset : Nat) : Prop :=
  ∃ row head col,row<r.rows ∧ head<r.heads key ∧ col<2*r.half ∧ offset=r.offset key row head col

def WrittenRegion (r : Request) (name : RegionName) (offset : Nat) : Prop :=
  ∃ key,name=region key ∧ Written r key offset

theorem written_bound (r : Request) (key : Bool) (offset : Nat) (h : Written r key offset) : offset<r.size key := by
  rcases h with ⟨row,head,col,hr,hh,hc,rfl⟩
  have dim : 2*r.half≤r.dimension := by unfold Request.half; omega
  exact offset_bound _ _ _ _ _ hr hh (by omega)

theorem grid_exterior {α β : Type} [Inhabited α] (op : Operations α) (embed : β→α)
    (c : Call) (before : Nat→β) (name : RegionName) (offset : Nat)
    (outside : ¬WrittenRegion c.geometry name offset) :
    grid op.backend c.geometry (memory embed before c) (launchRows c.geometry) name offset=
      memory embed before c name offset := by
  apply grid_frame
  intro row member key side
  by_cases same : name=region key
  · right
    intro i active equal
    apply outside
    obtain ⟨hh,hj⟩ := (active_iff _ _ i).mp active
    refine ⟨key,same,row,i.1.val,i.2.1.val+if side then c.geometry.half else 0,
      launch_member c row member,hh,?_,?_⟩
    · cases side <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
    · simpa only [address,address_coordinate] using equal
  · exact Or.inl same

#axiomsClean scalar_binding
#axiomsClean grid_success
#axiomsClean grid_values
#axiomsClean grid_exterior
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
