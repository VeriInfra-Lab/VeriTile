import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000

theorem offset_bound (r : Request) (key : Bool) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads key) (hc : col<r.dimension) :
    r.offset key row head col<RequestSize r key :=
  lt_of_lt_of_le (offset_window r key row head col hh hc).2
    (Nat.mul_le_mul_right (r.stride key) (by omega : row+1≤r.rows))

theorem coefficient_bound (r : Request) (row lane : Nat) (hr : row<r.rows) (hl : lane<r.half) :
    row%r.sequence*r.half+lane<r.sequence*r.half := by
  have seq : 0<r.sequence := by
    by_contra h
    have zero : r.sequence=0 := by omega
    simp [Request.rows,zero] at hr
  have hm:=Nat.mod_lt row seq
  have prod:=Nat.mul_le_mul_right r.half (show row%r.sequence+1≤r.sequence by omega)
  simp only [Nat.add_mul,Nat.one_mul] at prod
  omega

def tensorScalar {α β : Type} (M : Algebra α) (embed : β→α) (c : Call) (before : Nat→β)
    (key : Bool) (row head col : Nat) : α :=
  let j:=if col<c.geometry.half then col else col-c.geometry.half
  scalar M (embed (logicalRead before (c.q key) (c.geometry.offset key row head j)))
    (embed (logicalRead before (c.q key) (c.geometry.offset key row head (j+c.geometry.half))))
    (embed (logicalRead before (c.tables Bool.false) (row%c.geometry.sequence*c.geometry.half+j)))
    (embed (logicalRead before (c.tables Bool.true) (row%c.geometry.sequence*c.geometry.half+j)))
    (decide (c.geometry.half≤col))

def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (initial : State α) : Nat→Option (State α)
  | 0=>some initial
  | n+1=>(sourceRun M c initial n).bind (fun s=>FP.Structural.exec M c.source (Grid.setup c.geometry n s))

theorem sourceRun_binding {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (s : State α) (n : Nat) :
    sourceRun M c s n=Grid.sourceRun M c.geometry s n := by
  induction n with
  | zero=>rfl
  | succ n ih=>simp only [sourceRun,Grid.sourceRun,ih,source_binding]

theorem source_success {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (s : State α) :
    sourceRun M c s c.geometry.rows=some (Grid.run M c.geometry s c.geometry.rows) := by
  rw [sourceRun_binding]
  exact Grid.successful M c.geometry s c.geometry.rows

theorem grid_values {α β : Type} [Inhabited α] (M : Algebra α) (embed : β→α)
    (c : Call) (admit : Admitted c) (before : Nat→β) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) :
    ((Grid.run M c.geometry (entry embed before c) c.geometry.rows).mem
      (region key) (c.geometry.offset key row head col)).read .fp32=
      tensorScalar M embed c before key row head col := by
  let j:=if col<c.geometry.half then col else col-c.geometry.half
  have hj : j<c.geometry.half := by dsimp [j]; unfold Request.dimension at hc; split <;> omega
  have column : c.geometry.column (decide (c.geometry.half≤col)) j=col := by
    dsimp [Request.column,j]
    by_cases low : col<c.geometry.half
    · simp [low,show ¬c.geometry.half≤col from by omega]
    · simp [low,show c.geometry.half≤col from by omega]
  have out:=Grid.outputs M c.geometry (entry embed before c) c.geometry.rows
    key row head (decide (c.geometry.half≤col)) j hr hh hj
  rw [column] at out
  rw [out]
  change value M c.geometry (entry embed before c) key row head (decide (c.geometry.half≤col)) j=_
  have ha:=offset_bound c.geometry key row head j hr hh (by unfold Request.dimension; omega)
  have hb:=offset_bound c.geometry key row head (j+c.geometry.half) hr hh (by unfold Request.dimension; omega)
  have ht:=coefficient_bound c.geometry row j hr hj
  have cos:=table_read embed before c admit Bool.false _ ht
  have sin:=table_read embed before c admit Bool.true _ ht
  simp only [Bool.false_eq_true,if_false,if_true] at cos sin
  unfold value tensorScalar
  change scalar M (((memory embed before c) (region key) _).read .fp32)
    (((memory embed before c) (region key) _).read .fp32)
    (((memory embed before c) "C" _).read .fp32) (((memory embed before c) "S" _).read .fp32) _=_
  rw [q_read embed before c admit key _ ha,q_read embed before c admit key _ hb,cos,sin]

/-- Every FP32 cell of each prepared Q/K buffer is written exactly once. -/
def WrittenRegion (c : Call) (name : RegionName) (index : Nat) : Prop :=
  ∃ k,name=region k ∧ index<RequestSize c.geometry k

theorem grid_frame {α β : Type} [Inhabited α] (M : Algebra α) (embed : β→α)
    (c : Call) (before : Nat→β) (name : RegionName) (index : Nat)
    (outside : ¬WrittenRegion c name index) :
    (Grid.run M c.geometry (entry embed before c) c.geometry.rows).mem name index=
      memory embed before c name index := by
  apply Grid.prefix_frame
  intro k
  by_cases same : name=region k
  · right
    by_contra small
    exact outside ⟨k,same,by unfold RequestSize; omega⟩
  · exact Or.inl same

#axiomsClean sourceRun_binding
#axiomsClean source_success
#axiomsClean grid_values
#axiomsClean grid_frame
end
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
