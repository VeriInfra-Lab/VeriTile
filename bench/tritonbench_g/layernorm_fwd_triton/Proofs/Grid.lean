import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Execution

namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Execution
open Address Memory
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
variable {shape : Shape} {strides : Strides} {α Store Byte : Type}

def executeGrid (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (eps : α) (rows : List (Row shape)) (m : Store) : Store :=
  rows.foldl (fun state row => executeRow ops plan api layout eps row state) m

def gridTrace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α) (rows : List (Row shape)) : List (Write α) :=
  rows.flatMap (rowTrace ops plan api layout initial eps)

theorem run_append (api : API α Store Byte) (a b : List (Write α)) (m : Store) :
    run api (a++b) m=run api b (run api a m) := by simp [run,List.foldl_append]

theorem grid_trace (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial current : Store) (h : InputsEqual api layout initial current)
    (eps : α) (rows : List (Row shape)) :
    executeGrid ops plan api layout eps rows current=run api (gridTrace ops plan api layout initial eps rows) current  ∧ 
    InputsEqual api layout initial (executeGrid ops plan api layout eps rows current) := by
  induction rows generalizing current with
  | nil => exact ⟨rfl,h⟩
  | cons row rows ih =>
    obtain ⟨he,hi⟩ := row_trace ops plan api layout initial current h eps row
    have ht := ih (executeRow ops plan api layout eps row current) hi
    refine ⟨?_,ht.2⟩
    change executeGrid ops plan api layout eps rows (executeRow ops plan api layout eps row current)=_
    rw [ht.1]
    simp only [gridTrace,List.flatMap_cons,run_append,he]

theorem grid_trace_member (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α) (rows : List (Row shape)) (e : Write α)
    (h : e∈gridTrace ops plan api layout initial eps rows) :
    ∃ row∈rows, ∃ col : Fin shape.columns,
      e.address=yAddress layout row col  ∧ 
      e.value=Native.reference ops plan (xSnapshot ops api layout initial row)
        (wSnapshot ops api layout initial row) eps col.val := by
  obtain ⟨row,hr,he⟩ := List.mem_flatMap.mp h
  obtain ⟨col,_,hc⟩ := List.mem_map.mp he
  subst e
  exact ⟨row,hr,col,rfl,rfl⟩

/-- Full grid readback for any actual completed scheduling order containing
all programs. No serial numerical associativity is needed: each row reads only
inputs, and the fresh output windows are disjoint across all programs. -/
theorem grid_readback (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α) (rows : List (Row shape))
    (all : ∀ row : Row shape, row∈rows) (row : Row shape) (col : Fin shape.columns) :
    api.read (executeGrid ops plan api layout eps rows initial) (yAddress layout row col)=
      Native.reference ops plan (xSnapshot ops api layout initial row)
        (wSnapshot ops api layout initial row) eps col.val := by
  rw [(grid_trace ops plan api layout initial initial (inputs_refl api layout initial) eps rows).1]
  apply run_readback
  · refine ⟨⟨yAddress layout row col,Native.reference ops plan (xSnapshot ops api layout initial row)
      (wSnapshot ops api layout initial row) eps col.val⟩,?_,rfl,rfl⟩
    apply List.mem_flatMap.mpr
    refine ⟨row,all row,?_⟩
    exact List.mem_map.mpr ⟨col,mem_columns _ 0 col (Nat.zero_le _),rfl⟩
  · intro e he
    obtain ⟨r,_,j,ha,hv⟩ := grid_trace_member ops plan api layout initial eps rows e he
    by_cases eq : (r,j)=(row,col)
    · have hr : r=row := congrArg Prod.fst eq
      have hj : j=col := congrArg Prod.snd eq
      subst r
      subst j
      exact Or.inl ⟨ha,hv⟩
    · right
      rw [ha]
      exact outputs_apart layout (r,j) (row,col) eq

/-- Every byte outside all actual active output windows is unchanged. This
covers the whole caller backing, not only original view coordinates. -/
theorem grid_frame (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α) (rows : List (Row shape))
    (i : Nat) (outside : ∀ row col, i < yAddress layout row col  ∨  yAddress layout row col+4 ≤ i) :
    api.byte (executeGrid ops plan api layout eps rows initial) i=api.byte initial i := by
  rw [(grid_trace ops plan api layout initial initial (inputs_refl api layout initial) eps rows).1]
  apply run_byte_frame
  intro e he
  obtain ⟨r,_,j,ha,_⟩ := grid_trace_member ops plan api layout initial eps rows e he
  rw [ha]
  exact outside r j

theorem empty_grid (ops : Native.Ops α) (plan : Native.Plan shape.columns) (api : API α Store Byte)
    (layout : Layout shape strides) (initial : Store) (eps : α)
    (h : shape.sequences=0  ∨  shape.heads=0) :
    executeGrid ops plan api layout eps (allRows shape) initial=initial := by
  have noRow : ∀ row : Row shape, False := by
    intro row
    have hs := row.1.isLt
    have hh := row.2.isLt
    rcases h with h|h <;> omega
  have he : allRows shape=[] := by
    cases hl : allRows shape with
    | nil => rfl
    | cons row rows => exact False.elim (noRow row)
  simp [executeGrid,he]

#axiomsClean grid_trace
#axiomsClean grid_readback
#axiomsClean grid_frame
#axiomsClean empty_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Execution
end VeriTile.Bench.Repairs.LayernormFwdTriton
