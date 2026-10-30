import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedFinish
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Math

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.NestedLoopsProcessing
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
noncomputable section
open scoped Classical
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def offsets (cols shift : Nat) : OffsetTile := fun i ↦ address cols shift (i.1,i.2.1)

def inputCells (s : State α) : Nat → Cell α :=
  fun i ↦ .mk .int ((s.mem "input" i).read .int)

@[simp] theorem initialA_offsets (cols : Nat) : initialA cols = offsets cols 0 := by rfl
@[simp] theorem initialC_offsets (cols : Nat) : initialC cols = offsets cols 0 := by
  funext i
  simp [initialC, offsets, address, Nat.mul_comm]
@[simp] theorem bump_offsets (cols shift delta : Nat) :
    bump (offsets cols shift) delta = offsets cols (shift+delta) := by
  funext i
  simp [bump, offsets, address, Nat.add_assoc]

@[simp] theorem loaded_setReg (count : Nat) (s : State α) (n d sh v) (a : OffsetTile) :
    loaded count (s.setReg n d sh v) a = loaded count s a := rfl

@[simp] theorem scatter_other (count : Nat) (u : Unique) (s : State α)
    (a : OffsetTile) (v : Payload) (r : RegionName) (hr : r ≠ "output") :
    (scatter count u s a v).mem r = s.mem r := by
  have fold : ∀ (ls : List (TileIndex [2,2])) (s : State α),
      (ls.foldl (fun t i ↦ if decide (a i<count) && u i then
        t.write "output" (min (a i) (count-1)) (.mk .int (v i)) else t) s).mem r = s.mem r := by
    intro ls
    induction ls with
    | nil => intro s; rfl
    | cons i ls ih =>
      intro s
      rw [List.foldl_cons, ih]
      split
      · funext j; exact State.write_other _ _ _ _ _ _ (Or.inl hr)
      · rfl
  exact fold _ s

@[simp] theorem loaded_scatter (count n : Nat) (u : Unique) (s : State α)
    (a p : OffsetTile) (v : Payload) :
    loaded n (scatter count u s a v) p = loaded n s p := by
  funext i
  unfold loaded
  rw [scatter_other count u s a v "input" (by decide)]

@[simp] theorem inputCells_setReg (s : State α) (n d sh v) :
    inputCells (s.setReg n d sh v) = inputCells s := rfl

@[simp] theorem inputCells_scatter (count : Nat) (u : Unique) (s : State α)
    (a : OffsetTile) (v : Payload) :
    inputCells (scatter count u s a v) = inputCells s := by
  funext i
  unfold inputCells
  rw [scatter_other count u s a v "input" (by decide)]

theorem write_output (s : State α) (a : Nat) (v : Cell α) :
    (s.write "output" a v).mem "output" = Function.update (s.mem "output") a v := by
  funext i
  by_cases h : i=a <;> simp [State.write, Function.update, h]

theorem scatter_output (count cols shift : Nat) (s : State α) (v : Payload) :
    (scatter count (unique cols) s (offsets cols shift) v).mem "output" =
      uniqueTile count cols shift (fun l ↦ .mk .int (v (l.1,l.2,PUnit.unit))) (s.mem "output") := by
  have fold : ∀ (ls : List Lane) (s : State α),
      (ls.foldl (fun t l ↦ if decide (address cols shift l<count) && unique cols (l.1,l.2,PUnit.unit) then
        t.write "output" (min (address cols shift l) (count-1)) (.mk .int (v (l.1,l.2,PUnit.unit))) else t) s).mem "output" =
      ls.foldl (fun m l ↦ if activeLane cols l then
        put count (address cols shift l) (.mk .int (v (l.1,l.2,PUnit.unit))) m else m) (s.mem "output") := by
    intro ls
    induction ls with
    | nil => intro s; rfl
    | cons l ls ih =>
      intro s
      simp only [List.foldl_cons, ih]
      congr 1
      by_cases ha : activeLane cols l <;> by_cases hb : address cols shift l<count
      all_goals simp [unique, activeLane] at ha
      all_goals simp [unique, ha, hb, put, activeLane, write_output, or_assoc]
      all_goals rw [Nat.min_eq_left (show address cols shift l ≤ count-1 from by omega)]
      all_goals split <;> simp_all [write_output]
  have indices : TileShape.allIndices [2,2] = lanes.map (fun l ↦ (l.1,l.2,PUnit.unit)) := by decide
  simp only [scatter, indices, List.foldl_map, offsets, uniqueTile]
  exact fold lanes s

theorem loaded_cells (count cols shift : Nat) (s : State α) :
    (fun l : Lane ↦ Cell.mk .int (loaded count s (offsets cols shift) (l.1,l.2,PUnit.unit))) =
      inputTile (.mk .int 0) count cols (inputCells s) shift := by
  funext l
  by_cases h : address cols shift l<count
  · simp [loaded, offsets, inputTile, inputCells, h,
      Nat.min_eq_left (show address cols shift l ≤ count-1 from by omega)]
  · simp [loaded, offsets, inputTile, h]

theorem final_offsets (count cols : Nat) (s : State α) :
    (finalState count cols s).regs .nat [2,2] "a_offsets" = some (offsets cols 28) ∧
    (finalState count cols s).regs .nat [2,2] "c_offsets" = some (offsets cols 48) := by
  simp [finalState, outerLoopState, outerState, middleLoopState, middleState,
    middlePrefixState, innerLoopState, innerState, prefixState,
    advances, bump_offsets, initialA_offsets, initialC_offsets]

/-- Exact complete output-cell state of the typed source. This bridge is
proved by the successful source execution, cache stability and all 24 stores. -/
theorem final_output (count cols : Nat) (s : State α) :
    (finalState count cols s).mem "output" =
      logicalRun (.mk .int 0) count cols (inputCells s) (s.mem "output") := by
  simp only [finalState, outerLoopState, outerState, middleLoopState, middleState,
    middlePrefixState, innerLoopState, innerState, prefixState,
    loaded_setReg, loaded_scatter, State.setReg_mem, initialA_offsets, initialC_offsets,
    advances, bump_offsets, Nat.reduceAdd, scatter_output, loaded_cells,
    inputCells_setReg, inputCells_scatter]
  simp [logicalRun, source_store_schedule, List.range_succ, List.foldl_append,
    CopyMath.inputShift]

/-- Cells outside the declared output allocation keep both value and dtype. -/
theorem scatter_frame (count : Nat) (u : Unique) (s : State α)
    (a : OffsetTile) (v : Payload) (r : RegionName) (i : Nat)
    (h : r ≠ "output" ∨ count ≤ i) :
    (scatter count u s a v).mem r i = s.mem r i := by
  have fold : ∀ (ls : List (TileIndex [2,2])) (s : State α),
      (ls.foldl (fun t j ↦ if decide (a j<count) && u j then
        t.write "output" (min (a j) (count-1)) (.mk .int (v j)) else t) s).mem r i = s.mem r i := by
    intro ls
    induction ls with
    | nil => intro s; rfl
    | cons j ls ih =>
      intro s
      rw [List.foldl_cons, ih]
      split
      next active =>
        simp only [Bool.and_eq_true, decide_eq_true_eq] at active
        apply State.write_other
        rcases h with hr | hi
        · exact Or.inl hr
        · right
          rw [Nat.min_eq_left (show a j ≤ count-1 from by omega)]
          omega
      next => rfl
  exact fold _ s

theorem final_frame (count cols : Nat) (s : State α) (r : RegionName) (i : Nat)
    (h : r ≠ "output" ∨ count ≤ i) :
    (finalState count cols s).mem r i = s.mem r i := by
  simp only [finalState, outerLoopState, outerState, middleLoopState, middleState,
    middlePrefixState, innerLoopState, innerState, prefixState, State.setReg_mem,
    scatter_frame count _ _ _ _ r i h]

end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
