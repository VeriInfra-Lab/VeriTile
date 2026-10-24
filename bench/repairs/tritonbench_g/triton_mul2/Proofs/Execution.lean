import bench.repairs.tritonbench_g.triton_mul2.Proofs.Model
import bench.tritonbench_g.vector_addition.Proofs.HostGrid

namespace VeriTile.Bench.Repairs.TritonMul2
noncomputable section
set_option maxHeartbeats 4000000

theorem stores_frame (writes : List (Nat × α)) (s : Memory α) (cell : Nat)
    (miss : ∀ w∈writes,w.1≠cell) : stores writes s cell=s cell := by
  induction writes generalizing s with
  | nil => rfl
  | cons w rest ih =>
    change stores rest (Function.update s w.1 w.2) cell=s cell
    rw [ih _ (fun v hv=>miss v (by simp [hv]))]
    exact Function.update_of_ne (miss w (by simp)).symm _ _

theorem stores_value (writes : List (Nat × α)) (s : Memory α) (cell : Nat) (value : α)
    (same : ∀ w∈writes,w.1=cell → w.2=value)
    (present : (∃ w∈writes,w.1=cell) ∨ s cell=value) : stores writes s cell=value := by
  induction writes generalizing s with
  | nil => simpa [stores] using present
  | cons w rest ih =>
    change stores rest (Function.update s w.1 w.2) cell=value
    apply ih _ (fun v hv=>same v (by simp [hv]))
    by_cases h : w.1=cell
    · right
      rw [h,Function.update_self]
      exact same w (by simp) h
    · rcases present with ⟨v,hv,hc⟩ | he
      · left
        exact ⟨v,(List.mem_cons.mp hv).resolve_left (fun eq=>h (eq ▸ hc)),hc⟩
      · right
        rw [Function.update_of_ne (Ne.symm h)]
        exact he

theorem program_value (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α)
    (i : Nat) (hi : i < l.count) (lower : pid*l.block ≤ i) (upper : i < (pid+1)*l.block) :
    program ops l pid s (l.output+i)=ops.twice (s (l.input+i)) := by
  unfold program
  apply stores_value
  · intro w hw he
    obtain ⟨lane,hlane,hw⟩ := List.mem_filterMap.mp hw
    by_cases active : pid*l.block+lane < l.count
    · simp only [active,↓reduceIte,Option.some.injEq] at hw
      subst w
      have index : pid*l.block+lane=i := by omega
      simp only [index]
    · simp [active] at hw
  · left
    refine ⟨(l.output+i,ops.twice (s (l.input+i))),?_,rfl⟩
    apply List.mem_filterMap.mpr
    refine ⟨i-pid*l.block,?_,?_⟩
    · simp only [List.mem_range]
      have h : i < pid*l.block+l.block := by simpa only [Nat.add_mul,Nat.one_mul] using upper
      omega
    · have index : pid*l.block+(i-pid*l.block)=i := by omega
      simp [index,hi]

theorem program_frame (ops : Operations α) (l : Layout) (pid : Nat) (s : Memory α)
    (cell : Nat) (miss : ¬∃ i,i < l.count ∧ pid*l.block ≤ i ∧ i < (pid+1)*l.block ∧ cell=l.output+i) :
    program ops l pid s cell=s cell := by
  unfold program
  apply stores_frame
  intro w hw he
  obtain ⟨lane,hlane,hw⟩ := List.mem_filterMap.mp hw
  by_cases active : pid*l.block+lane < l.count
  · simp only [active,↓reduceIte,Option.some.injEq] at hw
    subst w
    apply miss
    refine ⟨pid*l.block+lane,active,by omega,?_,he.symm⟩
    have hl := List.mem_range.mp hlane
    simpa only [Nat.add_mul,Nat.one_mul] using Nat.add_lt_add_left hl (pid*l.block)
  · simp [active] at hw

theorem prefix_correct (ops : Operations α) (l : Layout) (allocated : Allocated l)
    (initial : Memory α) (count : Nat) :
    (∀ i,i < l.count → i < count*l.block → runPrefix ops l initial count (l.output+i)=
      ops.twice (initial (l.input+i))) ∧
    (∀ cell,(¬∃ i,i < l.count ∧ i < count*l.block ∧ cell=l.output+i) →
      runPrefix ops l initial count cell=initial cell) := by
  induction count with
  | zero => exact ⟨by simp,fun _ _=>rfl⟩
  | succ count ih =>
    constructor
    · intro i hi covered
      by_cases old : i < count*l.block
      · change program ops l count (runPrefix ops l initial count) (l.output+i)=_
        rw [program_frame ops l count _ _ ?_,ih.1 i hi old]
        rintro ⟨j,hj,lo,up,eq⟩
        omega
      · change program ops l count (runPrefix ops l initial count) (l.output+i)=_
        rw [program_value ops l count _ i hi (by omega) covered,ih.2 _ ?_]
        rintro ⟨j,hj,earlier,eq⟩
        rcases allocated.separation with same | separate
        · rw [same] at eq
          omega
        · exact separate j i hj hi eq.symm
    · intro cell miss
      change program ops l count (runPrefix ops l initial count) cell=initial cell
      rw [program_frame ops l count _ cell ?_,ih.2 cell ?_]
      · rintro ⟨i,hi,old,eq⟩
        apply miss
        exact ⟨i,hi,lt_of_lt_of_le old (Nat.mul_le_mul_right l.block (by omega)),eq⟩
      · rintro ⟨i,hi,lo,up,eq⟩
        exact miss ⟨i,hi,up,eq⟩

theorem launch_correct (ops : Operations α) (l : Layout) (allocated : Allocated l)
    (positive : 0 < l.block) (initial : Memory α) :
    Doubled ops.twice l.count (fun i=>initial (l.input+i))
      (fun i=>launch ops l initial (l.output+i)) ∧
    (∀ cell,(¬∃ i,i < l.count ∧ cell=l.output+i) → launch ops l initial cell=initial cell) := by
  have covers := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.grid_covers
    (⟨"input","input","output",l.count,l.block⟩ :
      VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.Parameters) positive
  change l.count ≤ grid l.count l.block*l.block at covers
  have result := prefix_correct ops l allocated initial (grid l.count l.block)
  exact ⟨fun i hi=>result.1 i hi (hi.trans_le covers),fun cell miss=>
    result.2 cell (fun ⟨i,hi,_,he⟩=>miss ⟨i,hi,he⟩)⟩

#axiomsClean program_value
#axiomsClean prefix_correct
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.TritonMul2
