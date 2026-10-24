import bench.repairs.tritonbench_g.rope_backward_transform.IO
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem launch_member (r : Request) (row : Nat) (h : row∈KernelContract.launchRows r) :
    row<r.rows ∧ KernelContract.launched r := by
  unfold KernelContract.launchRows at h
  split at h
  · exact ⟨List.mem_range.mp h, by assumption⟩
  · simp at h

theorem sequence_positive (r : Request) {row : Nat} (hr : row<r.rows) :
    0<r.sequence := by
  by_contra h
  have hz : r.sequence=0 := by omega
  simp [Request.rows,hz] at hr

theorem bounds_q {A : Type} (op : Operations A) (project : A → A) (r : Request)
    (xs : Inputs A r) (bounds : RegionName → Nat)
    (hb : (sourceIO op project r).InBounds bounds xs)
    (key : Bool) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads key) (hc : col<r.dimension) :
    r.offset key row head col < bounds (region key) := by
  cases key
  · exact hb.1.2.1 ⟨0,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
  · exact hb.1.2.1 ⟨1,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial

theorem bounds_coefficient {A : Type} (op : Operations A) (project : A → A) (r : Request)
    (xs : Inputs A r) (bounds : RegionName → Nat)
    (hb : (sourceIO op project r).InBounds bounds xs) (on : KernelContract.launched r)
    (sine : Bool) (row lane : Nat) (hr : row<r.sequence) (hl : lane<r.half) :
    row*r.rowStride sine+lane*r.colStride sine < bounds (if sine then "S" else "C") := by
  cases sine
  · exact hb.1.2.1 ⟨2,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit) on
  · exact hb.1.2.1 ⟨3,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨lane,hl⟩,PUnit.unit) on

theorem invocation_safe {A : Type} [Inhabited A] (op : Operations A)
    (project : A → A) (r : Request) (xs : Inputs A r) (bounds : RegionName → Nat)
    (hb : (sourceIO op project r).InBounds bounds xs) (row : Nat)
    (hr : row∈KernelContract.launchRows r) (s : State A) :
    (invocation op r row).Safe bounds s := by
  obtain ⟨rowBound,on⟩ := launch_member r row hr
  apply SourceSafety.source_safe
  · intro sine j hj
    exact bounds_coefficient op project r xs bounds hb on sine _ _
      (Nat.mod_lt _ (sequence_positive r rowBound)) hj
  · intro key side i hi
    obtain ⟨hh,hj⟩ := (active_iff _ _ i).mp hi
    simpa [address,address_coordinate,invocation,Invocation.initial,bindRegisters] using
      bounds_q op project r xs bounds hb key row i.1.val
        (i.2.1.val+if side then r.half else 0) rowBound hh (half_column r side _ hj)

theorem schedule_safe {A : Type} [Inhabited A] (op : Operations A)
    (project : A → A) (r : Request) (xs : Inputs A r) (bounds : RegionName → Nat)
    (hb : (sourceIO op project r).InBounds bounds xs) (s : State A) :
    ScheduleSafe bounds ((sourceIO op project r).schedule xs) s := by
  suffices ∀ rows : List Nat, (∀ row∈rows,row∈KernelContract.launchRows r) →
      ∀ t, ScheduleSafe bounds (rows.map (invocation op r)) t from
    this _ (fun _ h => List.mem_reverse.mp h) s
  intro rows
  induction rows with
  | nil => intro _ _; trivial
  | cons row rows ih =>
    intro member t
    exact ⟨invocation_safe op project r xs bounds hb row (member row (by simp)) t,
      fun next _ => ih (fun a h => member a (by simp [h])) next⟩

/-- Every invocation executes the same authoritative source as the old row
certificate. Local registers, undefined values and unused PID axes cannot
change its memory result, since all source operands are assigned before use. -/
theorem invocation_run {A : Type} [Inhabited A] (op : Operations A)
    (r : Request) (row : Nat) (s : State A) :
    ∃ final, (invocation op r row).run s=some final ∧
      final.mem=rowMemory op.backend r row s.mem := by
  refine ⟨(Source.execution op.backend (parameters r)
    ((invocation op r row).initial s)).val, ?_, ?_⟩
  · exact (Source.execution op.backend (parameters r)
      ((invocation op r row).initial s)).property
  · rw [Source.execution_memory]
    rfl

theorem schedule_run {A : Type} [Inhabited A] (op : Operations A) (r : Request)
    (rows : List Nat) (s : State A) :
    ∃ final, runSchedule (rows.reverse.map (invocation op r)) s=some final ∧
      final.mem=grid op.backend r s.mem rows := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons row rows ih =>
    obtain ⟨previous,ran,mem⟩ := ih s
    obtain ⟨final,last,out⟩ := invocation_run op r row previous
    refine ⟨final,?_,?_⟩
    · rw [List.reverse_cons,List.map_append,runSchedule_append,ran]
      simpa only [List.map_singleton,Option.bind_some,runSchedule_singleton] using last
    · simpa only [grid,mem] using out

theorem gridExec_memory {A : Type} [Inhabited A] (M : Algebra A) (r : Request)
    {rows : List Nat} {initial final : Source.Memory A}
    (h : GridExec M r rows initial final) : final=grid M r initial rows := by
  induction h with
  | nil => rfl
  | @cons xs m current after row prior last ih =>
    obtain ⟨t,ran,mem⟩ := last
    obtain ⟨u,ran',mem'⟩ := row_execution M r row current
    have same : t=u := Option.some.inj (ran.symm.trans ran')
    have h := mem.symm.trans ((congrArg State.mem same).trans mem')
    simpa only [grid,ih] using h

#axiomsClean invocation_safe
#axiomsClean schedule_safe
#axiomsClean schedule_run
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.PublicIO
