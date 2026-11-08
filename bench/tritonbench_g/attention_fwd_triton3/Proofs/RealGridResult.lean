import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealScatterResult
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealReadFrame

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def modelLaneValue (c : Call) (v : Inputs c) (idx : GridIndex c.grid) (lane : Lane c.BM c.D) : Real.R :=
  Real.laneValue c (Real.finalCarry c (localInputs c v idx) (c.tile idx)) lane

theorem model_program (c : Call) (valid : Valid c) (v : Inputs c) (s : State Real.R)
    (idx : GridIndex c.grid) (pins : PinnedAt c v idx s) (lane : Lane c.BM c.D) (live : active c lane) :
    (Reference.scatter c Real.ops (Reference.inputs (Reference.Composition.prepare c.grid idx s))
      (Reference.result c Real.ops (Reference.inputs (Reference.Composition.prepare c.grid idx s)))
      (Reference.Composition.prepare c.grid idx s)).mem (address c idx lane).1 (address c idx lane).2=
      .mk .real (modelLaneValue c v idx lane) := by
  have h:=Real.result_closed c valid.blockN _ (localInputs c v idx) (finite_pins_at c v s idx pins)
  rw [(Real.prepared_coordinate c s idx).1] at h
  exact Real.scatter_result c valid s idx _ _ h lane live

theorem model_schedule_outputs (c : Call) (valid : Valid c) (v : Inputs c)
    (jobs : List (GridIndex c.grid)) (distinct : jobs.Nodup) (s : State Real.R)
    (pins : ∀ idx∈jobs,PinnedAt c v idx s) :
    ∀ idx∈jobs,∀ lane,active c lane →
      (Reference.referenceSchedule c Real.ops jobs s).mem (address c idx lane).1 (address c idx lane).2=
        .mk .real (modelLaneValue c v idx lane) := by
  induction jobs generalizing s with
  | nil=>simp
  | cons idx rest ih=>
    have nodup:=List.nodup_cons.mp distinct
    let entry:=Reference.Composition.prepare c.grid idx s
    let a:=Reference.inputs entry
    let mid:=Reference.scatter c Real.ops a (Reference.result c Real.ops a) entry
    have remaining : ∀ jdx∈rest,PinnedAt c v jdx mid := by
      intro jdx member
      exact scatter_pinned_other c valid v s idx jdx (by intro eq; subst jdx; exact nodup.1 member)
        (pins jdx (List.mem_cons_of_mem idx member)) _
    have results:=ih nodup.2 mid remaining
    intro jdx member lane live
    rcases List.mem_cons.mp member with same|tail
    · subst jdx
      change (Reference.referenceSchedule c Real.ops rest mid).mem _ _=_
      rw [ArrayLocality.schedule_frame]
      · exact model_program c valid v s idx (pins idx List.mem_cons_self) lane live
      · intro k member x active eq
        have own:=(unique c valid k idx x lane eq).1
        subst k
        exact nodup.1 member
    · exact results jdx tail lane live

#axiomsClean model_schedule_outputs
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
