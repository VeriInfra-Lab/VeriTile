import bench.tritonbench_g.decay_cumsum.Proofs.Coverage
import Mathlib.Data.List.Nodup

namespace VeriTile.Bench.Repairs.DecayCumsum.Grid
open Geometry Coverage
set_option maxHeartbeats 4000000

abbrev Work (h : Call) := (p : Phase) × Program h p

def schedule (h : Call) : List (Work h) := (phases h).sigma (programs h)

theorem programs_nodup (h : Call) (p : Phase) : (programs h p).Nodup := by
  apply List.nodup_flatMap.mpr
  constructor
  · intro feature _
    apply List.nodup_flatMap.mpr
    constructor
    · intro chunk _
      apply (List.nodup_finRange h.rows).map
      intro a b he
      apply Fin.ext
      exact congrArg (fun (prog : Program h p) => prog.row) he
    · apply (List.nodup_finRange p.count).imp
      intro a b different prog memberA memberB
      obtain ⟨ra,_,rfl⟩ := List.mem_map.mp memberA
      obtain ⟨rb,_,he⟩ := List.mem_map.mp memberB
      apply different
      exact Fin.ext (congrArg (fun (prog : Program h p) => prog.chunk) he.symm)
  · apply (List.nodup_finRange h.features).imp
    intro a b different prog memberA memberB
    obtain ⟨ca,_,memberA⟩ := List.mem_flatMap.mp memberA
    obtain ⟨ra,_,rfl⟩ := List.mem_map.mp memberA
    obtain ⟨cb,_,memberB⟩ := List.mem_flatMap.mp memberB
    obtain ⟨rb,_,he⟩ := List.mem_map.mp memberB
    apply different
    exact Fin.ext (congrArg (fun (prog : Program h p) => prog.feature) he.symm)

theorem phases_nodup (h : Call) : (phases h).Nodup := by
  by_cases main : h.T/h.BT=0 <;> by_cases tail : h.T%h.BT=0
  · simp [phases,main,tail]
  · simp [phases,main,tail]
  · simp [phases,main,tail]
  · have different : mainPhase h≠tailPhase h := by
      intro he
      have starts := congrArg Phase.start he
      have positive : 0<h.T/h.BT*h.BT := Nat.mul_pos (Nat.pos_of_ne_zero main) h.timeTilePositive
      dsimp [mainPhase,tailPhase] at starts
      omega
    simp [phases,main,tail,different]

theorem schedule_nodup (h : Call) : (schedule h).Nodup :=
  (phases_nodup h).sigma (programs_nodup h)

theorem schedule_member (h : Call) (p : Phase) (hp : p∈phases h) (prog : Program h p) :
    (⟨p,prog⟩ : Work h)∈schedule h := by
  apply List.mem_sigma.mpr
  exact ⟨hp,all_programs h p prog⟩

theorem schedule_valid (h : Call) (work : Work h) (member : work∈schedule h) : work.1∈phases h :=
  (List.mem_sigma.mp member).1

theorem work_unique (h : Call) (a b : Work h) (ha : a∈schedule h) (hb : b∈schedule h)
    (t : Fin a.1.span) (t' : Fin b.1.span) (j j' : Fin h.BK)
    (same : offset h a.1 a.2 t.val j.val=offset h b.1 b.2 t'.val j'.val) : a=b := by
  rcases a with ⟨p,a⟩
  rcases b with ⟨p',b⟩
  obtain ⟨phase,rows,features,chunks,_,_⟩ := offset_unique h p p' (schedule_valid h _ ha)
    (schedule_valid h _ hb) a b t t' j j' same
  subst p'
  have programs : a=b := by cases a; cases b; simp_all
  subst b
  rfl

#axiomsClean schedule_nodup
#axiomsClean work_unique
end VeriTile.Bench.Repairs.DecayCumsum.Grid
