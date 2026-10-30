import bench.repairs.tritonbench_g.decay_cumsum.IO
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.IOSafetyLoops

namespace VeriTile.Bench.Repairs.DecayCumsum.DecayIO
open VeriTile.Triton FP.Structural Geometry
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem access_bounds (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) (vals : (io M kind h p prog scale).Inputs)
    (bounds : RegionName → Nat) (bounded : (io M kind h p prog scale).InBounds bounds vals)
    (r : RegionName) (member : r∈FPTile.inputs kind++FPTile.outputs kind)
    (t : Fin p.span) (j : Fin h.BK) : FPPrefix.off h p prog t.val j<bounds r := by
  cases kind
  case forward =>
    simp only [FPTile.inputs,FPTile.outputs,FPBackwardBase.inputRegions,FPBackwardBase.outputRegions,
      List.cons_append,List.nil_append,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl
    · exact bounded.2.1 (0 : Fin 1) (t,j,()) trivial
    · exact bounded.2.2.1 (0 : Fin 1) (t,j,()) trivial
  case prepare =>
    simp only [FPTile.inputs,FPTile.outputs,FPBackwardBase.inputRegions,FPBackwardBase.outputRegions,
      List.cons_append,List.nil_append,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl
    · exact bounded.2.1 (0 : Fin 3) (t,j,()) trivial
    · exact bounded.2.1 (1 : Fin 3) (t,j,()) trivial
    · exact bounded.2.1 (2 : Fin 3) (t,j,()) trivial
    · exact bounded.2.2.1 (0 : Fin 2) (t,j,()) trivial
    · exact bounded.2.2.1 (1 : Fin 2) (t,j,()) trivial
  case backward =>
    simp only [FPTile.inputs,FPTile.outputs,FPBackwardBase.inputRegions,FPBackwardBase.outputRegions,
      List.cons_append,List.nil_append,List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    · exact bounded.2.1 (0 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (1 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (2 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (3 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (4 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (5 : Fin 7) (t,j,()) trivial
    · exact bounded.2.1 (6 : Fin 7) (t,j,()) trivial
    · exact bounded.2.2.1 (0 : Fin 3) (t,j,()) trivial
    · exact bounded.2.2.1 (1 : Fin 3) (t,j,()) trivial
    · exact bounded.2.2.1 (2 : Fin 3) (t,j,()) trivial

theorem schedule_safe (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ)
    (vals : (io M kind h p prog scale).Inputs) (bounds : RegionName → Nat)
    (bounded : (io M kind h p prog scale).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds ((io M kind h p prog scale).schedule vals) s := by
  change ScheduleSafe bounds [launch M kind h p prog scale] s
  rw [scheduleSafe_singleton]
  exact IOSafety.source_safe M kind h p (phase_positive h p hp).2.1 prog scale
    ((launch M kind h p prog scale).initial s) rfl rfl rfl bounds
    (access_bounds M kind h p prog scale vals bounds bounded)

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.DecayCumsum.DecayIO
