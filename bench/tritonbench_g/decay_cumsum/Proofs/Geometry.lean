import bench.tritonbench_g.decay_cumsum.Proofs.Source
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.DecayCumsum.Geometry
set_option maxHeartbeats 3000000

theorem width_covers (h : Call) : h.DK ≤ h.width := by
  have hmod := Nat.mod_lt (h.DK+h.BK-1) h.featureTilePositive
  have hdiv := Nat.mod_add_div (h.DK+h.BK-1) h.BK
  have tile := h.featureTilePositive
  have sub : (h.DK+h.BK-1)+1=h.DK+h.BK := Nat.sub_add_cancel (by omega)
  dsimp only [Call.width,Call.features]
  nlinarith

theorem width_positive (h : Call) (positive : 0<h.DK) : 0<h.width :=
  positive.trans_le (width_covers h)

theorem tile_le_width (h : Call) (positive : 0<h.DK) : h.BK≤h.width := by
  have pos := width_positive h positive
  have features : 0<h.features := by
    by_contra hn
    have : h.features=0 := by omega
    simp [Call.width,this] at pos
  exact Nat.le_mul_of_pos_left _ features

theorem width_exact (h : Call) : h.width/h.BK=h.features := by
  simp [Call.width,Nat.ne_of_gt h.featureTilePositive]

theorem phase_positive (h : Call) (p : Phase) (hp : p∈phases h) :
    0<p.count ∧ 0<p.span ∧ p.start+p.count*p.span≤h.T := by
  have tile := h.timeTilePositive
  have rem := Nat.mod_lt h.T tile
  have div := Nat.mod_add_div h.T h.BT
  simp only [phases,List.mem_append] at hp
  rcases hp with hp|hp
  · by_cases hc : h.T/h.BT=0
    · simp [hc] at hp
    · simp only [if_neg hc,List.mem_singleton] at hp
      subst p
      exact ⟨Nat.pos_of_ne_zero hc,tile,by simpa [mainPhase] using Nat.div_mul_le_self h.T h.BT⟩
  · by_cases hc : h.T%h.BT=0
    · simp [hc] at hp
    · simp only [if_neg hc,List.mem_singleton] at hp
      subst p
      dsimp [tailPhase]
      exact ⟨by omega,Nat.pos_of_ne_zero hc,by nlinarith⟩

structure Program (h : Call) (p : Phase) where
  feature : Nat
  chunk : Nat
  row : Nat
  featureInside : feature<h.features
  chunkInside : chunk<p.count
  rowInside : row<h.rows

def time (p : Phase) (chunk step : Nat) : Nat := p.start+chunk*p.span+step
def column (h : Call) (feature lane : Nat) : Nat := feature*h.BK+lane
def offset (h : Call) (p : Phase) (prog : Program h p) (step lane : Nat) : Nat :=
  prog.row*h.headStride+time p prog.chunk step*h.width+column h prog.feature lane

theorem time_inside (h : Call) (p : Phase) (hp : p∈phases h)
    (prog : Program h p) (step : Fin p.span) : time p prog.chunk step<h.T := by
  have bounds := phase_positive h p hp
  have hc := Nat.mul_le_mul_right p.span (Nat.succ_le_of_lt prog.chunkInside)
  have hs := step.isLt
  dsimp [time]
  nlinarith

theorem column_inside (h : Call) (p : Phase) (prog : Program h p)
    (lane : Fin h.BK) : column h prog.feature lane<h.width := by
  have hf := Nat.mul_le_mul_right h.BK (Nat.succ_le_of_lt prog.featureInside)
  have hl := lane.isLt
  dsimp [column,Call.width]
  nlinarith

theorem offset_inside (h : Call) (p : Phase) (hp : p∈phases h)
    (prog : Program h p) (step : Fin p.span) (lane : Fin h.BK) :
    offset h p prog step.val lane.val<h.cells := by
  have ht := time_inside h p hp prog step
  have hk := column_inside h p prog lane
  have rows := Nat.mul_le_mul_right h.headStride (Nat.succ_le_of_lt prog.rowInside)
  have times := Nat.mul_le_mul_right h.width (Nat.succ_le_of_lt ht)
  dsimp [offset,Call.cells,Call.headStride] at *
  nlinarith

#axiomsClean offset_inside
end VeriTile.Bench.Repairs.DecayCumsum.Geometry
