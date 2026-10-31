import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealHStep

/-! Every iteration of the complete H loop. The carry is the independent
closed form, history retains every earlier pre-update state, and unrelated
complete memory cells are preserved. Zero unspecified values are an explicit
premise for this zero-extended formula; the step theorem itself is unrestricted. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoop
open VeriTile.Triton RealHMath RealHProgram RealHTiles
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def address (p : Layout) (l : Launch) (c e j : Nat) : Nat :=
  RealHStore.address p l (RealHStep.historyBase p l c) p.historyKey p.historyValue e j

def HistoryNoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e f : Fin p.BK) (i j : Fin p.BV), c < p.NT → d < p.NT →
    RealHStore.active p l e.val i.val → RealHStore.active p l f.val j.val →
    address p l c e.val i.val = address p l d f.val j.val → c = d ∧ e = f ∧ i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = r.history ∧ ∃ (c : Nat) (e : Fin p.BK) (j : Fin p.BV), c < p.NT ∧
    RealHStore.active p l e.val j.val ∧ cell.2 = address p l c e.val j.val

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHLoop
