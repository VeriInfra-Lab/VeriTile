import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SignedBodies
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SignedIndexed

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def columns (B i : Nat) : Tile .nat [B] := ⟨fun lane ↦ i+lane.1.val⟩
def offsets (N B i p : Nat) : Tile .nat [B] := ⟨fun lane ↦ p*N+(i+lane.1.val)⟩
def pointers (X : RegionName) (N B i p : Nat) : Tile .ptr [B] :=
  ⟨fun lane ↦ (X,p*N+(i+lane.1.val))⟩

def prefixIntState (X : RegionName) (N B i p : Nat) (s : BlockState) : BlockState :=
  (((s.setReg "start_n" .int [] (Tile.scalar (i:Int))).setReg "n_offsets" .int [B]
    (castIndex (columns B i))).setReg "offset" .int [B] (castIndex (offsets N B i p))).setReg
    "input_ptrs" .ptr [B] (pointers X N B i p)

def prefixNatState (X : RegionName) (N B i p : Nat) (s : BlockState) : BlockState :=
  (((s.setReg "start_n" .nat [] (Tile.scalar i)).setReg "n_offsets" .nat [B]
    (columns B i)).setReg "offset" .nat [B] (offsets N B i p)).setReg
    "input_ptrs" .ptr [B] (pointers X N B i p)

theorem prefix_states_related (X : RegionName) (N B i p : Nat) {s t : BlockState}
    (h : LiveRelated s t) : IndexedRelated B
      (prefixIntState X N B i p s) (prefixNatState X N B i p t) := by
  have h0 := h.setScratch "start_n" .int .nat [] []
    (Tile.scalar (i:Int)) (Tile.scalar i) (Or.inl rfl)
  have h1 := h0.setScratch "n_offsets" .int .nat [B] [B]
    (castIndex (columns B i)) (columns B i) (Or.inr (Or.inl rfl))
  have h2 := h1.setScratch "offset" .int .nat [B] [B]
    (castIndex (offsets N B i p)) (offsets N B i p) (Or.inr (Or.inr rfl))
  refine ⟨h2.setShared "input_ptrs" .ptr [B] (pointers X N B i p) (by simp [Shared, Scratch]), ?_, ?_⟩
  · simp only [prefixIntState, prefixNatState, BlockState.setReg_ne_name,
      BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, Option.map_some]
  · simp only [prefixIntState, prefixNatState, BlockState.setReg_ne_name,
      BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true, Option.map_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
