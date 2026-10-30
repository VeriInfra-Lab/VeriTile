import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.IOResults
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Native

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

theorem real_pair (c : Config) (hc : c.Valid)
    (xs : ChannelValues ℝ (inputChannels c.shape)) (w : Work) (hw : c.shape.Active w) (high : Bool) :
    component (pairValues c.shape (fun _=>realOps) w (inputMemory c.shape xs)) high =
      formula c.shape (inputMemory c.shape xs) w.kind w.token w.head (c.shape.dim w high) :=
  pair_real_formula c.shape hc.shape w hw _ high

theorem native_pair (c : Config) (p : Native.Instructions)
    (xs : ChannelValues Native.Word (inputChannels c.shape)) (w : Work)
    (hw : c.shape.Active w) (high : Bool) :
    (component (pairValues c.shape (fun _=>NativeSource.ops p) w
        (fun r a _=>Native.atom (inputMemory c.shape xs r a))) high w.kind).value =
      wordFormula c.shape p (inputMemory c.shape xs) w.kind w.token w.head (c.shape.dim w high) := by
  exact (NativeSource.pair_projection p c.shape w (inputMemory c.shape xs) high).trans
    (pairWord_formula c.shape w hw p (inputMemory c.shape xs) high)

#axiomsClean real_pair
#axiomsClean native_pair
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
