import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.IO
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.Native
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.IOPair
open VeriTile Triton FP.Structural FusedRotaryEmbedding
noncomputable section

def io (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config) :=
  PublicIO.io rest (NativeSource.ops p) (fun word _ => Native.atom word)
    (fun kind value => (value kind).value) c

def job (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config) (pid : Program) :
    Invocation NativeSource.Payload :=
  { Source.invocation rest (NativeSource.ops p) c pid with
    kernel := Source.kernelSharedQOffset (Source.parameters c.shape) }

def variant (rest : Algebra NativeSource.Payload) (p : Native.Instructions) (c : Config) :
    ProgramIO.Variant (io rest p c) where
  schedule := fun _ => (programs c.shape).map (job rest p c)
  observe := (io rest p c).observe

end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.IOPair
