import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.PublicIO
import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Proofs.IOInterpretation

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.FPEquiv
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

/-- Same source, shapes, indirect cache windows and complete grid. FP32 words
are unrestricted. Each Q/K/V output observes its own source-selected profile. -/
def rotaryNativeIO (rest : Algebra NativeSource.Payload) (p : Native.Instructions)
    (c : Config) : ProgramIO Native.Word NativeSource.Payload where
  inputs := PublicIO.inputChannels c.shape
  outputs := PublicIO.outputChannels c.shape
  scratch := []
  buffers := ["q","k","v","cos","sin","context_lengths","BLOCK_TABLES","k_cache","v_cache"]
  read := PublicIO.inputRead c.shape
  inputMask := fun _ _ _ => True
  write := PublicIO.outputWrite c
  outputMask := fun _ _ _ => True
  scratchWrite := fun k=>k.elim0
  scratchMask := fun k=>k.elim0
  schedule := fun _ => (programs c.shape).map (Source.invocation rest (NativeSource.ops p) c)
  encode := fun word _=>Native.atom word
  observe := PublicIO.observeOutput c.shape (fun kind value=>(value kind).value)

/-- Source-ordered native formula, retaining Q/K's distinct multiply/FMA
operand order and V's raw word copy. No native-to-Real identity is asserted. -/
def rotaryWordFormula (c : Config) (p : Native.Instructions)
    (xs : ChannelValues Native.Word (PublicIO.inputChannels c.shape)) :=
  PublicIO.outputFormula c.shape (wordFormula c.shape p (PublicIO.inputMemory c.shape xs))

/-- Existing primitive instruction semantics are the local numerical model.
This is source-to-native observation, with no floating transformation law. -/
theorem fused_rotary_native (rest : Algebra NativeSource.Payload) (c : Config)
    (valid : c.Valid) (p : Native.Instructions) :
    (rotaryNativeIO rest p c).ImplementsOn (PublicIO.metadataDomain c) (rotaryWordFormula c p) := by
  exact PublicIO.implements rest (NativeSource.ops p) (fun word _=>Native.atom word)
    (fun kind value=>(value kind).value) c valid
    (fun xs=>wordFormula c.shape p (PublicIO.inputMemory c.shape xs))
    (fun xs w hw high=>PublicIO.native_pair c p xs w hw high)

#axiomsClean fused_rotary_native
#specNonCircular rotaryWordFormula avoiding [Source.kernel,Source.grid,actualSource,Native.selected]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.FPEquiv
