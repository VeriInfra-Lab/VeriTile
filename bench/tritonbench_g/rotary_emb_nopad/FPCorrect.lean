import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPWitness
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Triton FP.Structural
noncomputable section

specification rotary_nopad_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPBlocks.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPBlocks.active Bool.false p s) (FPBlocks.offset Bool.false p s)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPBlocks.source Bool.false p) s = some final ∧
      FPBlocks.Post Bool.false p s final (FPBlocks.roundedValue cfg Bool.false p s) :=
  FPBlocks.software_correctness rest cfg Bool.false p s hw

specification rotary_nopad_k_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPBlocks.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPBlocks.active Bool.true p s) (FPBlocks.offset Bool.true p s)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPBlocks.source Bool.true p) s = some final ∧
      FPBlocks.Post Bool.true p s final (FPBlocks.roundedValue cfg Bool.true p s) :=
  FPBlocks.software_correctness rest cfg Bool.true p s hw

specification rotary_nopad_v2_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPV2.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.false p s)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest cfg) (FPV2.source Bool.false p) s = some final ∧
      FPV2.Post Bool.false p s final (FPV2.roundedValue cfg p s) :=
  FPV2.software_correctness rest cfg Bool.false p s hw

specification rotary_nopad_v2_cache_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPV2.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.true p s)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest cfg) (FPV2.source Bool.true p) s = some final ∧
      FPV2.Post Bool.true p s final (FPV2.roundedValue cfg p s) :=
  FPV2.software_correctness rest cfg Bool.true p s hw

theorem rotary_nopad_block_source_projection (key : Bool) (p : FPBlocks.Parameters) :
    (FPBlocks.source key p).eraseDType.surfaceBody = (FPBlocks.legacySource key p).eraseDType.surfaceBody :=
  FPBlocks.source_projection key p

theorem rotary_nopad_v2_source_projection (cache : Bool) (p : FPV2.Parameters) :
    (FPV2.source cache p).eraseDType.surfaceBody = (FPV2.legacySource cache p).eraseDType.surfaceBody :=
  FPV2.source_projection cache p

theorem rotary_nopad_blocks_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (key : Bool) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPBlocks.source key FPExamples.blocks) FPExamples.initial = some final ∧
      final.mem "x" 0 = .mk .fp32 FPExamples.one ∧ final.mem "x" 1 = .mk .fp32 FPExamples.one ∧
      final.mem "x" 2 = FPExamples.initial.mem "x" 2 ∧ FPExamples.tiny ≠ FPExamples.one :=
  FPExamples.blocks_nonempty rest key

theorem rotary_nopad_v2_bit_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cache : Bool) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest .ieeeValue)
      (FPV2.source cache FPExamples.v2) FPExamples.initial = some final ∧
      final.mem (FPV2.outputRegion cache FPExamples.v2) (if cache then 304 else 0) = .mk .fp32 FPExamples.one ∧
      final.mem (FPV2.outputRegion cache FPExamples.v2) (if cache then 305 else 1) = .mk .fp32 FPExamples.one ∧
      final.mem (FPV2.outputRegion cache FPExamples.v2) (if cache then 306 else 2) =
        FPExamples.initial.mem (FPV2.outputRegion cache FPExamples.v2) (if cache then 306 else 2) ∧
      FPExamples.tiny ≠ FPExamples.one :=
  FPExamples.v2_nonempty rest cache

theorem rotary_nopad_v2_masked_default_success [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest .ieeeValue)
      (FPV2.source Bool.true FPExamples.v2) FPExamples.outOfTokenInitial = some final ∧
      final.mem "cache" 704 = .mk .fp32 (FP.Value.zero .fp32) ∧
      final.mem "cache" 705 = .mk .fp32 (FP.Value.zero .fp32) :=
  FPExamples.v2_masked_defaults rest

#axiomsClean rotary_nopad_q_fp32_correctness
#axiomsClean rotary_nopad_k_fp32_correctness
#axiomsClean rotary_nopad_v2_q_fp32_correctness
#axiomsClean rotary_nopad_v2_cache_fp32_correctness
#axiomsClean rotary_nopad_block_source_projection
#axiomsClean rotary_nopad_v2_source_projection
#axiomsClean rotary_nopad_blocks_bit_success
#axiomsClean rotary_nopad_v2_bit_success
#axiomsClean rotary_nopad_v2_masked_default_success
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad
