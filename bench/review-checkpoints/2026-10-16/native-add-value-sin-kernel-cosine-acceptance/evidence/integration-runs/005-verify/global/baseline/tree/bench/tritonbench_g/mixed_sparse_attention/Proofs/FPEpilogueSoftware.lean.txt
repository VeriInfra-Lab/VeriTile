import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPEpilogue
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Concrete fp32 scratch loads/division and an explicit final fp16 conversion.
The narrow conversion is a primitive binding, never silently identity. -/
def softwareEpilogue (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (toHalf : FP.Value .fp32 → FP.Value .fp32) : Algebra (FP.Value .fp32) :=
  { FP.Software.scalarFP32 rest cfg with
    cast := fun p src dst x => if p=none ∧ src=.real ∧ dst=.fp16 then toHalf x
      else (FP.Software.scalarFP32 rest cfg).cast p src dst x }

def roundedEpilogueMemory [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (toHalf : FP.Value .fp32 → FP.Value .fp32) (AccPre LPre : RegionName)
    (c : Config) (s : State (FP.Value .fp32)) : State (FP.Value .fp32) :=
  (TileShape.allIndices [c.BM,c.BD]).foldl (fun t i =>
    if rowStart c s+i.1.val < sequenceLength c s then
      t.write c.out (queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok)
        (.mk .fp16 (toHalf (FP.div cfg
          ((s.mem AccPre (queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok)).read .real)
          ((s.mem LPre (rowStart c s+i.1.val)).read .real))))
    else t) s

specification epilogue_software_with_cast [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (toHalf : FP.Value .fp32 → FP.Value .fp32) (AccPre LPre : RegionName)
    (c : Config) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest cfg toHalf) (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (roundedEpilogueMemory cfg toHalf AccPre LPre c s).mem := by
  obtain ⟨final,he,hm⟩ := epilogue_fp16 (softwareEpilogue rest cfg toHalf) AccPre LPre c s
  refine ⟨final,he,?_⟩
  simpa [outputMemory,epilogueInputs,roundedEpilogueMemory,softwareEpilogue,
    FP.Software.scalarFP32,FP.Software.fp32Context] using hm

/-- The source's scalar division is genuinely rounded before narrow storage. -/
theorem third_rounding :
    FP.div .ieeeValue (FP.Value.ofBits .fp32 0x3f800000) (FP.Value.ofBits .fp32 0x40400000) =
      FP.Value.ofBits .fp32 0x3eaaaaab ∧
      (11184811 / 33554432 : Rat) ≠ 1/3 := by
  constructor
  · decide +kernel
  · norm_num

#axiomsClean epilogue_software_with_cast
#axiomsClean third_rounding
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
