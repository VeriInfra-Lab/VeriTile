import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPWhole
import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPEpilogueSoftware
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- A nonempty 16x16x16 source profile. Success is interpreter success; native
compilation, tensor layouts and primitive conformance remain separate. -/
def witnessConfig : Config where
  q := "q"
  k := "k"
  v := "v"
  out := "out"
  seq := ⟨"seq"⟩
  blocks := ⟨"blocks"⟩
  offsets := ⟨"offsets"⟩
  counts := ⟨"counts"⟩
  columns := ⟨"columns"⟩
  sqz := 256
  sqh := 256
  sqm := 16
  sqk := 1
  skz := 256
  skh := 256
  skn := 16
  skk := 1
  svz := 256
  svh := 256
  svn := 16
  svk := 1
  soz := 256
  soh := 256
  som := 16
  sok := 1
  Z := 1
  H := 1
  T := 16
  NR := 1
  NS := 8
  NV := 16
  BM := 16
  BN := 16
  BD := 16
  scale := 1

def witnessInitial {α : Type} [Inhabited α] : State α where
  mem := fun r _ =>
    if r="seq" then .mk .nat 16
    else if r="blocks" ∨ r="counts" then .mk .nat 1
    else if r="offsets" ∨ r="columns" then .mk .nat 0
    else .mk .fp16 default
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

specification forward_nonempty {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) :
    ∃ final, FP.Structural.exec (backend M D) (kernel witnessConfig) witnessInitial = some final ∧
      final.mem = (outputMemory M witnessConfig witnessInitial
        (finalValues M D witnessConfig witnessInitial)).mem ∧
      (∃ i : TileIndex [16,16], rowStart witnessConfig (witnessInitial : State α)+i.1.val <
        sequenceLength witnessConfig (witnessInitial : State α)) := by
  obtain ⟨final,he,hm⟩ := whole_fp16 M D witnessConfig witnessInitial (by decide)
    (by simp [rowStart,sequenceLength,witnessConfig,witnessInitial,Region.cast,Cell.read])
  refine ⟨final,he,hm,?_⟩
  exact ⟨⟨⟨0,by decide⟩,⟨0,by decide⟩,()⟩,by
    simp [rowStart,sequenceLength,witnessConfig,witnessInitial,Region.cast,Cell.read]⟩

def epilogueWitnessConfig : Config :=
  {witnessConfig with BM := 2, BD := 1, som := 1, sok := 1, sqz := 0, sqh := 0}

def epilogueWitnessInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r _ =>
    if r="seq" then .mk .nat 1
    else if r="acc" then .mk .real (FP.Value.ofBits .fp32 0x3f800000)
    else if r="den" then .mk .real (FP.Value.ofBits .fp32 0x40400000)
    else .mk .fp16 (FP.Value.ofBits .fp32 0x3f800000)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

specification epilogue_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (toHalf : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest .ieeeValue toHalf)
      (epilogueSource "acc" "den" epilogueWitnessConfig) epilogueWitnessInitial = some final ∧
      final.mem "out" 0 = .mk .fp16 (toHalf (FP.Value.ofBits .fp32 0x3eaaaaab)) ∧
      final.mem "out" 1 = epilogueWitnessInitial.mem "out" 1 ∧
      (11184811/33554432 : Rat) ≠ 1/3 := by
  obtain ⟨final,he,hm⟩ := epilogue_software_with_cast rest .ieeeValue toHalf "acc" "den"
    epilogueWitnessConfig epilogueWitnessInitial
  refine ⟨final,he,?_,?_,third_rounding.2⟩
  all_goals rw [hm]
  all_goals simp [roundedEpilogueMemory,epilogueWitnessConfig,witnessConfig,epilogueWitnessInitial,
    rowStart,queryBase,sequenceLength,Region.cast,Cell.read,TileShape.allIndices,List.finRange_succ,
    State.write,third_rounding.1]

#axiomsClean forward_nonempty
#axiomsClean epilogue_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
