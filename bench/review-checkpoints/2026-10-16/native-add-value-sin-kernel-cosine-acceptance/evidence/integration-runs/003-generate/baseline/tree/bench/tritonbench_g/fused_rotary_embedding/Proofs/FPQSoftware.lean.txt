import bench.tritonbench_g.fused_rotary_embedding.Proofs.FPQContract
namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (second : Bool) (i : TileIndex [p.half]) : FP.Value .fp32 :=
  let x0 := (s.mem p.q (offset p s Bool.false i)).read .fp32
  let x1 := (s.mem p.q (offset p s Bool.true i)).read .fp32
  let c := (s.mem p.cosine (cacheOffset p s i)).read .fp32
  let z := (s.mem p.sine (cacheOffset p s i)).read .fp32
  if second then FP.add cfg (FP.mul cfg x0 z) (FP.mul cfg x1 c)
  else FP.sub cfg (FP.mul cfg x0 c) (FP.mul cfg x1 z)

theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hstride : p.dimensionStride ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (source p) s = some final ∧
      Post p s final (roundedValue cfg p s) := by
  obtain ⟨final, he, hp⟩ := correctness (FP.Software.scalarFP32 rest cfg) p s hstride
  refine ⟨final, he, ?_, hp.2⟩
  intro second i
  simpa [value, roundedValue, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 second i

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessTiny : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_rounding :
    FP.sub .ieeeValue (FP.mul .ieeeValue witnessOne witnessOne)
      (FP.mul .ieeeValue witnessTiny witnessOne) = witnessOne ∧
    FP.add .ieeeValue (FP.mul .ieeeValue witnessOne witnessOne)
      (FP.mul .ieeeValue witnessTiny witnessOne) = witnessOne ∧
    witnessOne.toRat? = some 1 ∧ witnessTiny.toRat? = some (1 / 33554432) ∧
    (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 ∧
    witnessTiny ≠ witnessOne := by decide +kernel

def witnessParameters : Parameters := ⟨"q", "cos", "sin", 2, 2, 1, 1, 1, 1⟩

def witnessInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r o => .mk .fp32 (if r = ("q" : RegionName) ∧ o = 1 then witnessTiny else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "q" 0 = .mk .fp32 witnessOne ∧
      final.mem "q" 1 = .mk .fp32 witnessOne ∧
      final.mem "q" 2 = witnessInitial.mem "q" 2 ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 ∧
      witnessTiny ≠ witnessOne := by
  obtain ⟨final, he, hp⟩ := software_correctness rest .ieeeValue witnessParameters
    witnessInitial (by decide)
  refine ⟨final, he, ?_, ?_, ?_, witness_rounding.2.2.2.2⟩
  · have hv := hp.1 Bool.false (⟨0, by decide⟩, PUnit.unit)
    simpa [roundedValue, offset, cacheOffset, witnessParameters, witnessInitial,
      witness_rounding.1] using hv
  · have hv := hp.1 Bool.true (⟨0, by decide⟩, PUnit.unit)
    simpa [roundedValue, offset, cacheOffset, witnessParameters, witnessInitial,
      witness_rounding.2.1] using hv
  · apply hp.2 "q" 2
    right
    intro second i
    have hi : i.1.val = 0 := by
      have hil : i.1.val < 1 := i.1.isLt
      omega
    cases second <;> simp [offset, witnessParameters, witnessInitial, hi]

#axiomsClean software_correctness
#axiomsClean witness_rounding
#axiomsClean nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding.FPQ
