import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPBlocksContract
import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPV2Contract
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

namespace FPBlocks

def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (key : Bool)
    (p : Parameters) (s : State (FP.Value .fp32)) (second : Bool) (i : Index p) : FP.Value .fp32 :=
  let x0 := (s.mem p.input (offset key p s Bool.false i)).read .fp32
  let x1 := (s.mem p.input (offset key p s Bool.true i)).read .fp32
  let c := (s.mem p.cosine (cacheOffset p s i)).read .fp32
  let z := (s.mem p.sine (cacheOffset p s i)).read .fp32
  if second then FP.add cfg (FP.mul cfg x0 z) (FP.mul cfg x1 c)
  else FP.sub cfg (FP.mul cfg x0 c) (FP.mul cfg x1 z)

theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (key : Bool)
    (p : Parameters) (s : State (FP.Value .fp32))
    (hw : Pair.WritesApart (active key p s) (offset key p s)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (source key p) s = some final ∧
      Post key p s final (roundedValue cfg key p s) := by
  obtain ⟨final, he, hp⟩ := correctness (FP.Software.scalarFP32 rest cfg) key p s hw
  refine ⟨final, he, ?_, hp.2⟩
  intro second i hi
  simpa [value, roundedValue, FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 second i hi
#axiomsClean software_correctness
end FPBlocks

namespace FPV2

/-- Concrete implementation of the only real-literal conversion reached here:
positive zero's conversion to binary32. Other real conversions remain in rest;
no identity law for arbitrary real values is asserted. -/
def zeroDefaults (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  { rest with
    literal := fun p d x => if p = none ∧ d = .real ∧ x = 0 then FP.Value.zero .fp32
      else rest.literal p d x
    cast := fun p src dst v =>
      if p = none ∧ src = .real ∧ dst = .fp32 ∧ v = FP.Value.zero .fp32
      then FP.Value.zero .fp32 else rest.cast p src dst v }

def softwareBackend (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  FP.Software.scalarFP32 (zeroDefaults rest) cfg

theorem zero_software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) :
    zero (softwareBackend rest cfg) = FP.Value.zero .fp32 := by
  norm_num [zero, softwareBackend, FP.Software.scalarFP32, FP.Software.fp32Context, zeroDefaults]

def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) (second : Bool)
    (i : TileIndex [p.half]) : FP.Value .fp32 :=
  let x0 := (s.mem p.input (inputOffset p s Bool.false i)).read .fp32
  let x1 := (s.mem p.input (inputOffset p s Bool.true i)).read .fp32
  let c := if s.pids 1 < p.totalTokens then (s.mem p.cosine (cacheOffset p s i)).read .fp32
    else FP.Value.zero .fp32
  let z := if s.pids 1 < p.totalTokens then (s.mem p.sine (cacheOffset p s i)).read .fp32
    else FP.Value.zero .fp32
  if second then FP.add cfg (FP.mul cfg x0 z) (FP.mul cfg x1 c)
  else FP.sub cfg (FP.mul cfg x0 c) (FP.mul cfg x1 z)

theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (cache : Bool)
    (p : Parameters) (s : State (FP.Value .fp32))
    (hw : Pair.WritesApart (active p s) (outputOffset cache p s)) :
    ∃ final, FP.Structural.exec (softwareBackend rest cfg) (source cache p) s = some final ∧
      Post cache p s final (roundedValue cfg p s) := by
  obtain ⟨final, he, hp⟩ := correctness (softwareBackend rest cfg) cache p s hw
  refine ⟨final, he, ?_, hp.2⟩
  intro second i hi
  have hv := hp.1 second i hi
  simp only [value, coefficient, zero_software] at hv
  simpa [roundedValue, softwareBackend, FP.Software.scalarFP32, FP.Software.fp32Context] using hv
#axiomsClean zero_software
#axiomsClean software_correctness
end FPV2
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad
