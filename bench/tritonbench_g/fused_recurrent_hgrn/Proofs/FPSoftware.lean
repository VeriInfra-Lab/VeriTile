import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.FPStages
import VeriTile.Triton.Float.ScalarBackend

/-! Concrete binary32 software contracts for the selected HGRN stages.
Every scalar operation below invokes ScalarOps and rounds its result. The
unfused fp32 software profile is not a CUDA/compiler conformance assertion.
No callback from `rest` appears in any output formula. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

def bitsAt [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32))
    (R : RegionName) (off : Nat) : FP.Value .fp32 := (s.mem R off).read .real

def roundedForward [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (BHPrev X G : RegionName) (t T D BD : Nat)
    (i : TileIndex [BD]) : FP.Value .fp32 :=
  FP.add cfg
    (FP.mul cfg (bitsAt s G (rowOffset s t T D BD i))
      (bitsAt s BHPrev (stateOffset s D BD i)))
    (bitsAt s X (rowOffset s t T D BD i))

specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (roundedForward cfg s BHPrev X G t T D BD) := by
  obtain ⟨final, he, hp⟩ := forward_fp32_stage (FP.Software.scalarFP32 rest cfg)
    BHPrev X G O t T D BD s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [forwardValues, forwardValue, maskedInput, roundedForward, bitsAt, hi,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def roundedDx [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (DHPrev DO : RegionName) (t T D BD : Nat)
    (i : TileIndex [BD]) : FP.Value .fp32 :=
  FP.add cfg (bitsAt s DHPrev (rowOffset s t T D BD i))
    (bitsAt s DO (rowOffset s t T D BD i))

specification dx_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDx cfg s DHPrev DO t T D BD) := by
  obtain ⟨final, he, hp⟩ := dx_fp32_stage (FP.Software.scalarFP32 rest cfg) DHPrev DO DX t T D BD s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [dxValues, dxValue, maskedInput, roundedDx, bitsAt, hi,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def roundedDxTwo [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (DHPrev DO G : RegionName) (t T D BD : Nat)
    (i : TileIndex [BD]) : FP.Value .fp32 :=
  FP.add cfg
    (FP.mul cfg
      (FP.add cfg (bitsAt s DHPrev (rowOffset s (t+1) T D BD i))
        (bitsAt s DO (rowOffset s (t+1) T D BD i)))
      (bitsAt s G (rowOffset s (t+1) T D BD i)))
    (bitsAt s DO (rowOffset s t T D BD i))

specification dx_two_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDxTwo cfg s DHPrev DO G t T D BD) := by
  obtain ⟨final, he, hp⟩ := dx_two_fp32_stage (FP.Software.scalarFP32 rest cfg) DHPrev DO G DX t T D BD s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [dxTwoValues, dxTwoValue, maskedInput, roundedDxTwo, bitsAt, hi,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def roundedDg [Inhabited (FP.Value .fp32)] (cfg : FP.Config)
    (s : State (FP.Value .fp32)) (DHPrev DO O H0 : RegionName) (U : Bool) (t T D BD : Nat)
    (i : TileIndex [BD]) : FP.Value .fp32 :=
  let previous := if 0 < t then bitsAt s O (rowOffset s (t-1) T D BD i)
    else if U then bitsAt s H0 (stateOffset s D BD i) else FP.Value.zero .fp32
  FP.mul cfg
    (FP.add cfg (bitsAt s DHPrev (rowOffset s t T D BD i)) (bitsAt s DO (rowOffset s t T D BD i)))
    previous

specification dg_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (roundedDg cfg s DHPrev DO O H0 U t T D BD) := by
  obtain ⟨final, he, hp⟩ := dg_fp32_stage (FP.Software.scalarFP32 rest cfg) DHPrev DO O H0 DG U t T D BD s
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [dgValues, maskedInput, roundedDg, bitsAt, hi,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 i hi

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessQuarterUlp : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_rounding :
    FP.add .ieeeValue (FP.mul .ieeeValue witnessOne witnessOne) witnessQuarterUlp = witnessOne ∧
      witnessOne.toRat? = some 1 ∧ witnessQuarterUlp.toRat? = some (1 / 33554432) ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

/-- One active lane and one masked tail, empty initial registers. The exact
input 2^-25 is lost at the fp32 addition to 1, so this cannot be an identity
rounding/exact-real witness. Initial output memory contains the tiny value. -/
def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun R _ => .mk .real
    (if R = ("carry" : RegionName) ∨ R = ("gate" : RegionName) then witnessOne else witnessQuarterUlp)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (forward "carry" "input" "gate" "out" 0 1 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := forward_software_fp32 rest .ieeeValue
    "carry" "input" "gate" "out" 0 1 1 2 roundingInitial
  refine ⟨final, he, ?_, ?_, witness_rounding.2.2.2⟩
  · have hv := hp.1 (⟨0, by decide⟩, PUnit.unit) (by simp [channel, roundingInitial])
    simpa [roundedForward, bitsAt, rowOffset, stateOffset, channel, roundingInitial,
      witness_rounding.1] using hv
  · apply hp.2 "out" 1
    right
    intro i hi
    simp [channel, roundingInitial] at hi
    simp [rowOffset, channel, roundingInitial]
    omega

#axiomsClean forward_software_fp32
#axiomsClean dx_software_fp32
#axiomsClean dx_two_software_fp32
#axiomsClean dg_software_fp32
#axiomsClean witness_rounding
#axiomsClean forward_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages
