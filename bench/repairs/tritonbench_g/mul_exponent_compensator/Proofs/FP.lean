import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Host
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Grid
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPProof
open VeriTile.Triton FP.Structural
noncomputable section
abbrev FPWord := FP.Value .fp32
def constant : FPWord := ⟨constantWord⟩
def nativeReference (multiply : FPWord → FPWord → FPWord) (x : FPWord) : FPWord := multiply x constant
def softwareReference (x : FPWord) : FPWord := FP.mul FP.Config.ieeeValue x constant
def finite (x : FPWord) : Prop := x.toRat?.isSome

/-- Exactly the historical local port: native FP32 x*2^112 agrees with IEEE
nearest-even/no-FTZ multiplication on finite x. No NaN payload assumption. -/
def LocalMultiplyExact (multiply : FPWord → FPWord → FPWord) : Prop :=
  ∀ x,finite x → nativeReference multiply x=softwareReference x

def operations (multiply : FPWord → FPWord → FPWord) : TritonMul2.Operations FPWord :=
  ⟨nativeReference multiply⟩

def algebra (rest : Algebra FPWord) (multiply : FPWord → FPWord → FPWord) : Algebra FPWord :=
  {rest with
    literal := fun _ _ _ => constant
    fp32Load := id
    cast := fun _ _ _ x => x
    binary :=fun precision dtype op x y => match op with
      | .mul => multiply x y
      | _ => rest.binary precision dtype op x y}

theorem source_scalar (rest : Algebra FPWord) (multiply : FPWord → FPWord → FPWord) (x : FPWord) :
    scalar (algebra rest multiply) x=nativeReference multiply x := rfl

theorem constant_exact : constant.toRat?=some ((2 : Rat)^112) := by
  have raw : constant.raw=2004877312 := rfl
  norm_num [FP.Value.toRat?,FP.Value.decode,FP.Value.exponent,FP.Value.fraction,
    FP.Value.sign,raw,FP.Format.fractionBits,FP.pow2,Int.toNat]

#print LocalMultiplyExact
#specNonCircular softwareReference avoiding [Host.runWrapper,Native.sourceLaunch,TritonMul2.launch]
#specNonCircular LocalMultiplyExact avoiding [Host.runWrapper,Native.sourceLaunch,TritonMul2.launch]
#axiomsClean source_scalar
#axiomsClean constant_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.FPProof
