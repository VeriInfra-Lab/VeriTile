import bench.tritonbench_g.flash_decode2_phi.Proofs.FPContract

/-! Binary32 add/sub/mul/div and explicit exp32. The transcendental callback
is a named remaining binding; no mathematical `Real.exp` interpretation is
substituted. All basic arithmetic and memory results are independent of rest.
-/
namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
open VeriTile.Triton FP.Structural
open FlashDecode2Llama.FPStages (Word address scalarAddress roundedMaximum)
noncomputable section
set_option linter.unusedSimpArgs false

def bitBackend (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word) : Algebra Word :=
  let base := FlashDecode2Llama.FPStages.bitBackend rest cfg
  { base with unary := fun p op x =>
      if p = some .fp32 ∧ op = .exp then exp32 x else base.unary p op x }

def bits [Inhabited Word] (s : State Word) (R : RegionName) (o : Nat) : Word :=
  (s.mem R o).read .real

def maskedBits [Inhabited Word] (s : State Word) (R : RegionName) (o : Nat)
    (active : Prop) [Decidable active] : Word :=
  if active then bits s R o else FP.Value.zero .fp32

def roundedNormalization [Inhabited Word] (cfg : FP.Config) (s : State Word)
    (Acc SumExp : RegionName) (p : OutputParams) : TileIndex [p.width] → Word := fun i =>
  FP.div cfg (maskedBits s Acc (address s p.input i) (i.1.val < p.headDim))
    (bits s SumExp (scalarAddress s p.sum))

def copiedBits [Inhabited Word] (s : State Word) (Final : RegionName) (p : OutputParams) :
    TileIndex [p.width] → Word := fun i =>
  maskedBits s Final (address s p.input i) (i.1.val < p.headDim)

def roundedOldScale [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Prev Next : RegionName) (p : StepParams) : Word :=
  exp32 (FP.sub cfg (bits s Prev (scalarAddress s p.logic)) (bits s Next (scalarAddress s p.logic)))

def roundedExpLogic [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Log Next : RegionName) (p : StepParams) : Word :=
  exp32 (FP.sub cfg (bits s Log (scalarAddress s p.log + p.block)) (bits s Next (scalarAddress s p.logic)))

def roundedAccumulator [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Mid Log AccIn Prev Next : RegionName) (p : StepParams) :
    TileIndex [p.width] → Word := fun i =>
  FP.add cfg
    (FP.mul cfg (maskedBits s AccIn (address s p.acc i) (i.1.val < p.headDim))
      (roundedOldScale cfg exp32 s Prev Next p))
    (FP.mul cfg (roundedExpLogic cfg exp32 s Log Next p)
      (maskedBits s Mid (midAddress s p i) (i.1.val < p.headDim)))

def roundedSum [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Log SumIn Prev Next : RegionName) (p : StepParams) : Word :=
  FP.add cfg (FP.mul cfg (bits s SumIn (scalarAddress s p.sum))
    (roundedOldScale cfg exp32 s Prev Next p)) (roundedExpLogic cfg exp32 s Log Next p)

theorem read_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (R : RegionName) (o : Nat) :
    read (bitBackend rest cfg exp32) s R o = bits s R o := rfl

theorem maskedRead_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (R : RegionName) (o : Nat) (active : Prop) [Decidable active] :
    maskedRead (bitBackend rest cfg exp32) s R o active = maskedBits s R o active := by
  simp [maskedRead, maskedBits, bits, bitBackend, FlashDecode2Llama.FPStages.bitBackend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem normalization_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Acc SumExp : RegionName) (p : OutputParams) :
    normalizationValues (bitBackend rest cfg exp32) s Acc SumExp p = roundedNormalization cfg s Acc SumExp p := by
  funext i
  simp only [normalizationValues, roundedNormalization, read_software, maskedRead_software]
  simp [bitBackend, FlashDecode2Llama.FPStages.bitBackend, FP.Software.scalarFP32, FP.Software.fp32Context]

theorem copy_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Final : RegionName) (p : OutputParams) :
    copyValues (bitBackend rest cfg exp32) s Final p = copiedBits s Final p := by
  funext i
  exact maskedRead_software rest cfg exp32 s Final _ _

theorem oldScale_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Prev Next : RegionName) (p : StepParams) :
    oldScale (bitBackend rest cfg exp32) s Prev Next p = roundedOldScale cfg exp32 s Prev Next p := by
  simp only [oldScale, roundedOldScale, read_software]
  simp [bitBackend, FlashDecode2Llama.FPStages.bitBackend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem expLogic_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Log Next : RegionName) (p : StepParams) :
    expLogic (bitBackend rest cfg exp32) s Log Next p = roundedExpLogic cfg exp32 s Log Next p := by
  simp only [expLogic, roundedExpLogic, read_software]
  simp [bitBackend, FlashDecode2Llama.FPStages.bitBackend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem accumulator_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Mid Log AccIn Prev Next : RegionName) (p : StepParams) :
    accumulatorValues (bitBackend rest cfg exp32) s Mid Log AccIn Prev Next p =
      roundedAccumulator cfg exp32 s Mid Log AccIn Prev Next p := by
  funext i
  simp only [accumulatorValues, roundedAccumulator, maskedRead_software, oldScale_software, expLogic_software]
  simp [bitBackend, FlashDecode2Llama.FPStages.bitBackend, FP.Software.scalarFP32, FP.Software.fp32Context]

theorem sum_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (s : State Word) (Log SumIn Prev Next : RegionName) (p : StepParams) :
    sumValue (bitBackend rest cfg exp32) s Log SumIn Prev Next p =
      roundedSum cfg exp32 s Log SumIn Prev Next p := by
  simp only [sumValue, roundedSum, read_software, oldScale_software, expLogic_software]
  simp [bitBackend, FlashDecode2Llama.FPStages.bitBackend, FP.Software.scalarFP32, FP.Software.fp32Context]

#axiomsClean normalization_software
#axiomsClean copy_software
#axiomsClean accumulator_software
#axiomsClean sum_software
end
end VeriTile.Bench.TritonBenchG.FlashDecode2Phi.FPStages
