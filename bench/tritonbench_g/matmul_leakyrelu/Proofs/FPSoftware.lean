import bench.tritonbench_g.matmul_leakyrelu.Proofs.FPComplete
import bench.tritonbench_g.triton_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option linter.unusedSimpArgs false

/-- Named literal conversion policy: exact decimal 1/100 rounded to binary32.
This is a software profile, not a claim about unspecified compiler constants. -/
def slope : Word := FP.round .fp32 .ieeeValue (1/100 : Rat)
def bitLe (x y : Word) : Bool := FP.lt .ieeeValue x y || FP.eq .ieeeValue x y

def software (rest : Algebra Word) : Algebra Word :=
  {TritonMatmul.FPValue.software rest with
    literal := fun p d x => if x=0.01 then slope
      else (TritonMatmul.FPValue.software rest).literal p d x
    -- Both carriers are this same binary32 word under an explicit fp32 scope.
    -- This is a tag transport, never an identity model for a numeric narrowing.
    cast := fun p src dst x => if p=some .fp32 ∧ src=.fp32 ∧ dst=.real then x
      else (TritonMatmul.FPValue.software rest).cast p src dst x}

/-- Each source dot starts at +0 and uses ascending binary32 FMA after exact
binary16 decoding; its result is separately added to the live fp32 accumulator. Comparison is ordered: NaNs compare false. -/
def sequentialDot : Dot Word :=
  {apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b =>
      TritonMatmul.FPValue.bitDotAcc .half .half a b (fun _ => FP.Value.zero .fp32)
   le := fun _ _ => bitLe}

def bitA (p : Parameters) (s : State Word) (k : Nat) (i : AIndex p) : Word :=
  if i.2.1.val<p.K-k*BK p then read .half (s.mem p.A (aOffset p s k i))
  else QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.Value.zero .fp32)
def bitB (p : Parameters) (s : State Word) (k : Nat) (i : BIndex p) : Word :=
  if i.1.val<p.K-k*BK p then read .half (s.mem p.B (bOffset p s k i))
  else QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.Value.zero .fp32)
def bitUpdate (p : Parameters) (s : State Word) (k : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  fun i => FP.add .ieeeValue (a i)
    (TritonMatmul.FPValue.bitDotAcc .half .half (batch:=[]) (bitA p s k) (bitB p s k)
      (fun _ => FP.Value.zero .fp32) i)
def bitAccumulation (p : Parameters) (s : State Word) : CIndex p → Word :=
  foldRange (bitUpdate p s) 0 (trips p) 1 (fun _ => FP.Value.zero .fp32)
def bitActivation (enabled : Bool) (x : Word) : Word :=
  if enabled then if bitLe (FP.Value.zero .fp32) x then x else FP.mul .ieeeValue slope x else x
def bitOutput (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (bitActivation p.ACTIVATION (bitAccumulation p s i))

/-- No fallback callback remains in the value recurrence, including the
optional activation's fp32 literal, ordered comparison and rounded multiply. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (software rest) sequentialDot p .half s=bitOutput p s := by
  unfold outputValue activation accumulation update aValue bValue bitOutput bitActivation
    bitAccumulation bitUpdate bitA bitB
  simp [software,sequentialDot,dtype,TritonMatmul.FPValue.software,TritonMatmul.FPValue.encode,
    FP.Software.scalarFP32,FP.Software.fp32Context]
  norm_num

#axiomsClean software_reference
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue
