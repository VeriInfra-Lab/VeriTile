import bench.tritonbench_g.matmul_triton_autotune.Proofs.FPComplete
import bench.tritonbench_g.triton_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
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
      else (TritonMatmul.FPValue.software rest).literal p d x}

/-- The complete source dotAcc uses the live seed and ascending fused binary32
FMA after exact binary16 decoding. Comparison is ordered: NaNs compare false. -/
def sequentialDotAcc : DotAcc Word :=
  {apply := fun _ _ _ _ _ {batch} {m} {k} {n} a b c =>
      TritonMatmul.FPValue.bitDotAcc .half .half a b c
   le := fun _ _ => bitLe}

def bitA (p : Parameters) (s : State Word) (k : Nat) (i : AIndex p) : Word :=
  if i.2.1.val<p.K-k*BK p then read .half (s.mem p.A (aOffset p s k i))
  else QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.Value.zero .fp32)
def bitB (p : Parameters) (s : State Word) (k : Nat) (i : BIndex p) : Word :=
  if i.1.val<p.K-k*BK p then read .half (s.mem p.B (bOffset p s k i))
  else QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.Value.zero .fp32)
def bitUpdate (p : Parameters) (s : State Word) (k : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  TritonMatmul.FPValue.bitDotAcc .half .half (batch:=[]) (bitA p s k) (bitB p s k) a
def bitAccumulation (p : Parameters) (s : State Word) : CIndex p → Word :=
  foldRange (bitUpdate p s) 0 p.numKBlocks 1 (fun _ => FP.Value.zero .fp32)
def bitActivation (enabled : Bool) (x : Word) : Word :=
  if enabled then if bitLe (FP.Value.zero .fp32) x then x else FP.mul .ieeeValue slope x else x
def bitOutput (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (bitActivation p.ACTIVATION (bitAccumulation p s i))

/-- No fallback callback remains in the value recurrence, including the
optional activation's fp32 literal, ordered comparison and rounded multiply. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (software rest) sequentialDotAcc p .half s=bitOutput p s := by
  unfold outputValue activation accumulation update aValue bValue bitOutput bitActivation
    bitAccumulation bitUpdate bitA bitB
  simp [software,sequentialDotAcc,dtype,TritonMatmul.FPValue.software,TritonMatmul.FPValue.encode,
    FP.Software.scalarFP32,FP.Software.fp32Context]
  norm_num

#axiomsClean software_reference
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
