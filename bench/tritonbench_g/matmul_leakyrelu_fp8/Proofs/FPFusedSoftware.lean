import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPFusedComplete
import bench.tritonbench_g.triton_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPFused
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
FMA after exact binary16 or E4M3FN decoding. Comparison is ordered: NaNs compare false. -/
inductive BitStorage where | half | e4m3
  deriving DecidableEq
def storage : BitStorage → Storage | .half => .half | .e4m3 => .e4m3
def bitStorage : BitStorage → TritonMatmul.FPValue.Storage | .half => .half | .e4m3 => .e4m3

def sequentialDotAcc : DotAcc Word :=
  {apply := fun _ l r _ _ {batch} {m} {k} {n} a b c =>
      TritonMatmul.FPValue.bitDotAcc (if l=.f8e4 then .e4m3 else .half)
        (if r=.f8e4 then .e4m3 else .half) a b c
   le := fun _ _ => bitLe}

def bitA (p : Parameters) (st : BitStorage) (s : State Word) (k : Nat) (i : AIndex p) : Word :=
  if i.2.1.val<p.K-k*BK p then read (storage st) (s.mem p.A (aOffset p s k i))
  else TritonMatmul.FPValue.encode (bitStorage st) (FP.Value.zero .fp32)
def bitB (p : Parameters) (st : BitStorage) (s : State Word) (k : Nat) (i : BIndex p) : Word :=
  if i.1.val<p.K-k*BK p then read (storage st) (s.mem p.B (bOffset p s k i))
  else TritonMatmul.FPValue.encode (bitStorage st) (FP.Value.zero .fp32)
def bitUpdate (p : Parameters) (st : BitStorage) (s : State Word) (k : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  TritonMatmul.FPValue.bitDotAcc (bitStorage st) (bitStorage st) (batch:=[]) (bitA p st s k) (bitB p st s k) a
def bitAccumulation (p : Parameters) (st : BitStorage) (s : State Word) : CIndex p → Word :=
  foldRange (bitUpdate p st s) 0 (trips p) 1 (fun _ => FP.Value.zero .fp32)
def bitActivation (enabled : Bool) (x : Word) : Word :=
  if enabled then if bitLe (FP.Value.zero .fp32) x then x else FP.mul .ieeeValue x slope else x
def bitOutput (p : Parameters) (st : BitStorage) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (bitActivation p.ACTIVATION (bitAccumulation p st s i))

/-- No fallback callback remains in the value recurrence, including the
optional activation's fp32 literal, ordered comparison and rounded multiply. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (st : BitStorage) (s : State Word) :
    outputValue (software rest) sequentialDotAcc p (storage st) s=bitOutput p st s := by
  cases st
  all_goals unfold outputValue activation accumulation update aValue bValue bitOutput bitActivation bitAccumulation bitUpdate bitA bitB
  all_goals simp [storage,bitStorage,software,sequentialDotAcc,dtype,
    TritonMatmul.FPValue.software,TritonMatmul.FPValue.encode,
    FP.Software.scalarFP32,FP.Software.fp32Context]
  all_goals norm_num

#axiomsClean software_reference
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPFused
