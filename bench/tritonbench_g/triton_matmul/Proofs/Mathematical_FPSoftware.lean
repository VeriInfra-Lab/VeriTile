import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPComplete
import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_E4M3
import bench.tritonbench_g.quantize_kv_transform.Sources.Mathematical_MixedBits
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option linter.unusedSimpArgs false

def encode (st : Storage) (x : Word) : Word :=
  match st with
  | .half => QuantizeKvTransform.MixedBits.narrow .ieeeValue x
  | .e4m3 => E4M3.narrow x

def decode (st : Storage) (x : Word) : Word :=
  match st with
  | .half => QuantizeKvTransform.MixedBits.widen x
  | .e4m3 => E4M3.widen x

def software (rest : Algebra Word) : Algebra Word :=
  {FP.Software.scalarFP32 rest .ieeeValue with
    literal := fun precision d x => if x=0 then FP.Value.zero .fp32
      else (FP.Software.scalarFP32 rest .ieeeValue).literal precision d x
    cast := fun precision src dst x =>
      match src,dst with
      | .real,.fp16 => encode .half x
      | .fp16,.fp16 => x
      | .f8e4,.f8e4 => x
      | .real,.f8e4 => encode .e4m3 x
      | src,dst => (FP.Software.scalarFP32 rest .ieeeValue).cast precision src dst x}

/-- The software primitive takes the live accumulator as its seed and uses
one genuinely fused fp32 FMA per K lane in ascending order. It is an explicit
software dotAcc profile; no GPU tensor-core schedule is inferred. -/
def bitDotAcc (left right : Storage) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word)
    (c : TileIndex (batch++[m,n]) → Word) : TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc e =>
      FP.fma .ieeeValue (decode left (a (i.1,e,PUnit.unit)))
        (decode right (b (e,i.2.1,PUnit.unit))) acc) (c i)
  | _ :: tail => fun i => bitDotAcc left right (batch:=tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length

def sequentialDotAcc : DotAcc Word :=
  ⟨fun _ l r _ _ {batch} {m} {k} {n} a b c =>
    bitDotAcc (if l=.f8e4 then .e4m3 else .half)
      (if r=.f8e4 then .e4m3 else .half) a b c⟩

def bitA (p : Parameters) (st : Storage) (s : State Word) (k : Nat) (i : AIndex p) : Word :=
  if i.2.1.val<p.K-k*BK p then read st (s.mem p.A (aOffset p s k i))
  else encode st (FP.Value.zero .fp32)
def bitB (p : Parameters) (st : Storage) (s : State Word) (k : Nat) (i : BIndex p) : Word :=
  if i.1.val<p.K-k*BK p then read st (s.mem p.B (bOffset p s k i))
  else encode st (FP.Value.zero .fp32)
def bitUpdate (p : Parameters) (st : Storage) (s : State Word) (k : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  bitDotAcc st st (batch:=[]) (bitA p st s k) (bitB p st s k) a
def bitAccumulation (p : Parameters) (st : Storage) (s : State Word) : CIndex p → Word :=
  foldRange (bitUpdate p st s) 0 p.numKBlocks 1 (fun _ => FP.Value.zero .fp32)
def bitOutput (p : Parameters) (st : Storage) (s : State Word) (i : CIndex p) : Word :=
  encode st (bitAccumulation p st s i)

/-- Every used callback is closed for both storage profiles, independently of
arbitrary rest. Source masks, clamps, K trips and the live dot accumulator are
retained in this independent bit-pattern recurrence. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (st : Storage) (s : State Word) :
    outputValue (software rest) sequentialDotAcc p st s=bitOutput p st s := by
  cases st <;>
    unfold outputValue accumulation update aValue bValue bitOutput bitAccumulation bitUpdate bitA bitB
  all_goals simp [software,sequentialDotAcc,dtype,encode,FP.Software.scalarFP32,FP.Software.fp32Context]
  all_goals norm_num

#axiomsClean software_reference
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
