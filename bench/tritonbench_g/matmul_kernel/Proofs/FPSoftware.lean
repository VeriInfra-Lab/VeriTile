import bench.tritonbench_g.matmul_kernel.Proofs.FPComplete
import bench.tritonbench_g.triton_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option linter.unusedSimpArgs false
/-- Reuse the checked binary16 / binary32 encoding and scalar profile. -/
abbrev software := TritonMatmul.FPValue.software
abbrev narrow := QuantizeKvTransform.MixedBits.narrow .ieeeValue
/-- Explicit software schedule: each dot receives its live carry and performs
ascending binary32 fused multiply-adds on decoded binary16 inputs. Tensor-core
lowering, FTZ and implementation-specific NaN policy remain native bindings. -/
def sequentialDotAcc : DotAcc Word :=
  ⟨fun _ _ _ _ _ {batch} {m} {k} {n} a b c =>
    TritonMatmul.FPValue.bitDotAcc .half .half a b c⟩
def bitA (p : Parameters) (s : State Word) (k : Nat) (i : AIndex p) : Word :=
  (s.mem p.A (aOffset p s k i)).read .fp16
def bitB (p : Parameters) (s : State Word) (k : Nat) (i : BIndex p) : Word :=
  (s.mem p.B (bOffset p s k i)).read .fp16
def bitUpdate (p : Parameters) (s : State Word) (k : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  TritonMatmul.FPValue.bitDotAcc .half .half (batch:=[]) (bitA p s k) (bitB p s k) a
def bitAccumulation (p : Parameters) (s : State Word) : CIndex p → Word :=
  foldRange (bitUpdate p s) 0 p.numKBlocks 1 (fun _ => FP.Value.zero .fp32)
def bitOutput (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  narrow (bitAccumulation p s i)
/-- The full output recurrence is closed and independent of every fallback
callback. Loads remain unmasked, no tail arithmetic is removed. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (software rest) sequentialDotAcc p .half s=bitOutput p s := by
  unfold outputValue accumulation update aValue bValue bitOutput bitAccumulation bitUpdate bitA bitB
  simp [software,sequentialDotAcc,dtype,read,narrow,TritonMatmul.FPValue.software,
    TritonMatmul.FPValue.encode,FP.Software.scalarFP32,FP.Software.fp32Context]
#axiomsClean software_reference
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
