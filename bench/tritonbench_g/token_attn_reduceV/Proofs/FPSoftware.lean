import bench.tritonbench_g.token_attn_reduceV.Proofs.FPContract
import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedSoftware
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed (sequentialSum)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- The erased fp32 carrier conversions are bit preserving. The only untagged
real literal used by this source is exact +0 for masked loads; it is bound
explicitly here. Unsupported callbacks stay in `rest`. -/
def software (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend rest cfg with
    literal := fun p d x => if p = none ∧ d = .real ∧ x = 0 then FP.Value.zero .fp32
      else (FP.Software.scalarFP32 rest cfg).literal p d x }

/-- An explicit ascending-lane reduction, positive-zero seed, and one rounded
fp32 addition per leaf. This is a software plan; GPU tree equivalence is separate. -/
def softwareSum (cfg : FP.Config) : SumPrimitive (FP.Value .fp32) where
  apply := fun _ {_} ax keep xs => sequentialSum cfg ax keep xs

def bitProbability [Inhabited (FP.Value .fp32)] (p : Parameters) (s : State (FP.Value .fp32))
    (start n : Nat) : FP.Value .fp32 :=
  if start+n < count p s then (s.mem p.prob (probOffset p s n+start)).read .fp32
  else FP.Value.zero .fp32

def bitValue [Inhabited (FP.Value .fp32)] (p : Parameters) (s : State (FP.Value .fp32))
    (start n d : Nat) : FP.Value .fp32 :=
  if start+n < count p s then
    (s.mem p.value (valueOffset p s d+gather p s start n*p.valueBatchStride)).read .fp32
  else FP.Value.zero .fp32

def bitProducts [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (start : Nat) : TileIndex [p.block,p.width] → FP.Value .fp32 :=
  fun i => FP.mul cfg (bitProbability p s start i.1.val) (bitValue p s start i.1.val i.2.1.val)

def bitUpdate [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (start : Nat) (a : TileIndex [p.width] → FP.Value .fp32) :
    TileIndex [p.width] → FP.Value .fp32 :=
  fun i => FP.add cfg (a i) (sequentialSum cfg ⟨0, by simp⟩ Bool.false (bitProducts cfg p s start) i)

def bitOutput [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) : TileIndex [p.width] → FP.Value .fp32 :=
  foldRange (bitUpdate cfg p s) 0 (count p s) p.block (fun _ => FP.Value.zero .fp32)

theorem products_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (s : State (FP.Value .fp32)) (start : Nat) :
    products (software rest cfg) p s start = bitProducts cfg p s start := by
  funext i
  simp [products, probability, value, padding, bitProducts, bitProbability, bitValue,
    software, VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

theorem update_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (s : State (FP.Value .fp32)) :
    update (software rest cfg) (softwareSum cfg) p s = bitUpdate cfg p s := by
  funext start a i
  unfold update
  rw [products_software]
  simp [bitUpdate, softwareSum, software,
    VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

/-- Every output depends only on ScalarOps, the explicit reduction plan and
initial fp32 cells. No unused `rest` callback can influence the result. -/
theorem output_software [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (s : State (FP.Value .fp32)) :
    outputValues (software rest cfg) (softwareSum cfg) p s = bitOutput cfg p s := by
  funext i
  unfold outputValues accumulated
  rw [update_software]
  have hz : initial (software rest cfg) p = (fun _ => FP.Value.zero .fp32) := by
    funext j
    simp [initial, software, FP.Software.scalarFP32, FP.Software.fp32Context]
  rw [hz]
  simp [bitOutput, software, VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed.backend,
    FP.Software.scalarFP32, FP.Software.fp32Context]

#axiomsClean products_software
#axiomsClean update_software
#axiomsClean output_software
end
end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
