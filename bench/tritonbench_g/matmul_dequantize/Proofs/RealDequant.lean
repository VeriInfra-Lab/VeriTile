import bench.tritonbench_g.matmul_dequantize.Proofs.RealBase
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant32
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealDequant
open VeriTile.Triton FP.Structural
open Real
noncomputable section
namespace D16
open VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
def inputs (p : Parameters) (s : State ℝ) : RawInputs where
  a := fun _ _ => 0
  weight := fun k n => (s.mem p.weight (k*p.weightRowStride+n*p.weightColumnStride)).read .int
  zero := fun g n => (s.mem p.zero (g*p.zeroRowStride+n*p.zeroColumnStride)).read .int
  scale := fun g n => (s.mem p.scale (g*p.scaleRowStride+n*p.scaleColumnStride)).read .fp16
theorem formula (p : Parameters) (s : State ℝ) (i : Index p) :
    value scalar p s i=rawDequant (inputs p s) p.groupSize (row p s i) (column p s i) := by
  simp [value,difference,weightOffset,zeroOffset,scaleOffset,inputs,rawDequant,signedNibble,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble,scalar]
end D16
namespace D32
open Dequant32
def inputs (p : Parameters) (s : State ℝ) : RawInputs where
  a := fun _ _ => 0
  weight := fun k n => (s.mem p.weight (k*p.weightRowStride+n*p.weightColumnStride)).read .int
  zero := fun g n => (s.mem p.zero (g*p.zeroRowStride+n*p.zeroColumnStride)).read .int
  scale := fun g n => (s.mem p.scale (g*p.scaleRowStride+n*p.scaleColumnStride)).read .fp32
theorem formula (p : Parameters) (s : State ℝ) (i : Index p) :
    value scalar p s i=rawDequant (inputs p s) p.groupSize (row p s i) (column p s i) := by
  simp [value,difference,weightOffset,zeroOffset,scaleOffset,inputs,rawDequant,signedNibble,Dequant32.nibble,scalar]
end D32
#axiomsClean D16.formula
#axiomsClean D32.formula
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealDequant
