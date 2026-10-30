import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelNativeResult
import VeriTile.Triton.Float.BackendMemory

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.SourceEquivalence
open VeriTile Triton FP.Structural
open TritonBenchG.ReluStridedBuffer.Whole
open scoped VeriTile.Spec
noncomputable section
local instance wordInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def assumptions : Spec.Assumptions ComputeStmt := []

def original (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    FP.BackendMemory.Program (fp32Backend rest) := ⟨kernel c⟩

def folded (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    FP.BackendMemory.Program (fp32Backend rest) := ⟨foldedKernel c⟩

theorem successful_equivalence (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    FP.BackendMemory.Equivalent (original rest c) (folded rest c) :=
  KernelNativeResult.relu_strided_buffer_folded_equiv rest c

theorem source_equivalence (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    original rest c ≡[assumptions] folded rest c := by
  apply Spec.FloatingPoint.ofStructural (structural := FP.BackendMemory.Equivalent)
  · rfl
  · change ((kernel c).inputs,(kernel c).outputs)=
      ((foldedKernel c).inputs,(foldedKernel c).outputs)
    cases branch : c.oneTile <;> simp only [kernel,foldedKernel,branch] <;> rfl
  · exact successful_equivalence rest c

#print_fp_assumptions source_equivalence
#axiomsClean successful_equivalence
#axiomsClean source_equivalence
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.SourceEquivalence
