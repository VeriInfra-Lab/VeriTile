import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.IOResults
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
open VeriTile.Triton FP.Structural
noncomputable section

local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def nativeTarget (a : KernelNativeProfile.Arithmetic) (p : Launch) := IO.result p
  (fun values row col => ((KernelNativeProfile.quantized a p (values row) col).val : Int))
  (fun values row => KernelEncoding.packHalf (KernelNativeProfile.storedMaximum a p (values row)))

theorem native (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Launch) (schedule : Schedule p) :
    (IO.quantizerIO (KernelNativeProfile.algebra a p rest) (KernelNativeProfile.llrint a)
      (KernelNativeProfile.equal a) p schedule).ObservesInteger (IO.byteView p) (nativeTarget a p) := by
  exact IOImplements.observed (KernelNativeProfile.operations a)
    (fun x => KernelEncoding.packHalf (a.half x)) (KernelNativeProfile.llrint a) rest p schedule

#axiomsClean native
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Floating
open VeriTile.Triton FP.Structural KernelNativeProfile
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent source-order native formulas, preserving the zero denominator
branch, fixed reduction tree, signed64 llrint policy, low-byte INT8 result and
FP16 maximum. packHalf is an injective carrier encoding; its cell tag is FP16. -/
def formula (a : Arithmetic) (p : Launch) := IO.result p
  (fun values row col => ((quantized a p (values row) col).val : Int))
  (fun values row => KernelEncoding.packHalf (storedMaximum a p (values row)))

abbrev quantizerIO (a : Arithmetic) (rest : Algebra Word) (p : Launch) (schedule : Schedule p) :=
  IO.quantizerIO (algebra a p rest) (llrint a) (equal a) p schedule

/-- All FP32 words remain admitted. The original scalar/lowering model is
explicit; this is a native observation theorem, with no numerical rewrite law. -/
theorem rowwise_grid_fp32 (a : Arithmetic) (rest : Algebra Word)
    (p : Launch) (schedule : Schedule p) :
    (quantizerIO a rest p schedule).ObservesInteger (IO.byteView p) (formula a p) :=
  IOResults.native a rest p schedule

#specNonCircular formula avoiding [RowwiseQuantizationTriton.source,KernelExecution.exec,KernelGrid.run]
#axiomsClean rowwise_grid_fp32
#stmtSurfaceSubset rowwise_grid_fp32 ⊆
  [Arithmetic,FP.Structural.Algebra,Word,Launch,Schedule,quantizerIO,IntegerProgramIO.ObservesInteger,
   IO.byteView,formula,instInhabitedWord]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Floating
