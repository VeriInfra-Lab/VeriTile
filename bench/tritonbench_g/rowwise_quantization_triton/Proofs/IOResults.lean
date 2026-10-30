import bench.tritonbench_g.rowwise_quantization_triton.Proofs.IOImplements
import bench.tritonbench_g.rowwise_quantization_triton.Proofs.KernelRealResult

namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
open VeriTile.Triton FP.Structural
noncomputable section

def realTarget (p : Launch) := IO.result p
  (fun values row col => ((mathematicalByte (values row) p.columns col).val : Int))
  (fun values row => logicalMaximum (values row) p.columns)

theorem real (rest : Algebra ℝ) (p : Launch) (schedule : Schedule p) :
    (IO.quantizerIO (KernelRealResult.algebra p rest) KernelRealResult.round
      (KernelProfile.equal KernelRealResult.ops) p schedule).ObservesInteger (IO.byteView p) (realTarget p) := by
  have h := IOImplements.observed KernelRealResult.ops id KernelRealResult.round rest p schedule
  have eq : IOImplements.observedTarget KernelRealResult.ops id KernelRealResult.round p=realTarget p := by
    funext vals o j
    fin_cases o
    · change ((narrow8 (BitVec.ofInt 64 (roundEven (rowScaled KernelRealResult.ops
          (IO.values p vals j.1.val j.2.1.val)
          (rowMaximum KernelRealResult.ops p (IO.values p vals j.1.val))))).toInt).val : Int)=_
      rw [KernelEncoding.narrow64]
      exact congrArg (fun b : Byte => (b.val : Int))
        (source_quantization_is_logical (fun _ => ⟨0,by decide⟩) p
          (IO.values p vals j.1.val) j.2.1.val j.2.1.isLt)
    · exact source_maximum_is_logical (fun _ => ⟨0,by decide⟩) p (IO.values p vals j.1.val)
  rw [eq] at h
  exact h

#axiomsClean real
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.IOResults
