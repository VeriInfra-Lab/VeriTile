import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Frames
import bench.repairs.tritonbench_g.streamk_matmul_tile_checked.Proofs.Schedule

/-! Complete historical raw results for reuse by the source-bound shared IO adapter.
No source, domain, precision, output, byte or frame premise is changed. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.KernelResults
open Whole001 Checked KernelMemory
open VeriTile.Triton
noncomputable section

/-- Complete decoded source execution on raw arrays and memory, with the
independent matrix product at every logical and physical output and a frame.
The initial binding only describes arbitrary output/scratch contents. -/
theorem real (p : Source.Parameters) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (before : Nat → ℝ)
    (initial : Source.State ℝ p exterior) (memory : ValueBinding v initial before) :
    Spec.Real (∃ final,
      runValues p Arithmetic.real v initial before=some final ∧
      (∀ cell : Returned.Cell p,
        final.1.output (Returned.index p cell)=
          Reference.matrixProduct p initial.inputA initial.inputB cell.1 cell.2 ∧
        final.2 (v.outputBase+4*(Returned.index p cell).val)=
          Reference.matrixProduct p initial.inputA initial.inputB cell.1 cell.2) ∧
      final.1.inputA=initial.inputA ∧ final.1.inputB=initial.inputB ∧
      final.1.exterior=initial.exterior ∧
      (∀ address,ByteMemory.Outside (ByteExecution.cView v) address →
        ByteMemory.Outside (ByteExecution.sView v) address → final.2 address=before address)) := by
  let logical := Source.call p Arithmetic.real initial
  refine ⟨(logical,ValueMemory.rendered before v logical),
    values_run p Arithmetic.real v laws initial before memory,?_,
    (Source.call_frame p Arithmetic.real initial).1,
    (Source.call_frame p Arithmetic.real initial).2.1,
    (Source.call_frame p Arithmetic.real initial).2.2,values_frame v logical before⟩
  intro cell
  have value := Returned.real_contract p initial cell
  exact ⟨value,(ValueMemory.render_at _ v.outputBase logical.output (Returned.index p cell)).trans value⟩

#axiomsClean real
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.KernelResults
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.KernelResults
open Whole001 Checked KernelMemory
open VeriTile.Torch VeriTile.Triton
noncomputable section

/-- Full native source execution and independent ordered instruction target.
The established MMA, half conversion and atomic-add models are unchanged;
all FP32 words, output bytes and the exterior memory frame are covered. -/
theorem native (ref : Native.Reference) (p : Source.Parameters)
    (v : ByteExecution.Layout p) (laws : ByteExecution.Laws v) (before : ByteMemory.Read)
    (initial : Source.State FP32.Word p exterior) (memory : ByteBinding v initial before) :
    let ops := Native.operations (Native.referenceInstructions ref)
    ∃ final,runBytes p ops v initial before=some final ∧
      (∀ cell : Returned.Cell p,
        final.1.output (Returned.index p cell)=Returned.value p ops initial.inputA initial.inputB cell ∧
        ByteMemory.read32 final.2 (v.outputBase+4*(Returned.index p cell).val)=
          (Returned.value p ops initial.inputA initial.inputB cell).bits ∧
        ∀ j,j<4 → final.2 (v.outputBase+4*(Returned.index p cell).val+j)=
          ByteMemory.byte (Returned.value p ops initial.inputA initial.inputB cell).bits j) ∧
      final.1.inputA=initial.inputA ∧ final.1.inputB=initial.inputB ∧
      final.1.exterior=initial.exterior ∧
      (∀ address,ByteMemory.Outside (ByteExecution.cView v) address →
        ByteMemory.Outside (ByteExecution.sView v) address → final.2 address=before address) := by
  let ops := Native.operations (Native.referenceInstructions ref)
  let logical := Source.call p ops initial
  let after := ByteExecution.rendered before v logical
  refine ⟨(logical,after),bytes_run p ops v laws initial before memory,?_,
    (Source.call_frame p ops initial).1,(Source.call_frame p ops initial).2.1,
    (Source.call_frame p ops initial).2.2,bytes_frame v logical before⟩
  intro cell
  have value := Returned.value_contract p ops initial cell
  refine ⟨value,?_,?_⟩
  · exact (ByteMemory.render_read _ (ByteExecution.cView v) (ByteMemory.extend logical.output)
      (Returned.index p cell).val (Returned.index p cell).isLt).trans
      (congrArg (fun x : FP32.Word => x.bits)
        ((ByteMemory.extend_at logical.output (Returned.index p cell)).trans value))
  · intro j hj
    change ByteMemory.render _ (ByteExecution.cView v) (ByteMemory.extend logical.output)
      ((ByteExecution.cView v).base+4*(Returned.index p cell).val+j)=_
    rw [ByteMemory.render_at _ (ByteExecution.cView v) _ _ j (Returned.index p cell).isLt hj,
      ByteMemory.extend_at]
    exact congrArg (fun x : FP32.Word => ByteMemory.byte x.bits j) value

#axiomsClean native
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.KernelResults
