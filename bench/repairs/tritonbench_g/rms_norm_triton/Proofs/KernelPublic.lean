import bench.repairs.tritonbench_g.rms_norm_triton.Kernels
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.LegacyPastedSurface
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualRow
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.KernelNative
import bench.tritonbench_g.rms_norm_triton.Proofs.KernelIndex
import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBinding
import bench.tritonbench_g.rms_norm_triton.Proofs.NativeKernel

namespace VeriTile.Bench.Repairs.RmsNormTriton
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
variable {x w xc wc y : ActualTensor}

 theorem ActualTensor.read_view (t : ActualTensor) (memory : Whole.Memory α)
    (i : Nat) (hi : i < t.tensor.metadata.numel) : t.view.read memory i = t.read memory i := by
  simp only [Whole.TensorView.read,ActualTensor.view,Whole.TensorView.offset,ActualTensor.read,t.indices i hi]

/-- Agreement with the shared Torch logical-value API through the physical
reader of the same backing allocation. The selected guards rule out lazy neg. -/
 theorem ActualTensor.read_logicalValue (t : ActualTensor)
    (read : Nat → VeriTile.Torch.FP32.Word) (memory : Whole.Memory (BitVec 32))
    (physical : ∀ offset, memory t.tensor.storage.object offset =
      (read (t.tensor.storage.address offset)).bits)
    (noNeg : t.tensor.metadata.isNeg=Bool.false) (i : Nat) :
    t.read memory i = (VeriTile.Torch.FP32.logicalValue read t.tensor t.laws i).bits := by
  simp only [ActualTensor.read,VeriTile.Torch.FP32.logicalValue,noNeg,Bool.false_eq_true,
    if_false,physical]

/-- Metadata and storage bindings refer to these same actual source arguments
and materialization results. The API laws do not supply any RMSNorm value. -/
structure KernelTensorBinding (a : Whole.Prepared α input weight entry live)
    (x w xc wc y : ActualTensor) : Prop where
  views : Whole.TensorBindings a x.view w.view xc.view wc.view y.view
  inputLive : live input.allocation
  weightLive : live weight.allocation
  copyFlags : xc.tensor.metadata.isNeg=Bool.false ∧ wc.tensor.metadata.isNeg=Bool.false ∧
    y.tensor.metadata.isNeg=Bool.false ∧ xc.isConj=Bool.false ∧ wc.isConj=Bool.false ∧ y.isConj=Bool.false

def KernelTensorBinding.entry {α : Type} {input weight : Whole.Tensor}
    {entry : Whole.Memory α} {live : Nat → Prop} {a : Whole.Prepared α input weight entry live}
    (b : KernelTensorBinding a x w xc wc y) :
    Binding x.info w.info input weight live where
  inputExisted := b.inputLive
  weightExisted := b.weightLive
  input := ⟨b.views.input.shape,fun _ => b.views.input.dtype,b.views.input.device⟩
  weight := ⟨b.views.weight.shape,fun _ => b.views.weight.dtype,b.views.weight.device⟩

/-- Ordinary storage identity/capacity observations connect the finite source
index proof to the actual Tensor allocation, including the original views. -/
structure KernelStorage (api : PreparedAPI a n) (x w xc wc y : ActualTensor) : Prop where
  pointers : api.pointers = ⟨xc.tensor.dataPtr,wc.tensor.dataPtr,y.tensor.dataPtr⟩
  storage : ∀ v ∈ [x.view,w.view,xc.view,wc.view,y.view],
    api.byteBase v.allocation=v.storageBase ∧ api.capacity v.allocation=v.capacity

 theorem legacy_source_row_identity (c : Whole.Layout) (eps : ℝ) :
    c.source eps = LegacyPastedSurface.kernel c.Y c.X c.W c.N c.B eps := rfl

 theorem source_row_identity (c : Whole.Layout) :
    ActualRow.source c = runtimeKernel c.Y c.X c.W c.N c.B := rfl


 theorem logical_tensor_result [Inhabited α] (a : Whole.Prepared α input weight entry live)
    (mdl : Algebra α) (x w xc wc y : ActualTensor) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x.info w.info n eps) (binding : KernelTensorBinding a x w xc wc y)
    (schedule : Whole.Schedule input n.dimensions) :
    ∃ final, a.entryRun mdl n.dimensions eps.value schedule = some final ∧
      Whole.KernelResult (fun k => Whole.rowValue mdl (a.layout n.dimensions) eps.value
        (fun j => x.read entry ((k/columns n.dimensions)*columns n.dimensions+j))
        (w.read entry) (k%columns n.dimensions)) y.view a.output.memory (a.commit final) := by
  have domain := binding.entry.domain n eps c
  obtain ⟨final,ran,result⟩ := Whole.prepared_tensor_result a mdl n.dimensions eps.value domain binding.views schedule
  refine ⟨final,ran,?_,result.2⟩
  intro k hk
  rw [result.1 k hk]
  have shape := Whole.shape_product domain
  have kn : k < input.numel := by rw [← a.output_numel binding.views]; exact hk
  have rb : k / columns n.dimensions < Whole.rows input n.dimensions :=
    (Nat.div_lt_iff_lt_mul c.positive).2 (by simpa only [Whole.columns,columns] using (shape ▸ kn))
  have cb := Nat.mod_lt k c.positive
  apply Whole.rowValue_congr _ _ _ _ _ _ _ _ cb
  · intro j hj
    apply x.read_view
    have bound : k/columns n.dimensions*columns n.dimensions+j < input.numel := by
      rw [shape]
      calc
        _ < (k/columns n.dimensions)*columns n.dimensions+columns n.dimensions := Nat.add_lt_add_left hj _
        _ = (k/columns n.dimensions+1)*columns n.dimensions := by ring
        _ ≤ Whole.rows input n.dimensions*Whole.columns n.dimensions := Nat.mul_le_mul_right _ rb
    simpa only [Whole.Tensor.numel,binding.entry.input.shape,ActualTensor.info] using bound
  · intro j hj
    apply w.read_view
    simpa only [VeriTile.Torch.TensorMetadata.numel,← c.weightShape,ActualTensor.info,Whole.columns] using hj


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsNormTriton
