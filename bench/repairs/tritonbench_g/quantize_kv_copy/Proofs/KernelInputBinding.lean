import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelInputBinding
open VeriTile.Triton FP.Structural Kernel KernelWhole KernelTensor
noncomputable section
local instance : Inhabited Word := ⟨KernelModels.zero⟩

/-- The complete-source entry reads both actual caller inputs. Its FP32 load
is the four bytes at the caller input Tensor coordinate; its destination load
is the signed INT32 interpretation of that Tensor's actual four backing bytes.
Only the original data spans need be disjoint. -/
def SourceInputs (t : Tensors) (before : Heap) : Prop :=
  let m := metadata t
  let w := words t before
  (∀ pid g c,pid<(request m w).programs → g<(request m w).groups → c<8 →
    let i := inputOffset (parameters m w) pid g c
    (initial m before).mem "K" i=.mk .real (decodeWord (fun b => before (base m .input+4*i+b.val))) ∧
    values m w before pid g c=logicalFP32 before (t .input) i) ∧
  (∀ token,token<(request m w).tokens →
    (initial m before).mem "D" token=.mk .int (int32At before (base m .destination+4*token)).toInt ∧
    words t before token=logicalInt32 before (t .destination) token)

theorem actual (t : Tensors) (laws : ∀ r,Laws (t r)) (before : Heap)
    (h : InputContract (metadata t) (words t before)) : SourceInputs t before := by
  constructor
  · intro pid g c hp hg hc
    exact ⟨rfl,KernelTensor.input_logical t laws before h pid g c hp hg hc⟩
  · intro token ht
    exact ⟨rfl,metadata_logical t laws before h token ht⟩

/-- On the admitted domain, the direct FP32 logical view equals Torch's
negative-view interpretation; the unchanged header rules out that flag. -/
theorem admitted_fp32_view (t : Tensors) (w : Nat → MetadataWord) (h : InputContract (metadata t) w)
    (before : Heap) (r : Role) (i : Nat) :
    (if (t r).metadata.isNeg then (logicalFP32 before (t r) i).neg else logicalFP32 before (t r) i)=
      logicalFP32 before (t r) i := by
  have hn := h.noNeg r
  change (t r).metadata.isNeg=Bool.false at hn
  simp [hn]

#axiomsClean actual
#axiomsClean admitted_fp32_view
#specNonCircular SourceInputs avoiding [Kernel.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelInputBinding
