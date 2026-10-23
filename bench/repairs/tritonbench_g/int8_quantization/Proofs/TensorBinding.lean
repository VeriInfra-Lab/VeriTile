import bench.repairs.tritonbench_g.int8_quantization.Proofs.DirectResults

namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section

/-- Exact little-endian assembly, preserving sign, infinities and NaN payloads.
This is byte serialization, not an arbitrary numerical decoding function. -/
def fp32Word (word : Word) : FP.Value .fp32 :=
  ⟨(word 3 ++ word 2 ++ word 1 ++ word 0)⟩

def tensorRead (memory : Memory) (address : Nat) : FP.Value .fp32 :=
  fp32Word (Bytes.read32 memory address)

/-- The original Tensor's logical value is its actual backing-storage word.
The negative-view flag is excluded by the exact checked wrapper contract. -/
theorem input_tensor_logical (c : Request) (memory : Memory) (i : Nat) :
    fp32Word (inputValue c memory i)=
      VeriTile.Torch.FP32.logicalValue (tensorRead memory) c.x c.laws i := by
  simp [VeriTile.Torch.FP32.logicalValue,c.contract.noNeg,tensorRead,inputValue]

/-- Initial typed source loads and logical Tensor inputs share the same bytes;
Tensor storage offsets and the observed data pointer are retained. -/
theorem bound_load_tensor_logical (c : Request) (before : Heap) (out : Allocation c before)
    (i : Nat) (valid : i < c.x.metadata.numel) :
    fp32Word (((Bytes.bind (layout c out) before.memory).mem xRegion i).read .real)=
      VeriTile.Torch.FP32.logicalValue (tensorRead before.memory) c.x c.laws i := by
  change fp32Word (Bytes.read32 before.memory ((view c).base+4*i))=_
  rw [input_pointer c before.memory i valid]
  exact input_tensor_logical c before.memory i

/-- The same Tensor API pointer checked for alignment is used by native loads. -/
theorem input_actual_pointer (c : Request) (nonempty : 0 < c.x.metadata.numel) :
    BitVec.ofNat 64 (view c).base=c.x.metadata.dataPtr := by
  simpa [VeriTile.Torch.FP32.DataView.address] using
    VeriTile.Torch.FP32.actual_pointer c.x c.laws c.contract.contiguous nonempty

def Outside (c : Request) {before : Heap} (out : Allocation c before) (a : Nat) : Prop :=
  ¬Bytes.outputWindow (p c) (layout c out) a ∧
  ¬Bytes.scaleWindow (p c) (layout c out) a

/-- Full memory frame, including unallocated caller bytes. -/
theorem after_frame (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (out : Allocation c before) (a : Nat) (outside : Outside c out a) :
    (after c M ops before out).memory a=before.memory a :=
  Bytes.store_frame _ _ _ _ _ outside.1 outside.2

theorem pair_frame (q k : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (qa : Allocation q before) (ka : Allocation k (after q M ops before qa))
    (a : Nat) (outsideQ : Outside q qa a) (outsideK : Outside k ka a) :
    (after k M ops (after q M ops before qa) ka).memory a=before.memory a :=
  (after_frame k M ops _ ka a outsideK).trans (after_frame q M ops before qa a outsideQ)

#axiomsClean input_tensor_logical
#axiomsClean bound_load_tensor_logical
#axiomsClean input_actual_pointer
#axiomsClean pair_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.Direct
