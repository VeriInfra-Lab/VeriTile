import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelGrid
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelGeometry
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open KernelTensor KernelGrid
noncomputable section
set_option maxHeartbeats 4000000

theorem coordinate_launch (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (content : contentActive c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation)
    (hh : head < rows c.input.observation) (hn : n < length c metadataRead b) :
    VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation := by
  have bound := KernelAdmission.bounds c metadataRead contract content b hb
  exact ⟨content.1,content.2.1,content.2.2,by change 0 < c.maximum; omega⟩

theorem output_physical (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (content : contentActive c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation)
    (hh : head < rows c.input.observation) (hn : n < length c metadataRead b) :
    outputAddress c metadataRead b head n=
      c.output.tensor.storage.address (c.output.tensor.physicalIndex
        (head*cols c.output.observation+start c metadataRead b+n)) := by
  rcases contract.metadata.tensors with ⟨_,_,_,_,_,_,_,rank,shape,_,_,_⟩
  change c.output.tensor.metadata.shape=c.input.tensor.metadata.shape at shape
  have outRank : c.output.tensor.metadata.shape.length=2 := by rw [shape]; exact rank
  have rowsEq : rows c.output.observation=rows c.input.observation := congrArg (fun s : List Nat => s.headD 0) shape
  have colsEq : cols c.output.observation=cols c.input.observation := congrArg (fun s : List Nat => s[1]?.getD 0) shape
  have bound := KernelAdmission.bounds c metadataRead contract content b hb
  have token : start c metadataRead b+n < cols c.output.observation := by rw [colsEq];omega
  have pointer := payload_address c.output outRank head (start c metadataRead b+n) (by rwa [rowsEq]) token
  simpa [outputAddress,Nat.add_assoc] using pointer

theorem output_storage_bounds (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (content : contentActive c.observation)
    (b head n : Nat) (hb : b < rows c.start.observation)
    (hh : head < rows c.input.observation) (hn : n < length c metadataRead b) :
    c.output.tensor.storage.base ≤ outputAddress c metadataRead b head n ∧
      outputAddress c metadataRead b head n+4 ≤ c.output.tensor.storage.base+4*c.output.tensor.storage.elements := by
  rcases contract.metadata.tensors with ⟨_,_,_,_,_,_,_,rank,shape,_,_,_⟩
  change c.output.tensor.metadata.shape=c.input.tensor.metadata.shape at shape
  have outRank : c.output.tensor.metadata.shape.length=2 := by rw [shape]; exact rank
  have rowsEq : rows c.output.observation=rows c.input.observation := congrArg (fun s : List Nat => s.headD 0) shape
  have colsEq : cols c.output.observation=cols c.input.observation := congrArg (fun s : List Nat => s[1]?.getD 0) shape
  have bound := KernelAdmission.bounds c metadataRead contract content b hb
  have token : start c metadataRead b+n < cols c.output.observation := by rw [colsEq];omega
  have flat := VeriTile.Bench.Repairs.SquareMatrix.flat_lt (rows c.output.observation)
    (cols c.output.observation) head (start c metadataRead b+n) (by rwa [rowsEq]) token
  have inShape : head*cols c.output.observation+start c metadataRead b+n < c.output.tensor.metadata.numel := by
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [shape_two _ outRank]
    simpa [rows,cols,Payload.observation,Nat.add_assoc] using flat
  have room := c.output.laws.inBounds _ inShape
  rw [output_physical c metadataRead contract content b head n hb hh hn]
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

theorem read_exterior (c : TensorCall) (metadataRead : Nat → Word)
    (contract : InputContract c.observation (observedStart c metadataRead) (observedLength c metadataRead))
    (launch : VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked.active c.observation)
    (t : Observation) (readTensor : t∈[c.input.observation,c.start.observation,c.length.observation])
    (i : Nat) (inside : i < t.storage.elements) : Exterior c metadataRead (t.storage.address i) := by
  intro b hb head hh n hn equal
  have content : contentActive c.observation := ⟨launch.1,launch.2.1,launch.2.2.1⟩
  have bounds := output_storage_bounds c metadataRead contract content b head n hb hh hn
  rcases contract.metadata.launch launch with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,apart⟩
  have separate := apart t readTensor
  rw [←equal] at bounds
  simp only [VeriTile.Torch.FP32.Storage.address,TensorCall.observation,Payload.observation] at bounds separate
  rcases separate with l|r <;> omega

#axiomsClean coordinate_launch
#axiomsClean output_physical
#axiomsClean output_storage_bounds
#axiomsClean read_exterior
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.KernelGeometry
