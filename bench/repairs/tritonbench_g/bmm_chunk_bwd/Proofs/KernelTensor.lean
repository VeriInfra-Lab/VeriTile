import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelFP
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open DirectFP NativeGraph
noncomputable section

/-- Ordinary Tensor/storage binding for the metadata observed by Python. No
execution result, numerical accuracy, or host object identity is assumed. -/
structure Inputs (p : Metadata) where
  tensor : Role → VeriTile.Torch.FP32.Tensor
  laws : ∀ r,Used p r → VeriTile.Torch.FP32.TensorLaws (tensor r)
  metadata : ∀ r,Used p r → (tensor r).metadata=p.tensor r

/-- The output buffer already exists when the kernel starts. These are the
ordinary output buffer API facts, not an allocator or wrapper theorem. -/
structure Output (p : Metadata) (allocation : Direct.Allocation p) where
  tensor : VeriTile.Torch.FP32.Tensor
  laws : VeriTile.Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape=(p.tensor .a).shape
  device : tensor.metadata.device=(p.tensor .a).device
  contiguous : tensor.metadata.isContiguous=Bool.true
  noNeg : tensor.metadata.isNeg=Bool.false
  pointer : tensor.metadata.dataPtr.toNat=allocation.pointer
  /-- Existing supplied-output guard or ordinary fresh allocation API guarantee. -/
  aligned : 0<tensor.metadata.numel → tensor.metadata.dataPtr.toNat%4=0

abbrev readWord (heap : Heap) (address : Nat) : VeriTile.Torch.FP32.Word := ⟨load32 heap address 0⟩

theorem physical_address (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (contiguous : t.metadata.isContiguous=Bool.true)
    (i : Nat) (bound : i<t.metadata.numel) :
    t.metadata.dataPtr.toNat+4*i=t.storage.address (t.physicalIndex i) := by
  have positive : 0<t.metadata.numel := Nat.zero_lt_of_lt bound
  have cap := VeriTile.Torch.FP32.contiguous_capacity t laws contiguous positive
  have span := t.storage.spanFits
  have fit : t.storage.address t.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address]
    omega
  rw [show t.metadata.dataPtr=t.dataPtr from rfl,laws.pointer positive,
    BitVec.toNat_ofNat,Nat.mod_eq_of_lt fit,laws.contiguous contiguous i bound]
  simp only [VeriTile.Torch.FP32.Storage.address]
  omega

theorem direct_word (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (contiguous : t.metadata.isContiguous=Bool.true)
    (noNeg : t.metadata.isNeg=Bool.false) (heap : Heap) (i : Nat) (bound : i<t.metadata.numel) :
    ⟨load32 heap t.metadata.dataPtr.toNat i⟩ =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t laws i := by
  simp only [VeriTile.Torch.FP32.logicalValue,noNeg,Bool.false_eq_true,↓reduceIte,readWord]
  have addr := physical_address t laws contiguous i bound
  apply congrArg (fun w => (⟨w⟩ : VeriTile.Torch.FP32.Word))
  unfold load32
  congr 2
  apply List.map_congr_left
  intro b _
  rw [←addr]
  simp

/-- Every admitted input load is the actual logical Tensor element, including
nonzero storage offsets. Contiguity and no-negative-view checks are used here. -/
theorem input_load (p : Metadata) (h : InputContract p) (inputs : Inputs p)
    (heap : Heap) (r : Role) (used : Used p r) (i : Nat) (bound : i<(p.tensor r).numel) :
    ⟨load32 heap (base p r) i⟩=
      VeriTile.Torch.FP32.logicalValue (readWord heap) (inputs.tensor r) (inputs.laws r used) i := by
  simpa only [base,inputs.metadata r used] using
    direct_word (inputs.tensor r) (inputs.laws r used)
      (by simpa only [inputs.metadata r used] using h.contiguous r used)
      (by simpa only [inputs.metadata r used] using h.noNeg r used)
      heap i (by simpa only [inputs.metadata r used] using bound)

def roleRegion : Role → RegionName
  | .a => "A" | .dout => "Dout" | .residual => "Residual" | .output => "Out"

/-- Original input bytes feed the graph leaves of the actual source. Optional
output is excluded because the output-buffer binding handles that resource. -/
theorem initial_input (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : Inputs p) (heap : Heap) (r : Role) (used : Used p r) (notOut : r≠.output)
    (i : Nat) (bound : i<(p.tensor r).numel) :
    (initial p a heap).mem (roleRegion r) i=.mk .fp32
      (.leaf (VeriTile.Torch.FP32.logicalValue (readWord heap)
        (inputs.tensor r) (inputs.laws r used) i).bits) := by
  have load := input_load p h inputs heap r used i bound
  have word := congrArg VeriTile.Triton.FP.Value.bits load
  cases r with
  | a => simpa [initial,roleRegion,Direct.byteBase] using congrArg (fun w => Cell.mk .fp32 (Expr.leaf w)) word
  | dout => simpa [initial,roleRegion,Direct.byteBase] using congrArg (fun w => Cell.mk .fp32 (Expr.leaf w)) word
  | residual =>
    have present : p.hasResidual=Bool.true := by simpa [Used] using used
    simpa [initial,roleRegion,Direct.byteBase,present] using congrArg (fun w => Cell.mk .fp32 (Expr.leaf w)) word
  | output => exact (notOut rfl).elim

/-- Physical storage coordinate for any complete output tensor coordinate. -/
theorem output_address (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (out : Output p a) (b s g n : Nat)
    (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    out.tensor.storage.address (out.tensor.physicalIndex (Direct.tensorIndex p b s g n))=
      a.pointer+4*Direct.tensorIndex p b s g n := by
  have bound := Direct.tensor_bound p h b s g n hb hs hg hn
  have numel : out.tensor.metadata.numel=(p.tensor .a).numel := by
    simp only [VeriTile.Torch.TensorMetadata.numel,out.shape]
  simpa only [out.pointer] using
    (physical_address out.tensor out.laws out.contiguous _ (by simpa only [numel] using bound)).symm

/-- Active typed output stores satisfy PTX natural alignment. -/
theorem output_aligned (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (out : Output p a) (b s g n : Nat)
    (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    (out.tensor.storage.address (out.tensor.physicalIndex (Direct.tensorIndex p b s g n)))%4=0 := by
  have bound := Direct.tensor_bound p h b s g n hb hs hg hn
  have positive : 0<out.tensor.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,out.shape] using Nat.zero_lt_of_lt bound
  have align := out.aligned positive
  rw [out.pointer] at align
  rw [output_address p h a out b s g n hb hs hg hn]
  simp [Nat.add_mod,align]

/-- Four bytes at the actual output Tensor coordinate agree with the complete
independent recurrence. Pointer arithmetic, output serialization and frame are
separate from the parameterized native arithmetic primitives. -/
theorem output_bytes (rest : Algebra Expr) (arith : Arithmetic) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (out : Output p a) (heap : Heap) (choice : Fin 9)
    (b s g n : Nat) (hb : b<batch p) (hs : s<seqlen p) (hg : g<groups p) (hn : n<width p) :
    ∃ q∈HostFP.grid (call p) (tuning choice),
      ∃ i : TileIndex [(tuning choice).BM,(tuning choice).BN],
        FPReference.active (HostFP.launch (call p) (tuning choice) q (initial p a heap))
          (tuning choice).BM (tuning choice).BN (HostContract.parameters (call p)) i ∧
        FPFlat.outOffset (call p) (tuning choice) q (initial p a heap) i=Direct.tensorIndex p b s g n ∧
        ∀ j : Fin 4,
          outputHeap p a arith (KernelFP.native rest p h a choice (initial p a heap)) heap
            (out.tensor.storage.address (out.tensor.physicalIndex (Direct.tensorIndex p b s g n))+j.val)=
            byte (interpret arith (FPReference.expected (algebra rest) (dot choice)
              (tuning choice).BM (tuning choice).BN (tuning choice).BCS (HostContract.parameters (call p))
              (HostFP.launch (call p) (tuning choice) q
                (FPFlat.lift (call p) (Direct.placement p) (initial p a heap))) i)) j.val := by
  obtain ⟨q,hq,i,hi,offset,read⟩ := KernelFP.outputs rest p h a choice (initial p a heap)
    (initial_good p a heap) b s g n hb hs hg hn
  refine ⟨q,hq,i,hi,offset,?_⟩
  intro j
  rw [output_address p h a out b s g n hb hs hg hn,
    output_byte p a arith _ heap _ (Direct.tensor_bound p h b s g n hb hs hg hn) j]
  unfold graph
  rw [read]
  rfl

#axiomsClean input_load
#axiomsClean initial_input
#axiomsClean output_aligned
#axiomsClean output_address
#axiomsClean output_bytes
end
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelTensor
