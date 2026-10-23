import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Misc.GridResults
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeArithmetic
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.LoadSafety

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def resultShape (i : Contract.Inputs) : List Nat :=
  let s := RmsMatmulRbe.Checked.shape i.original
  [s.B,s.M,s.heads,s.headDim]

/-- The ordinary contiguous reshape of an existing output Tensor. The
storage base, extent, offset, pointer, dtype and device are inherited. -/
def resultTensor (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (phase : Fin 3) :
    RmsMatmulRbe.Contract.Tensor :=
  let original := RmsMatmulRbe.Checked.output i.original q phase
  let s := RmsMatmulRbe.Checked.shape i.original
  {original with metadata:={original.metadata with shape:=resultShape i},strides:=[s.M*s.N,s.N,s.headDim,1]}

theorem result_storage (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (phase : Fin 3) :
    (resultTensor i q phase).base=(RmsMatmulRbe.Checked.output i.original q phase).base ∧
    (resultTensor i q phase).bytes=(RmsMatmulRbe.Checked.output i.original q phase).bytes ∧
    (resultTensor i q phase).offset=(RmsMatmulRbe.Checked.output i.original q phase).offset ∧
    (resultTensor i q phase).metadata.dataPtr=(RmsMatmulRbe.Checked.output i.original q phase).metadata.dataPtr ∧
    (resultTensor i q phase).metadata.dtype=(RmsMatmulRbe.Checked.output i.original q phase).metadata.dtype ∧
    (resultTensor i q phase).metadata.device=(RmsMatmulRbe.Checked.output i.original q phase).metadata.device := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem result_count (i : Contract.Inputs) (checked : Contract.InputContract i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (phase : Fin 3) :
    (resultTensor i q phase).metadata.numel=(RmsMatmulRbe.Checked.output i.original q phase).metadata.numel := by
  rw [caller_output_count i checked q hq]
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  simp [resultTensor,resultShape,VeriTile.Torch.TensorMetadata.numel,f.valid.2.2.2.2,Nat.mul_assoc]

def cellAddress (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) : Nat :=
  address (resultTensor i q c.1) (RmsMatmulRbe.Checked.flat3
    (RmsMatmulRbe.Checked.shape i.original).M (RmsMatmulRbe.Checked.shape i.original).N c.2.1 c.2.2.1 c.2.2.2)

theorem commit_cell_address (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (c : Commit) (hc : c.Valid i) :
    cellAddress i q (c.cell i hc)=outputAddress i q c.id c.phase c.lane := by
  simp only [cellAddress,Commit.cell,outputAddress,resultTensor,address,logicalOffset,RmsMatmulRbe.Checked.flat3]
  congr 2
  ring

theorem owner_address (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) :
    outputAddress i q (owner i c).id (owner i c).phase (owner i c).lane=cellAddress i q c := by
  rw [←commit_cell_address i q (owner i c) (owner_valid i c),owner_cell]

theorem head_offset (i : Contract.Inputs) (checked : Contract.InputContract i) (b m h d : Nat) :
    let s := RmsMatmulRbe.Checked.shape i.original
    ((b*s.M+m)*s.heads+h)*s.headDim+d=RmsMatmulRbe.Checked.flat3 s.M s.N b m (h*s.headDim+d) := by
  have f := RmsMatmulRbe.Checked.facts i.original checked.1
  simp only [RmsMatmulRbe.Checked.flat3,f.valid.2.2.2.2]
  ring

def nativeTarget (p : Native.Primitives) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor)
    (initial : Bytes.Store) (epsilon theta : HostScalar)
    (c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original)) : Native.Word :=
  (event Bytes.access (Native.arithmetic p) (Native.materialization p) Bool.true
    i q initial epsilon theta (owner i c)).value

/-- Checked caller metadata and the same host objects drive native execution
and reference. Q/K/V are the existing Tensor backings; returned values are
raw FP16 words or int8-backed E5 bytes, with a complete untouched-byte frame. -/
theorem native_tensor (p : Native.Primitives) (zero : Native.ZeroRealization p)
    (i : Contract.Inputs) (checked : Contract.checkInputs i=.ok ()) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Bytes.Store)
    (epsilon theta : HostScalar) (observed : ScalarObservations i epsilon theta) :
    (epsilonType epsilon=some .float32 ∧ thetaType theta=some .float32) ∧
    (∀ phase,(resultTensor i q phase).metadata.shape=resultShape i ∧
      (resultTensor i q phase).metadata.numel=(RmsMatmulRbe.Checked.output i.original q phase).metadata.numel) ∧
    ∃ final,grid Bytes.access (Native.arithmetic p) (Native.materialization p) Bool.true i q initial epsilon theta=some final ∧
      (∀ c : RmsMatmulRbe.Cell (RmsMatmulRbe.Checked.shape i.original),
        Bytes.access.read (qkvOutputFP8 (useFP8 i) c.1) final (cellAddress i q c)=nativeTarget p i q initial epsilon theta c) ∧
      (∀ j,¬WrittenByte i q j → final j=initial j) ∧
      (∀ t∈i.original.reads,∀ j,t.base≤j → j<t.base+t.bytes → final j=initial j) := by
  have contract := (Contract.checkInputs_iff i).mp checked
  refine ⟨checked_scalar_types i contract epsilon theta observed,
    fun phase => ⟨rfl,result_count i contract q hq phase⟩,?_⟩
  obtain ⟨final,success,readback,frame⟩ := grid_results Bytes.access (Native.arithmetic p) (Native.zero_laws p zero)
    (Native.materialization p) Bool.true i contract ordinary q hq initial epsilon theta (Native.output_storable p)
  refine ⟨final,success,?_,frame,?_⟩
  · intro c
    have result := readback (owner i c) (owner_valid i c)
    rw [owner_address] at result
    exact result
  · intro t ht j lo hi
    exact frame j (input_backing_unwritten i contract ordinary q hq t ht j ⟨lo,hi⟩)

#axiomsClean result_storage
#axiomsClean result_count
#axiomsClean owner_address
#axiomsClean head_offset
#axiomsClean native_tensor
#specNonCircular nativeTarget avoiding [grid,gridAux,program,programTrace,storedEvents]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
