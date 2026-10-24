import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Raw
import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.RawLifting

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RawTensor
open VeriTile.Triton FP.Structural
open MatmulDequantInt4.Direct
noncomputable section
set_option maxHeartbeats 4000000

structure Arguments (p : RawInfo) where
  input : Tensor.Actual p.input
  weight : Tensor.Actual p.weight
  scale : Tensor.Actual p.scale
  zero : Tensor.Actual p.zero
  output : Tensor.Actual p.output
  /-- Ordinary allocation identity laws, independent of kernel behavior. -/
  inputStorage : input.storageObject=output.storageObject →
    p.input.tensor.base=p.output.tensor.base ∧ p.input.tensor.storageBytes=p.output.tensor.storageBytes
  weightStorage : weight.storageObject=output.storageObject →
    p.weight.tensor.base=p.output.tensor.base ∧ p.weight.tensor.storageBytes=p.output.tensor.storageBytes
  scaleStorage : scale.storageObject=output.storageObject →
    p.scale.tensor.base=p.output.tensor.base ∧ p.scale.tensor.storageBytes=p.output.tensor.storageBytes
  zeroStorage : zero.storageObject=output.storageObject →
    p.zero.tensor.base=p.output.tensor.base ∧ p.zero.tensor.storageBytes=p.output.tensor.storageBytes

def observation (i : Info) (t : Tensor.Actual i) : MatmulDequantizeInt4.Checked.Observation :=
  ⟨i.tensor.objectId,t.metadata,t.storageObject,i.tensor.base,i.tensor.storageBytes,
    i.tensor.offset,i.tensor.lazyConj⟩

def metadata (p : RawInfo) (t : Arguments p) : MatmulDequantizeInt4.Checked.Metadata :=
  ⟨Raw.call p,observation p.input t.input,observation p.weight t.weight,
    observation p.scale t.scale,observation p.zero t.zero,observation p.output t.output⟩

/-- Adapt actual ordinary Tensor coordinate/storage laws to the existing
mixed-width tensor ABI; no host success or numerical predicate is added. -/
def actual (info : Info) (t : Tensor.Actual info) (dtype : MatmulDequantInt4.Host.DType)
    (rows columns device : Nat) (h : Ordinary info dtype rows columns device) :
    MatmulDequantizeInt4.TensorBinding.Tensor (observation info t) (Raw.view info.tensor) where
  physicalIndex := t.physicalIndex
  coordinates := by intro r n _ _;exact t.coordinates r n
  nonempty := by
    intro nonempty
    have ordinary := h.2.2.2.2.2.2.1
    have capacity := ordinary.2.2.2.2.2.2.2.2.2.2.2.1
    have width : 0<dtype.width := by cases dtype <;> decide
    change 0 < info.tensor.rows*info.tensor.columns at nonempty
    rw [ordinary.1,ordinary.2.1] at nonempty
    change 0 < info.tensor.storageBytes
    nlinarith
  capacity := by
    intro r n hr hn
    have ordinary := h.2.2.2.2.2.2.1
    have rr : r<rows := by simpa [Raw.view,ordinary.1] using hr
    have nn : n<columns := by simpa [Raw.view,ordinary.2.1] using hn
    rw [t.coordinates,MatmulDequantInt4.Host.ordinary_active _ _ _ _ _ _ _ ordinary rr nn]
    have bound := MatmulDequantInt4.rowMajor_bound rows columns r n rr nn
    simpa [observation,Raw.view,ordinary.1,ordinary.2.1] using Nat.add_lt_add_left bound info.tensor.offset

def output (p : RawInfo) (t : Arguments p) (h : Raw.InputContract p) :
    MatmulDequantizeInt4.TensorBinding.Tensor (metadata p t).output (metadata p t).call.C :=
  actual p.output t.output .fp16 (rawShape p).m (rawShape p).n p.input.tensor.device h.2.2.2.2.2.2.2.1

/-- Real dense product and complete allocation frame at actual raw Tensor
coordinates, for all 17 configurations and both retained group branches. -/
theorem real (p : RawInfo) (t : Arguments p) (h : Raw.InputContract p) (i : Fin 17)
    (raw : MatmulDequantizeInt4.TensorBinding.Heap) (heap : MatmulDequantizeInt4.RealTensor.RealHeap) :
    ∃ state,MatmulDequantizeInt4.KernelSource.run MatmulDequantizeInt4.KernelReal.scalar
        MatmulDequantizeInt4.KernelReal.dot (Raw.call p) i
          (MatmulDequantizeInt4.RealTensor.entry (metadata p t) raw heap)=some state ∧
      RawLifting.RealPost (metadata p t) (output p t h) i raw heap
        (MatmulDequantizeInt4.RealTensor.physical (metadata p t) heap state) :=
  RawLifting.real (metadata p t) (output p t h) (Raw.valid p h) i raw heap

/-- Ordered native fused result and every output byte/frame at actual Tensor
coordinates, with only the retained scalar and fixed-K16 primitive ports. -/
theorem native (p : RawInfo) (t : Arguments p) (h : Raw.InputContract p) (i : Fin 17)
    (I : MatmulDequantizeInt4.Native.Instructions)
    (mma : MatmulDequantizeInt4.Opcode → MatmulDequantizeInt4.ISA.MMA16)
    (ports : MatmulDequantizeInt4.Native.Obligations I mma)
    (heap : MatmulDequantizeInt4.TensorBinding.Heap) :
    ∃ state,MatmulDequantizeInt4.Native.grid I (Raw.call p) i
        (MatmulDequantizeInt4.Physical.load (MatmulDequantizeInt4.TensorBinding.memory (metadata p t) heap))=some state ∧
      RawLifting.NativePost (metadata p t) (output p t h) mma i heap
        (MatmulDequantizeInt4.TensorBinding.physical (metadata p t) heap
          (MatmulDequantizeInt4.Physical.commit (MatmulDequantizeInt4.TensorBinding.memory (metadata p t) heap) state)) :=
  RawLifting.native (metadata p t) (output p t h) (Raw.valid p h) I mma ports i heap

/-- Checked interval separation entails preservation of all actual caller
allocations; storage identity laws are observational, not output premises. -/
theorem allocations (p : RawInfo) (t : Arguments p) (h : Raw.InputContract p) :
    t.input.storageObject≠t.output.storageObject ∧ t.weight.storageObject≠t.output.storageObject ∧
    t.scale.storageObject≠t.output.storageObject ∧ t.zero.storageObject≠t.output.storageObject := by
  rcases h with ⟨_,_,_,_,_,_,_,ho,_,_,_,_,_,apart⟩
  have ordinary := ho.2.2.2.2.2.2.1
  have cap := ordinary.2.2.2.2.2.2.2.2.2.2.2.1
  have pos : 0<p.output.tensor.storageBytes := lt_of_lt_of_le
    (Nat.mul_pos (Nat.mul_pos (by decide : 0<MatmulDequantInt4.Host.DType.fp16.width)
      ordinary.2.2.1) ordinary.2.2.2.1) cap
  have separate (info : MatmulDequantInt4.Host.Tensor)
      (member : info∈[p.input.tensor,p.weight.tensor,p.scale.tensor,p.zero.tensor])
      (s : Nat) (same : s=t.output.storageObject →
        info.base=p.output.tensor.base ∧ info.storageBytes=p.output.tensor.storageBytes) :
      s≠t.output.storageObject := by
    intro eq
    obtain ⟨base,bytes⟩ := same eq
    have interval := (List.pairwise_cons.mp apart).1 info member
    unfold MatmulDequantInt4.Host.Disjoint at interval
    rw [base,bytes] at interval
    omega
  exact ⟨separate _ (by simp) _ t.inputStorage,separate _ (by simp) _ t.weightStorage,
    separate _ (by simp) _ t.scaleStorage,separate _ (by simp) _ t.zeroStorage⟩

theorem native_inputs (p : RawInfo) (t : Arguments p) (h : Raw.InputContract p) (i : Fin 17)
    (mma : MatmulDequantizeInt4.Opcode → MatmulDequantizeInt4.ISA.MMA16)
    (initial final : MatmulDequantizeInt4.TensorBinding.Heap)
    (post : RawLifting.NativePost (metadata p t) (output p t h) mma i initial final) :
    ∀ storage∈[t.input.storageObject,t.weight.storageObject,t.scale.storageObject,t.zero.storageObject],
      ∀ byte,final storage byte=initial storage byte := by
  have names := allocations p t h
  intro storage member byte
  have ne : storage≠t.output.storageObject := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl
    · exact names.1
    · exact names.2.1
    · exact names.2.2.1
    · exact names.2.2.2
  exact post.2.2 storage byte (Or.inl ne)

#axiomsClean allocations
#axiomsClean native_inputs
#axiomsClean actual
#axiomsClean real
#axiomsClean native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.RawTensor
