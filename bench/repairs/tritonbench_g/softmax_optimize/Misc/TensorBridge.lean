import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelReal
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelNative

/-! Optional input-check correspondence and Tensor binding helpers. The
complete accepted kernel specifications remain in the primary entries. -/
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.TensorBridge
open VeriTile Triton FP.Structural Binding
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native (Term Word Instructions Schedule Reductions)
open NativeOrder (observe)
open KernelRealResults
noncomputable section

theorem python_checks_iff_preconditions (x : TensorMetadata) :
    check x=.ok () ↔ InputContract x := check_iff x

/-- Every valid row executes the actual source, writes its complete mathematical
softmax into physical output Tensor cells and preserves the entire exterior.
The zero-column branch performs no launch. -/
theorem softmax_kernel_correct (x y : Object) (h : InputContract x.metadata)
    (allocated : Allocated x.metadata y.metadata)
    (fresh : x.region≠y.region) (read : Nat → ℝ) (s : BlockState)
    (input : MemoryBinding BlockState.readMem read x s)
    (row : Fin (rows x.metadata)) (pid : s.pid=row.val) :
    Spec.Real (∃ t,KernelReal.execute x y s=some t ∧
      (∀ column : Fin (columns x.metadata),
        t.readMem y.region (y.tensor.physicalIndex (row.val*columns x.metadata+column.val))=
          reference (logicalValue Neg.neg read x) row column) ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i → t.mem r i=s.mem r i) ∧
      (∀ i,t.mem x.region i=s.mem x.region i)) := by
  have output := allocated_contract x.metadata y.metadata h allocated
  have shape := allocated.1
  obtain ⟨t,execution,values,frame⟩ := KernelReal.complete_all x y h read s input row pid
  refine ⟨t,execution,?_,frame,fun i => frame x.region i (Or.inl fresh)⟩
  intro column
  have bound : row.val*columns x.metadata+column.val<y.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel,shape] using flat_lt x h row column
  rw [y.laws.contiguous output.2.2.2.2.1 _ bound]
  simpa only [reference,Nat.add_assoc] using values column

/-- The candidate commutes precisely the four integer row products. Both
complete typed executions succeed from the original Tensor's storage-bound
leaves; every raw cell and every observed FP32 output word agrees. The native
reduction plans, operand order and FMA schedule are retained in the common
primitive model. Empty columns perform no launch. -/
theorem softmax_kernel_fp32 (R : Reductions) (I : Instructions) (S : Schedule) (O : NativeOrder.AddOrder)
    (x y : Object) (h : InputContract x.metadata)
    (allocated : Allocated x.metadata y.metadata)
    (fresh : x.region≠y.region) (read : Nat → Word)
    (row : Fin (rows x.metadata)) (s : State Term) (pid : s.pids 0=row.val) :
    ∃ source candidate,
      KernelNative.execute Bool.false R I S O x y read s=some source ∧
      KernelNative.execute Bool.true R I S O x y read s=some candidate ∧
      source=candidate ∧
      observe I S O source=observe I S O candidate ∧
      (∀ column : Fin (columns x.metadata),
        y.tensor.physicalIndex (row.val*columns x.metadata+column.val)=
          y.tensor.storageOffset+row.val*columns x.metadata+column.val ∧
        observe I S O source y.region (y.tensor.physicalIndex (row.val*columns x.metadata+column.val))=
          observe I S O candidate y.region (y.tensor.physicalIndex (row.val*columns x.metadata+column.val))) ∧
      (∀ r i,r≠y.region ∨ i<y.tensor.storageOffset+row.val*columns x.metadata ∨
        y.tensor.storageOffset+(row.val+1)*columns x.metadata≤ i →
        source.mem r i=(KernelNative.seed x read s).mem r i ∧
        candidate.mem r i=(KernelNative.seed x read s).mem r i) ∧
      (∀ i,source.mem x.region i=(KernelNative.seed x read s).mem x.region i) := by
  have output := allocated_contract x.metadata y.metadata h allocated
  have shape := allocated.1
  obtain ⟨t,source,candidate,frame⟩ := KernelNative.complete_all R I S O x y h read row s pid
  refine ⟨t,t,source,candidate,rfl,rfl,?_,?_,fun i => frame x.region i (Or.inl fresh)⟩
  · intro column
    have bound : row.val*columns x.metadata+column.val<y.metadata.numel := by
      simpa only [VeriTile.Torch.TensorMetadata.numel,shape] using flat_lt x h row column
    exact ⟨by simpa only [Nat.add_assoc] using y.laws.contiguous output.2.2.2.2.1 _ bound,rfl⟩
  · intro r i outside
    exact ⟨frame r i outside,frame r i outside⟩

#axiomsClean softmax_kernel_correct
#axiomsClean softmax_kernel_fp32
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.TensorBridge
