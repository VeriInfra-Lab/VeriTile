import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.FP
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.TensorBinding

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ReturnedTensor
open VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost WholeNative
noncomputable section

/-- Ordinary successful torch.empty result. It identifies the exact allocated
Tensor and its metadata/storage, without any computed output-value premise. -/
structure Allocation {x : Inputs} {entry : Nat → BitVec 32} {old : Nat → Prop}
    (a : Allocated (BitVec 32) x entry old) where
  tensor : Torch.FP32.Tensor
  laws : Torch.FP32.TensorLaws tensor
  shape : tensor.metadata.shape = [x.rows]
  cuda : tensor.metadata.device.kind = "cuda"
  device : tensor.metadata.device.index = x.device
  contiguous : tensor.metadata.isContiguous = Bool.true
  noNeg : tensor.metadata.isNeg = Bool.false
  /-- Binds the actual fresh Tensor to the word-address allocation model.
  The zero-length model base is unused and may be normalized to zero. -/
  base : (Torch.FP32.dataView tensor laws contiguous).base = 4*a.base

variable {x : Inputs} {entry : Nat → BitVec 32} {old : Nat → Prop}
    {a : Allocated (BitVec 32) x entry old}

def readWords (mem : Nat → BitVec 32) : Nat → Torch.FP32.Word :=
  fun addr => ⟨mem (addr/4)⟩

/-- The Python return expression returns the allocation object itself. -/
def returnAllocated (output : Torch.FP32.Tensor) : HostOutcome (BitVec 32) →
    Option (Torch.FP32.Tensor × (Nat → BitVec 32))
  | .returned out => some (output, out.memory)
  | _ => none

theorem span (allocation : Allocation a) : 4*a.base + 4*x.rows ≤ 2^64 := by
  have h := (Torch.FP32.dataView allocation.tensor allocation.laws allocation.contiguous).spanFits
  rw [allocation.base] at h
  simpa [Torch.FP32.dataView, TensorMetadata.numel, allocation.shape] using h

theorem pointer (allocation : Allocation a) (nonempty : 0 < x.rows) :
    BitVec.ofNat 64 (4*a.base) = allocation.tensor.dataPtr := by
  have hn : 0 < allocation.tensor.metadata.numel := by
    simpa [TensorMetadata.numel, allocation.shape] using nonempty
  have hp := Torch.FP32.actual_pointer allocation.tensor allocation.laws allocation.contiguous hn
  simpa [Torch.FP32.DataView.address, allocation.base] using hp

theorem logical_readback (allocation : Allocation a) (mem : Nat → BitVec 32)
    (i : Nat) (hi : i < x.rows) :
    (Torch.FP32.logicalValue (readWords mem) allocation.tensor allocation.laws i).bits =
      mem (a.base+i) := by
  have hn : i < allocation.tensor.metadata.numel := by
    simpa [TensorMetadata.numel, allocation.shape] using hi
  rw [←Torch.FP32.direct_read_eq_logical (readWords mem) allocation.tensor
    allocation.laws allocation.contiguous allocation.noNeg i hn]
  simp only [readWords, Torch.FP32.DataView.address, allocation.base]
  congr 1
  omega

/-- Transport the proven complete host result to the actual returned Tensor.
There is no second allocation or output-copy step. -/
theorem returns
    (native : Scalars) (graphs : (c : Config) → Nat → Graph c.blockRows c.blockColumns)
    (rt : Runtime (BitVec 32) x entry old)
    (compatible : x.columns = x.vectorLength)
    (allocated : rt.allocate = some a) (ready : rt.launchReady a = Bool.true)
    (allocation : Allocation a)
    {matrix : Nat → Nat → BitVec 32} {vector : Nat → BitVec 32}
    (binding : DirectBinding (BitVec 32) x entry matrix vector) :
    ∃ finalMemory,
      returnAllocated allocation.tensor (WholeNativeHost.execute native graphs rt) =
        some (allocation.tensor, finalMemory) ∧
      (∀ i : Fin x.rows,
        (Torch.FP32.logicalValue (readWords finalMemory) allocation.tensor allocation.laws i.val).bits =
          FP.formula native graphs (rt.plan a).selected.config x matrix vector i.val) ∧
      (∀ addr, old addr → finalMemory addr = entry addr) ∧
      (∀ addr, (∀ i, i < x.rows → addr ≠ a.base+i) → finalMemory addr = a.memory addr) := by
  have result := FP.outcomes native graphs binding compatible rt
  cases execution : WholeNativeHost.execute native graphs rt with
  | incompatibleDimensions =>
    rw [execution] at result
    exact False.elim (result compatible)
  | allocationFailure =>
    rw [execution] at result
    have bad := result.2
    rw [allocated] at bad
    contradiction
  | setupFailure =>
    rw [execution] at result
    rcases result with ⟨_, b, hb, notReady⟩
    have eqn : b = a := Option.some.inj (hb.symm.trans allocated)
    subst b
    rw [ready] at notReady
    contradiction
  | sourceFailure =>
    rw [execution] at result
    exact False.elim result
  | returned out =>
    rw [execution] at result
    rcases result with ⟨_, b, hb, _, _, _, _, _, base, values, exterior, frame⟩
    have eqn : b = a := Option.some.inj (hb.symm.trans allocated)
    subst b
    refine ⟨out.memory, rfl, ?_, frame, ?_⟩
    · intro i
      rw [logical_readback allocation out.memory i.val i.isLt, ←base]
      exact values i
    · simpa only [base] using exterior

#axiomsClean span
#axiomsClean pointer
#axiomsClean logical_readback
#axiomsClean returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ReturnedTensor
