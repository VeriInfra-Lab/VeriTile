import bench.repairs.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealWrapper

/-! Complete checked FP32 wrapper, interpreted with Real arithmetic. The
independent target is the inclusive time prefix of the original logical input.
Inactive lanes are explicitly zero in the repaired source. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.OptionalHostCorrect
open VeriTile.Triton VeriTile.Torch
noncomputable section

abbrev logicalPrefix := RealWrapper.formula

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype=.ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

theorem wrapper_correct (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (storage : Storage.Placement h)
    (binding : TensorBinding s h storage) (padding : RealWrapper.Oracle h) (api : Host.API MemCell)
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : RealStorage.Heap) (live : Nat → Prop) (logical : Nat → Nat → Nat → ℝ)
    (ordinaryAPI : Host.PreparationLaws api h storage RealStorage.read logical caller live)
    (allocation : Host.Allocated MemCell) (allocated : api.emptyLike caller=.ok allocation)
    (device : allocation.device=s.device) (launchDevice : api.launchDevice=s.device)
    (setups : allocation.shape.prod≠0 → Host.Setups api h 0 (tried++[selected])) :
    ∃ result, Host.guarded s dtype api (RealWrapper.grid h storage padding) h tried selected caller=
        .ok {allocation with heap:=result} ∧
      Host.metadata h storage {allocation with heap:=result} ∧
      (∀ (row : Fin (h.B*h.H)) (time : Fin h.T) (channel : Fin h.S),
        RealStorage.read (result (Storage.address storage.backing.output
          (h.outputStrides.address h.H row.val time.val channel.val)))=
          logicalPrefix h logical row.val time.val channel.val) ∧
      (∀ addr, live addr → result addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h storage addr → result addr=caller addr) ∧
      (∀ addr, ¬Storage.writes h storage (Grid.programs h) addr → result addr=allocation.heap addr) ∧
      allocation.device=s.device ∧ api.launchDevice=allocation.device := by
  have prepared : Host.prepare api Bool.true caller=.ok allocation := by simp [Host.prepare,allocated]
  obtain ⟨result,returned,metadata,out,liveFrame,frame,outputFrame⟩ :=
    RealWrapper.complete h binding.outputLayout storage padding api Bool.true tried valid selected chosen
      caller live logical ordinaryAPI allocation prepared setups
  exact ⟨result,(Host.guarded_eq s dtype contract api _ h tried selected caller).trans returned,
    metadata,out,liveFrame,frame,outputFrame,device,launchDevice.trans device.symm⟩

/-- Checked CUDA grid dimensions supply both native program-ID bounds. The
source coordinate/address certificates then apply to every owned grid point. -/
def native_pid := TensorBinding.pid
abbrev native_addresses := NativeIndex.source_addresses
abbrev native_coordinates := NativeIndex.source_coordinates
abbrev native_indices := NativeIndex.source_indices

#axiomsClean python_checks_iff_preconditions
#axiomsClean wrapper_correct
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.OptionalHostCorrect
