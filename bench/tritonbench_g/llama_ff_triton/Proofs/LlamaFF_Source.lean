import bench.tritonbench_g.llama_ff_triton.Proofs.SourceDefinitions
import bench.tritonbench_g.llama_ff_triton.Proofs.Numeric

namespace VeriTile.Bench.Repairs.LlamaFF
open VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 5000000

theorem bound_native_complete
    (actual reference : StorageOps (BitVec 32) (BitVec 16))
    (leaves : NumericLeaves actual reference)
    (abi : LocalEventABI (BitVec 32) (BitVec 16) σ actual)
    (p : Parameters (BitVec 32)) (b : Buffers) (sep : b.Separate)
    (injective : Function.Injective (outputAddress b : Cell s → Address))
    (binding : CompilerBinding (BitVec 32) (BitVec 16) s p b) (state : σ) :
    (∀ x : Cell s,abi.observe (boundExecute actual abi binding state).2 (outputAddress b x)=
      wordTarget reference p b selectedPlans selectedReductions (abi.observe state) x) ∧
    (∀ q,(∀ x : Cell s,q≠outputAddress b x) →
      abi.observe (boundExecute actual abi binding state).2 q=abi.observe state q) := by
  unfold boundExecute
  rw [binding.lowered]
  exact native_word_and_frame actual reference leaves abi p b sep injective
    selectedPlans selectedReductions binding.schedule state

theorem binding_unique (binding : CompilerBinding F H s p b) : binding.physicalLanes.Nodup := by
  have h := binding.schedule.unique
  rw [←binding.dispatched] at h
  exact List.Nodup.of_map _ h
theorem binding_covers (binding : CompilerBinding F H s p b) (lane : Lane s) : lane∈binding.physicalLanes := by
  have h := binding.schedule.complete lane.cell
  rw [←binding.dispatched] at h
  obtain ⟨other,mem,equal⟩ := List.mem_map.mp h
  exact (lane_unique equal) ▸ mem

#axiomsClean bound_native_complete
#axiomsClean binding_unique
#axiomsClean binding_covers
end
end VeriTile.Bench.Repairs.LlamaFF
