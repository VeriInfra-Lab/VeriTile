import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.KernelContract
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.DirectHost

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Triton VeriTile.Torch

namespace DirectHost
noncomputable def guarded {S β : Type} (s : TensorMetadata) (dtype : Option DType)
    (c : Layout.Shape) (p : Storage.Placement c)
    (src : Host.Source c S β) (entry : Heap β → S) (memory : S → RegionName → Nat → β)
    (api : API β) (trials : List Config) (selected : Config) (caller : Heap β) :=
  run c p src entry memory api (decide (checkTensorInput s dtype=.ok ())) trials selected caller

theorem guarded_eq {S β : Type} (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (c : Layout.Shape) (p : Storage.Placement c)
    (src : Host.Source c S β) (entry : Heap β → S) (memory : S → RegionName → Nat → β)
    (api : API β) (trials : List Config) (selected : Config) (caller : Heap β) :
    guarded s dtype c p src entry memory api trials selected caller=
      run c p src entry memory api Bool.true trials selected caller := by
  simp [guarded,(checkTensorInput_iff s dtype).mpr contract]

theorem guarded_reject {S β : Type} (s : TensorMetadata) (dtype : Option DType)
    (rejected : ¬InputContract s dtype) (c : Layout.Shape) (p : Storage.Placement c)
    (src : Host.Source c S β) (entry : Heap β → S) (memory : S → RegionName → Nat → β)
    (api : API β) (trials : List Config) (selected : Config) (caller : Heap β) :
    guarded s dtype c p src entry memory api trials selected caller=.error "unsupported input" := by
  have bad : checkTensorInput s dtype≠.ok () := fun h => rejected ((checkTensorInput_iff s dtype).mp h)
  simp [guarded,bad,run]
end DirectHost


end VeriTile.Bench.Repairs.ChunkCumsumKernel
