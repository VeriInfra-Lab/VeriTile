import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.KernelContract
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Host

namespace VeriTile.Bench.Repairs.ReversedCumsum
open VeriTile.Triton VeriTile.Torch

namespace Host
def guarded {β : Type} (s : TensorMetadata) (dtype : Option DType)
    (api : API β) (grid : Config → Heap β → Option (Heap β))
    (h : Call) (tried : List Config) (selected : Config) (caller : Heap β) :=
  run api grid h (decide (checkTensorInput s dtype = .ok ())) tried selected caller

theorem guarded_eq {β : Type} (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (api : API β)
    (grid : Config → Heap β → Option (Heap β))
    (h : Call) (tried : List Config) (selected : Config) (caller : Heap β) :
    guarded s dtype api grid h tried selected caller = run api grid h Bool.true tried selected caller := by
  simp [guarded,(checkTensorInput_iff s dtype).mpr contract]

theorem guarded_reject {β : Type} (s : TensorMetadata) (dtype : Option DType)
    (rejected : ¬ InputContract s dtype) (api : API β)
    (grid : Config → Heap β → Option (Heap β))
    (h : Call) (tried : List Config) (selected : Config) (caller : Heap β) :
    guarded s dtype api grid h tried selected caller =
      .error ⟨.admission,"unsupported input",caller⟩ := by
  have bad : checkTensorInput s dtype ≠ .ok () := fun ok => rejected ((checkTensorInput_iff s dtype).mp ok)
  simp [guarded,bad,run,prepare,ReversedCumsumScalar.Host.prepare]
end Host

end VeriTile.Bench.Repairs.ReversedCumsum
