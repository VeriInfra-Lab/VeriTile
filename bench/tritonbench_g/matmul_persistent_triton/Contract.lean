import bench.tritonbench_g.matmul_persistent_triton.Tensor
import bench.tritonbench_g.matmul_persistent_triton.Values

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton

/-- Standard declaration entry for the ordinary public input contract.
These are exactly the metadata, initial backing-memory and launch premises
used by the complete Real/native endpoints. The actual Tensor objects carry
their independent API indexing laws in `TensorBinding.Arguments`.
There is no arithmetic backend, output value, execution, or numerical port
in this contract; scalar/ISA numerical ports belong to `FPEquiv` separately. -/
structure Contract {α : Type} [Inhabited α] (p : Core.Parameters)
    (args : TensorBinding.Arguments) (initial : State α) (entry : Memory α) : Prop where
  layout : Layout p.problem args.descriptors
  strides : Core.Strides p args.descriptors
  inputs : Core.Inputs p args.descriptors initial entry
  outputBinding : ∀ offset,Core.readCell p.problem.profile (initial.mem "C" offset)=
    entry (args.descriptors.C.storage,args.descriptors.C.offset+offset)
  smsPositive : 0<p.problem.SMS

#specNonCircular Contract avoiding [Source.run,Source.step,Source.accumulator]
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
