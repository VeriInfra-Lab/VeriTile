import bench.tritonbench_g.block_sparse_attn.TargetTrace
import bench.tritonbench_g.block_sparse_attn.ScalarComplete
import bench.tritonbench_g.block_sparse_attn.Proofs.Native

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
namespace PhysicalKernel

/-- Local scalar primitives, source literals, and the raw runtime FP32 scale
argument. The scale word is independent of the literal-zero operation. -/
structure NativePrimitives where
  instructions : BlockSparseAttn.Native.Instructions
  literal : ℝ → Word
  scale : Word
  zero : literal 0=0
  negativeInfinity : instructions.add 0 0xff800000=0xff800000

def NativePrimitives.arithmetic (ops : NativePrimitives) : Arithmetic Word :=
  Arithmetic.ofLegacy (BlockSparseAttn.Native.ops ops.instructions) ops.literal (fun _ => ops.scale)

theorem native_scalar_argument (ops : NativePrimitives) (p : Parameters)
    (s : State (Replicas Word)) :
    (bindScale (broadcast (ops.arithmetic.argument p.scale)) s).regs .real [] "softmax_scale"=
      some (fun _ => broadcast ops.scale) := by
  simp [bindScale,NativePrimitives.arithmetic,Arithmetic.ofLegacy]

def nativeTarget (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store) (w : Work) : Word :=
  rowValue (layoutOf p batches (metadata access (buffers.input .rows) initial)
    (metadata access (buffers.input .cols) initial)) (BlockSparseAttn.Native.ops ops.instructions)
    (Physical.inputs access.toAPI 0 buffers initial) ops.scale w

theorem native_events (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) :
    gridEvents access ops.arithmetic Bool.true p buffers initial (programs p batches)=
      targetTrace p batches buffers (nativeTarget ops access p batches buffers initial) := by
  apply gridEvents_target access ops.arithmetic Bool.true p batches buffers initial q k v out rows cols abi
  intro id valid i _
  exact output_view (BlockSparseAttn.Native.ops ops.instructions) ops.literal (fun _ => ops.scale) ops.zero ops.negativeInfinity
    p batches (state access 0 buffers initial id) (Physical.inputs access.toAPI 0 buffers initial)
    _ _ (state_views access 0 buffers initial id)
    (first_nonnegative access 0 p batches buffers initial q k v out rows cols abi id valid)
    (column_nonnegative access 0 p batches buffers initial q k v out rows cols abi id valid) i

/-- Whole selected-native kernel execution, exact independent scalar target,
and every caller byte outside the active output windows. No successful-run
or whole-output equality is an input assumption. -/
theorem whole_native (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) :
    ∃ final, grid access ops.arithmetic Bool.true p buffers (programs p batches) initial=some final ∧
      (∀ w,w.active (shape p batches) →
        access.read final (Physical.outputAddress buffers (shape p batches) w)=
          nativeTarget ops access p batches buffers initial w) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  refine ⟨LayernormFwdTriton.Memory.run access.toAPI
    (targetTrace p batches buffers (nativeTarget ops access p batches buffers initial)) initial,?_,?_,?_⟩
  · rw [grid_success access ops.arithmetic Bool.true p batches buffers initial q k v out rows cols abi,
      native_events ops access p batches buffers initial q k v out rows cols abi]
  · exact fun w live => target_readback access.toAPI p batches buffers _ abi.header.shape.2.1 initial w live
  · exact fun address outside => target_frame access.toAPI p batches buffers _ abi.header.shape.2.1 initial address outside

end PhysicalKernel
#axiomsClean PhysicalKernel.native_scalar_argument
#axiomsClean PhysicalKernel.native_events
#axiomsClean PhysicalKernel.whole_native
#specNonCircular PhysicalKernel.nativeTarget avoiding [execution,output,accumulation,PhysicalKernel.grid]
#specNonCircular PhysicalKernel.NativePrimitives avoiding [execution,output,accumulation,rowValue,PhysicalKernel.grid]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
