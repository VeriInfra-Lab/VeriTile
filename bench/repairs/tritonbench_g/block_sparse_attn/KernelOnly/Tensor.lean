import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Native
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Real

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
namespace PhysicalKernel

def tensorAddress (buffers : Physical.Buffers) (out : Host.Tensor) (w : Work) : Nat :=
  buffers.output.base+4*(out.storage.offset+address out.strides w.batch w.head w.row w.col)

theorem tensor_pointer (access : Access α Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) (w : Work) :
    tensorAddress buffers out w=Physical.outputAddress buffers (shape p batches) w := by
  unfold tensorAddress Physical.outputAddress Physical.inputAddress
  rw [abi.outputStrides,Host.output_address,abi.outputStorage]
  rfl

theorem tensor_active (access : Access α Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) (w : Work)
    (live : w.batch<out.batches ∧ w.head<out.heads ∧ w.row<out.rows ∧ w.col<out.dim) :
    w.active (shape p batches) := by
  simpa only [abi.outputShape.1,abi.outputShape.2.1,abi.outputShape.2.2.1,
    abi.outputShape.2.2.2] using live

/-- Read the caller's actual existing tensor using its descriptor, including
its nonzero storage offset. The full quote/native execution is in the result. -/
theorem whole_tensor (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) :
    ∃ final, grid access ops.arithmetic Bool.true p buffers (programs p batches) initial=some final ∧
      (∀ w,w.batch<out.batches → w.head<out.heads → w.row<out.rows → w.col<out.dim →
        access.read final (tensorAddress buffers out w)=nativeTarget ops access p batches buffers initial w) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  obtain ⟨final,success,values,frame⟩ := whole_native ops access p batches buffers initial q k v out rows cols abi
  refine ⟨final,success,?_,frame⟩
  intro w hb hh hr hc
  rw [tensor_pointer access p batches buffers initial q k v out rows cols abi w]
  exact values w (tensor_active access p batches buffers initial q k v out rows cols abi w ⟨hb,hh,hr,hc⟩)

/-- Each of the four actual destination bytes is the corresponding byte of
the independent native word. Codec laws concern one local FP32 memory cell. -/
theorem whole_bytes (ops : NativePrimitives) (access : Access Word Store (BitVec 8))
    (codec : Physical.WordCodec access.toAPI) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols) :
    ∃ final, grid access ops.arithmetic Bool.true p buffers (programs p batches) initial=some final ∧
      (∀ w,w.active (shape p batches) → ∀ j : Fin 4,
        access.byte final (tensorAddress buffers out w+j.val)=
          Physical.wordByte (nativeTarget ops access p batches buffers initial w) j) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  obtain ⟨final,success,values,frame⟩ := whole_native ops access p batches buffers initial q k v out rows cols abi
  refine ⟨final,success,?_,frame⟩
  intro w live j
  rw [tensor_pointer access p batches buffers initial q k v out rows cols abi w,
    codec.bytes,values w live]

theorem whole_real_tensor (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols)
    (finite : FiniteInputs access buffers initial) :
    ∃ final, grid access mathArithmetic Bool.false p buffers (programs p batches) initial=some final ∧
      (∀ w,w.batch<out.batches → w.head<out.heads → w.row<out.rows → w.col<out.dim →
        access.read final (tensorAddress buffers out w)=some (realTarget access p batches buffers initial w)) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  obtain ⟨final,success,values,frame⟩ := whole_real access p batches buffers initial q k v out rows cols abi finite
  refine ⟨final,success,?_,frame⟩
  intro w hb hh hr hc
  rw [tensor_pointer access p batches buffers initial q k v out rows cols abi w]
  exact values w (tensor_active access p batches buffers initial q k v out rows cols abi w ⟨hb,hh,hr,hc⟩)

end PhysicalKernel
#axiomsClean PhysicalKernel.tensor_pointer
#axiomsClean PhysicalKernel.tensor_active
#axiomsClean PhysicalKernel.whole_tensor
#axiomsClean PhysicalKernel.whole_bytes
#axiomsClean PhysicalKernel.whole_real_tensor
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
