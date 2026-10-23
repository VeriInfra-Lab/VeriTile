import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PhysicalBacking
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.PublicCorrectness

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

theorem rawABI_of_abi {α : Type} [Inhabited α] (p : Parameters) (batches : Nat)
    (access : PhysicalKernel.Access α Store Byte) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : PhysicalKernel.ABI p batches access buffers initial q k v out rows cols) :
    PhysicalKernel.RawABI p batches access buffers initial :=
  ⟨abi.header.shape.2.1,PhysicalKernel.abi_stride p batches access buffers initial q k v out rows cols abi,
    abi.parameters,PhysicalKernel.output_window p batches access buffers initial q k v out rows cols abi,abi.separate,
    fun zero => PhysicalKernel.first_nonnegative access zero p batches buffers initial q k v out rows cols abi,
    fun zero => PhysicalKernel.column_nonnegative access zero p batches buffers initial q k v out rows cols abi⟩

/-- Concrete caller memory supplies the finite native input values, without
any value-domain restriction. The original physical stores are identified by
physical_program_trace and retain their complete byte/Tensor theorems. -/
theorem native_physical_inputs (ops : PhysicalKernel.NativePrimitives)
    (access : PhysicalKernel.Access Word Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : PhysicalKernel.ABI p batches access buffers initial q k v out rows cols)
    (id : PhysicalKernel.Program) :
    let visits := visitCapacity access p batches buffers initial
    let xs := snapshot access ops.arithmetic.zero (fun w=>w) p batches visits buffers initial
    Admissible p batches visits xs ∧
    (nativeIO ops p batches visits).InBounds (bufferBounds buffers) xs ∧
    (nativeIO ops p batches visits).InputBinding xs
      (PhysicalKernel.state access ops.arithmetic.zero buffers initial id) := by
  exact ⟨snapshot_admissible access _ _ p batches buffers initial (rawABI_of_abi p batches access buffers initial q k v out rows cols abi),
    physical_in_bounds access _ _ p batches _ buffers initial q k v out rows cols abi _ _ _ _,
    physical_input_binding access _ _ _ _ _ p batches _ buffers initial (fun _ _=>rfl) id⟩

/-- Real replay uses exactly its original real-valued input interpretation;
no new finiteness guard or numerical law is added. -/
theorem real_physical_inputs (access : PhysicalKernel.Access R Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : PhysicalKernel.ABI p batches access buffers initial q k v out rows cols)
    (finite : PhysicalKernel.FiniteInputs access buffers initial)
    (id : PhysicalKernel.Program) :
    let visits := visitCapacity access p batches buffers initial
    let xs := snapshot access (some 0) (WithBot.unbotD 0) p batches visits buffers initial
    Admissible p batches visits xs ∧
    (realIO p batches visits).InBounds (bufferBounds buffers) xs ∧
    (realIO p batches visits).InputBinding xs
      (PhysicalKernel.state access (some 0) buffers initial id) := by
  refine ⟨snapshot_admissible access _ _ p batches buffers initial (rawABI_of_abi p batches access buffers initial q k v out rows cols abi),
    physical_in_bounds access _ _ p batches _ buffers initial q k v out rows cols abi _ _ _ _,
    physical_input_binding access _ _ _ _ _ p batches _ buffers initial ?_ id⟩
  intro role address
  obtain ⟨x,hx⟩ := finite role address
  change some (WithBot.unbotD 0 (Physical.inputs access.toAPI (some 0) buffers initial role address))=
    Physical.inputs access.toAPI (some 0) buffers initial role address
  rw [hx]
  rfl

#axiomsClean rawABI_of_abi
#axiomsClean native_physical_inputs
#axiomsClean real_physical_inputs
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
