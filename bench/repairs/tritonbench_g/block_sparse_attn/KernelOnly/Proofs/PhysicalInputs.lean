import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.IOExecution
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Raw

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
variable {α β Store Byte : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

/-- Current physical metadata, with unused floating placeholders. This view
only computes the finite input addresses; it never evaluates a kernel. -/
def metadataInputs (access : PhysicalKernel.Access β Store Byte) (zero : α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store) :
    Inputs α p batches visits := fun
  | ⟨0,_⟩ => fun j=>PhysicalKernel.integer access (buffers.input .rows) m
      (layout p j.2.1.val*p.rowStride+j.1.val+j.2.2.1.val)
  | ⟨1,_⟩ => fun j=>PhysicalKernel.integer access (buffers.input .cols) m
      ((layout p j.2.1.val*p.colStride : Nat)+
        PhysicalKernel.integer access (buffers.input .rows) m (layout p j.2.1.val*p.rowStride+j.1.val)+
        j.2.2.1.val : Int).toNat
  | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun _=>zero

/-- Snapshot of exactly the current source-address reads. The physical
adapter's historical totalization remains unchanged; backing is a separate
property, not a hidden restriction on the old raw theorem. -/
def snapshot (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store) :
    Inputs α p batches visits := fun
  | ⟨0,_⟩ => metadataInputs access (decode zero) p batches visits buffers m 0
  | ⟨1,_⟩ => metadataInputs access (decode zero) p batches visits buffers m 1
  | ⟨2,_⟩ => fun j=>decode (Physical.read access.toAPI zero (buffers.input .query) m
      (inputOffset p batches visits 2 (metadataInputs access (decode zero) p batches visits buffers m) j))
  | ⟨3,_⟩ => fun j=>decode (Physical.read access.toAPI zero (buffers.input .key) m
      (inputOffset p batches visits 3 (metadataInputs access (decode zero) p batches visits buffers m) j))
  | ⟨4,_⟩ => fun j=>decode (Physical.read access.toAPI zero (buffers.input .value) m
      (inputOffset p batches visits 4 (metadataInputs access (decode zero) p batches visits buffers m) j))

theorem snapshot_offsets (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store)
    (port : Fin 5) :
    inputOffset p batches visits port (snapshot access zero decode p batches visits buffers m)=
      inputOffset p batches visits port (metadataInputs access (decode zero) p batches visits buffers m) := by
  rcases port with ⟨port,limit⟩
  interval_cases port <;> rfl

theorem physical_input_binding (access : PhysicalKernel.Access β Store Byte)
    (a : Arithmetic β) (native : Bool) (decode : β→α) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (buffers : Physical.Buffers) (m : Store)
    (law : ∀ role address,encode (decode (Physical.inputs access.toAPI a.zero buffers m role address))=
      Physical.inputs access.toAPI a.zero buffers m role address)
    (id : PhysicalKernel.Program) :
    (io a native encode observe p batches visits).InputBinding
      (snapshot access a.zero decode p batches visits buffers m)
      (PhysicalKernel.state access a.zero buffers m id) := by
  intro port j _
  rcases port with ⟨port,limit⟩
  change port<5 at limit
  interval_cases port
  · rfl
  · rfl
  · exact congrArg broadcast (law .q _).symm
  · exact congrArg broadcast (law .k _).symm
  · exact congrArg broadcast (law .v _).symm

/-- A finite capacity exists for every old signed-INT32 input, including
reversed endpoint intervals and repeated columns. It is a size parameter,
not an additional restriction on the original physical input domain. -/
def visitCapacity (access : PhysicalKernel.Access β Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store) : Nat :=
  ∑ r : Row p,∑ h : Head p batches,
    (PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val+1)-
      PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val)).toNat

theorem snapshot_capacity (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store)
    (r : Row p) (h : Head p batches) :
    count (snapshot access zero decode p batches (visitCapacity access p batches buffers m) buffers m) r h≤
      visitCapacity access p batches buffers m := by
  let f := fun (r : Row p) (h : Head p batches)=>
    (PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val+1)-
      PhysicalKernel.integer access (buffers.input .rows) m (layout p h.val*p.rowStride+r.val)).toNat
  change f r h≤∑ r,∑ h,f r h
  have inner : f r h≤∑ h,f r h := Finset.single_le_sum (fun _ _=>Nat.zero_le _) (Finset.mem_univ h)
  have outer : (∑ h,f r h)≤∑ r,∑ h,f r h :=
    Finset.single_le_sum (f := fun r : Row p => ∑ h : Head p batches,f r h)
      (fun _ _=>Nat.zero_le _) (Finset.mem_univ r)
  exact le_trans inner outer

theorem snapshot_admissible (access : PhysicalKernel.Access β Store Byte) (zero : β) (decode : β→α)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (m : Store)
    (abi : PhysicalKernel.RawABI p batches access buffers m) :
    Admissible p batches (visitCapacity access p batches buffers m)
      (snapshot access zero decode p batches (visitCapacity access p batches buffers m) buffers m) := by
  intro r h blocks
  let id : PhysicalKernel.Program := ⟨r.val,h.val,0⟩
  have valid : id.Valid p batches := ⟨r.isLt,h.isLt,blocks⟩
  refine ⟨PhysicalKernel.integer_range access _ _ _,PhysicalKernel.integer_range access _ _ _,?_,
    snapshot_capacity access zero decode p batches buffers m r h,?_⟩
  · simpa [KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.state,id,PhysicalKernel.Program.pids,first,snapshot,metadataInputs,layout,head] using
      abi.first zero id valid
  · intro v live
    refine ⟨PhysicalKernel.integer_range access _ _ _,?_⟩
    have active : v.val<visitCount p (PhysicalKernel.state access zero buffers m id) := by
      simpa [visitCount,KernelOnly.first,KernelOnly.last,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
        PhysicalKernel.state,id,PhysicalKernel.Program.pids,count,first,last,snapshot,metadataInputs,layout,head] using live
    simpa [csrColumn,KernelOnly.first,KernelOnly.rowPointer,KernelOnly.layout,KernelOnly.head,
      PhysicalKernel.state,id,PhysicalKernel.Program.pids,column,first,snapshot,metadataInputs,layout,head,Int.add_assoc] using
      abi.columns zero id valid v.val active

/-- The physical packet materializes the actual stores from precisely this
shared invocation. This identifies the existing byte/Tensor transport with
the execution foundation used by public IO, without changing its old scope. -/
theorem physical_program_trace (access : PhysicalKernel.Access β Store Byte)
    (a : Arithmetic β) (native : Bool) (p : Parameters) (buffers : Physical.Buffers)
    (m : Store) (id : PhysicalKernel.Program) :
    PhysicalKernel.programTrace access a native p buffers m id=
      ((invocation a native p id).run (PhysicalKernel.state access a.zero buffers m id)).map
        (PhysicalKernel.storedEvents p buffers (PhysicalKernel.state access a.zero buffers m id)) := by
  rw [invocation_run]
  rfl

#axiomsClean physical_input_binding
#axiomsClean snapshot_admissible
#axiomsClean physical_program_trace
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
