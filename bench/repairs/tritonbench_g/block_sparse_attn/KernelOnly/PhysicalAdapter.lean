import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Execution
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Logical
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Physical

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API Apart)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
namespace PhysicalKernel

/-- Local memory operations. INT32 loads share the same physical store; their
only law is preservation by a disjoint four-byte output write. -/
structure Access (α Store Byte : Type) extends API α Store Byte where
  read32 : Store → Nat → Word
  read32_apart : ∀ m a b v,Apart a b → read32 (write m a v) b=read32 m b

def integer (access : Access α Store Byte) (buffer : Physical.Buffer) (m : Store) (i : Nat) : Int :=
  if 4*(buffer.storage.offset+i+1)≤buffer.storage.bytes then
    (access.read32 m (Physical.inputAddress buffer i)).toInt else 0

def metadata (access : Access α Store Byte) (buffer : Physical.Buffer) (m : Store) (i : Int) : Int :=
  integer access buffer m i.toNat

structure Program where
  row : Nat
  head : Nat
  channel : Nat
  deriving DecidableEq

def Program.pids (program : Program) : Nat → Nat
  | 0 => program.row
  | 1 => program.head
  | 2 => program.channel
  | _ => 0

def state {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (buffers : Physical.Buffers) (m : Store) (program : Program) : State (Replicas α) where
  mem := fun region address =>
    if region="Q" then .mk .real (broadcast (Physical.read access.toAPI zero (buffers.input .query) m address))
    else if region="K" then .mk .real (broadcast (Physical.read access.toAPI zero (buffers.input .key) m address))
    else if region="V" then .mk .real (broadcast (Physical.read access.toAPI zero (buffers.input .value) m address))
    else if region="rows" then .mk .int (integer access (buffers.input .rows) m address)
    else if region="cols" then .mk .int (integer access (buffers.input .cols) m address)
    else if region="out" then .mk .real (broadcast (Physical.read access.toAPI zero buffers.output m address))
    else .mk .real (broadcast zero)
  regs := fun _ _ _ => none
  pids := program.pids
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem integer_range (access : Access α Store Byte) (buffer : Physical.Buffer) (m : Store) (i : Nat) :
    Int32Bits.inRange (integer access buffer m i)=Bool.true := by
  unfold integer
  split_ifs
  · have lo := BitVec.le_toInt (x := access.read32 m (Physical.inputAddress buffer i))
    have hi := BitVec.toInt_lt (x := access.read32 m (Physical.inputAddress buffer i))
    simp only [Int32Bits.inRange,decide_eq_true_eq]
    exact ⟨lo,hi⟩
  · rfl

theorem state_integers {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (buffers : Physical.Buffers) (m : Store) (program : Program) :
    Int32Memory (state access zero buffers m program) := by
  intro region member address
  simp only [List.mem_cons,List.mem_nil_iff,or_false] at member
  rcases member with rfl|rfl
  all_goals simp only [state,Cell.read_same,ite_true,ite_false]
  all_goals exact integer_range access _ m address

theorem state_views {α : Type} [Inhabited α] (access : Access α Store Byte) (zero : α)
    (buffers : Physical.Buffers) (m : Store) (program : Program) :
    Views (state access zero buffers m program) (Physical.inputs access.toAPI zero buffers m)
      (metadata access (buffers.input .rows) m) (metadata access (buffers.input .cols) m) := by
  constructor
  all_goals intro address
  all_goals simp [state,Physical.inputs,metadata]

end PhysicalKernel
#axiomsClean PhysicalKernel.integer_range
#axiomsClean PhysicalKernel.state_integers
#axiomsClean PhysicalKernel.state_views
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
