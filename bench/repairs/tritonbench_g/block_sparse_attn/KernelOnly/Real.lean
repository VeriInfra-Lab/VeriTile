import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.TargetTrace
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.RealComplete

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
namespace PhysicalKernel

/-- The exact-real replay starts with real-valued floating inputs. INT32 CSR
reads still use the same physical store and ordinary ABI guards. -/
def FiniteInputs (access : Access R Store Byte) (buffers : Physical.Buffers) (initial : Store) : Prop :=
  ∀ role i,∃ x : ℝ,Physical.inputs access.toAPI (some 0) buffers initial role i=some x

def realMemory (access : Access R Store Byte) (buffers : Physical.Buffers) (initial : Store) :
    BlockSparseAttn.Memory ℝ :=
  fun role i => WithBot.unbotD 0 (Physical.inputs access.toAPI (some 0) buffers initial role i)

theorem realMemory_lift (access : Access R Store Byte) (buffers : Physical.Buffers) (initial : Store)
    (finite : FiniteInputs access buffers initial) :
    Physical.inputs access.toAPI (some 0) buffers initial=BlockSparseAttn.Real.lift (realMemory access buffers initial) := by
  funext role i
  obtain ⟨x,hx⟩ := finite role i
  simp only [BlockSparseAttn.Real.lift,realMemory,hx]
  rfl

def realTarget (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (w : Work) : ℝ :=
  let x := layoutOf p batches (metadata access (buffers.input .rows) initial)
    (metadata access (buffers.input .cols) initial)
  Spec.output x.shape x.csr (BlockSparseAttn.Real.inputs x (realMemory access buffers initial))
    p.scale w.batch w.head w.row w.col

theorem real_events (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols)
    (finite : FiniteInputs access buffers initial) :
    gridEvents access mathArithmetic Bool.false p buffers initial (programs p batches)=
      targetTrace p batches buffers (fun w => some (realTarget access p batches buffers initial w)) := by
  apply gridEvents_target access mathArithmetic Bool.false p batches buffers initial q k v out rows cols abi
  intro id valid i live
  have views := state_views access (some 0) buffers initial id
  rw [realMemory_lift access buffers initial finite] at views
  exact output_real p batches (state access (some 0) buffers initial id)
    (realMemory access buffers initial) _ _ views
    (first_nonnegative access (some 0) p batches buffers initial q k v out rows cols abi id valid)
    (column_nonnegative access (some 0) p batches buffers initial q k v out rows cols abi id valid) i live

/-- Successful full shared-quote replay gives the independent causal CSR
occurrence-multiset attention quotient over every channel and output cell. -/
theorem whole_real (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store) (q k v out : Host.Tensor) (rows cols : Nat)
    (abi : ABI p batches access buffers initial q k v out rows cols)
    (finite : FiniteInputs access buffers initial) :
    ∃ final, grid access mathArithmetic Bool.false p buffers (programs p batches) initial=some final ∧
      (∀ w,w.active (shape p batches) →
        access.read final (Physical.outputAddress buffers (shape p batches) w)=
          some (realTarget access p batches buffers initial w)) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  refine ⟨LayernormFwdTriton.Memory.run access.toAPI
    (targetTrace p batches buffers (fun w => some (realTarget access p batches buffers initial w))) initial,?_,?_,?_⟩
  · rw [grid_success access mathArithmetic Bool.false p batches buffers initial q k v out rows cols abi,
      real_events access p batches buffers initial q k v out rows cols abi finite]
    rfl
  · exact fun w live => target_readback access.toAPI p batches buffers _ abi.header.shape.2.1 initial w live
  · exact fun address outside => target_frame access.toAPI p batches buffers _ abi.header.shape.2.1 initial address outside

end PhysicalKernel
#axiomsClean PhysicalKernel.realMemory_lift
#axiomsClean PhysicalKernel.real_events
#axiomsClean PhysicalKernel.whole_real
#specNonCircular PhysicalKernel.realTarget avoiding [execution,output,accumulation,PhysicalKernel.grid,rowValue]
#specNonCircular PhysicalKernel.FiniteInputs avoiding [execution,output,accumulation,PhysicalKernel.grid,Spec.output]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
