import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Correct
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.FPEquiv
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.KernelEquivalence
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.KernelContract

/-! Tensor applications of the complete raw-memory prefix and native-equivalence
results, with check correspondence and ordinary input representation helpers. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.TensorBridge
open VeriTile VeriTile.Triton VeriTile.Torch FP.Structural FP.ProfiledRewrite
open KernelRealResult KernelEquivalence
noncomputable section
variable {α : Type} [Inhabited α]

theorem python_checks_iff_preconditions (s : ArgumentMetadata) (dtype : Option DType) :
    checkInput s dtype = .ok () ↔ AdmittedArgument s dtype := checkInput_iff s dtype

/-- Ordinary physical input cells bind directly to the mathematical memory;
this is a representation law and contains no desired kernel output. -/
theorem input_memory (c : Layout.Shape) (p : Storage.Placement c)
    (other : MemCell) (heap : Nat → MemCell) (off : Nat) :
    Storage.lift c p other heap p.input off = heap (Storage.address p.backing.input off) :=
  Storage.lift_input c p other heap off

/-- The checked Tensor metadata supplies the native program ID. -/
abbrev native_pid := TensorBinding.pid

theorem tensor_input_memory {β γ : Type} (s : TensorMetadata) (c : Layout.Shape)
    (p : Storage.Placement c) (binding : TensorBinding s c p)
    (other : β) (heap : Nat → β) (read : β → γ) (logical : Nat → Nat → γ)
    (law : InputReadLaw s c heap read logical)
    (row time : Nat) (hr : row<c.B*c.H) (ht : time<c.T) :
    read (Storage.lift c p other heap p.input (row*c.T+time))=logical row time :=
  TensorBinding.logical_input_read s c p binding other heap read logical law row time hr ht

/-- The selected source starts from the Tensor's current physical heap. The
ordinary input read law turns its memory prefix into the logical Tensor prefix. -/
theorem kernel_correct (tensor : TensorMetadata) (shape : Layout.Shape)
    (p : Storage.Placement shape) (binding : TensorBinding tensor shape p)
    (heap : Nat → MemCell) (logical : Nat → Nat → ℝ)
    (reads : InputReadLaw tensor shape heap readReal logical)
    (c : Config) (s : BlockState)
    (memory : s.mem=Storage.lift shape p (MemCell.real 0) heap)
    (rowInside : s.pids 0<shape.B*shape.H) :
    Spec.Real (∃ t, exec (source p.input p.output shape.T c.block).eraseDType.toAlgKernel s=some t ∧
      (∀ i, i<shape.T → t.readMem p.output (s.pids 0*shape.T+i)=
        mathematical (logical (s.pids 0)) shape.T i) ∧
      (∀ r off, (r≠p.output ∨ ∀ i, i<shape.T → off≠s.pids 0*shape.T+i) →
        t.mem r off=s.mem r off)) := by
  obtain ⟨t,executed,values,frame⟩ := KernelRealResult.memory_correct p.input p.output shape.T c s p.distinct
  refine ⟨t,executed,?_,frame⟩
  intro i hi
  rw [values i hi]
  unfold mathematical
  apply Finset.sum_congr rfl
  intro k hk
  have kt := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
  change readReal (s.mem p.input (s.pids 0*shape.T+k))=logical (s.pids 0) k
  rw [memory]
  exact TensorBinding.logical_input_read tensor shape p binding (MemCell.real 0)
    heap readReal logical reads _ _ rowInside kt

/-- Both displayed sources consume the same Tensor-backed typed memory and
preserve its logical input values. Input readback has no numerical guard. -/
theorem chunk_sum_left_equiv (ops : FPRow.Ops α)
    (D : FP.Guarded.Domain α) (model : FP.Guarded.Models [] ops.algebra D)
    (tensor : Torch.TensorMetadata) (shape : Layout.Shape) (p : Storage.Placement shape)
    (binding : TensorBinding tensor shape p) (heap : Nat → Cell α)
    (logical : Nat → Nat → α)
    (reads : InputReadLaw tensor shape heap (Cell.read .fp32) logical)
    (c : Config) (u : State α) (other : Cell α)
    (memory : u.mem=Storage.lift shape p other heap)
    (finite : rewrite_domain ops.algebra D p.input p.output shape.T c.block u) :
    ∃ a b, FP.Structural.exec (engine ops.algebra) (original p.input p.output shape.T c.block) u=some a ∧
      FP.Structural.exec (engine ops.algebra) (sumLeft p.input p.output shape.T c.block) u=some b ∧
      a.mem=b.mem ∧ FPRow.Frame p.output (u.pids 0) shape.T u a ∧
      FPRow.Frame p.output (u.pids 0) shape.T u b ∧
      (∀ row time, row<shape.B*shape.H → time<shape.T →
        (a.mem p.input (row*shape.T+time)).read .fp32=logical row time ∧
        (b.mem p.input (row*shape.T+time)).read .fp32=logical row time) := by
  obtain ⟨a,b,ha,hb,equal,fa,fb⟩ :=
    KernelEquivalence.memory_equiv ops D model p.input p.output shape.T c u finite
  refine ⟨a,b,ha,hb,equal,fa,fb,?_⟩
  intro row time hr ht
  have input := TensorBinding.logical_input_read tensor shape p binding other heap
    (Cell.read .fp32) logical reads row time hr ht
  rw [fa p.input _ (Or.inl (Ne.symm p.distinct)),
    fb p.input _ (Or.inl (Ne.symm p.distinct)),memory]
  exact ⟨input,input⟩

#axiomsClean kernel_correct
#axiomsClean chunk_sum_left_equiv
#axiomsClean python_checks_iff_preconditions
#axiomsClean input_memory
#axiomsClean native_pid
#axiomsClean tensor_input_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.TensorBridge
