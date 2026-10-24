import bench.repairs.tritonbench_g.layernorm_fwd_triton.Correct
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelBoundary
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Contract

/-! Optional checked-input correspondence and application to the actual byte
heap. Primary complete typed-kernel results remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.TensorBridge
open VeriTile.Triton FP.Structural Address InputBinding
noncomputable section
variable {shape : Shape} {st : Strides} {eps : Contract.Epsilon}

theorem python_checks_iff_preconditions (x w : VeriTile.Torch.ArgumentMetadata) (eps : Contract.Epsilon) :
    Contract.checkArguments x w eps=.ok () ↔ Contract.AdmittedArguments x w eps :=
  Contract.checkArguments_iff x w eps

/-- Successful complete typed kernel grid, every logical output from original
Tensor storage, and the full exterior memory frame. Real inputs are arbitrary
and independent of native words; no host execution or FP-to-Real claim occurs. -/
theorem layernorm_fwd_triton_real (a : Arguments shape st eps)
    (memory : Nat → ℝ) (scratch : State ℝ) :
    ∃ final, Source.launch Mathematics.ideal shape st eps.value
        (state (fun _ => memory) a scratch)=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        final.mem "Y" (yOffset st row col.val)=.mk .fp32
          (Contract.Math.output shape.columns (logicalX (fun _ => memory) a row)
            (logicalW (fun _ => memory) a row) eps.value col.val)) ∧
      (∀ region off, KernelGrid.Outside st (Source.programs shape) region off →
        final.mem region off=(state (fun _ => memory) a scratch).mem region off) := by
  obtain ⟨final,run,outputs,frame⟩ := KernelBoundary.complete Mathematics.ideal a (fun _ => memory) scratch eps.value
  refine ⟨final,run,?_,frame⟩
  intro row col
  rw [outputs row col,Mathematics.output_real _ _ _ _ _ col.isLt]

/-- All raw FP32 words, the selected actual scalar ABI, FMA, sqrt/division,
operand order, identity casts, and explicit complete reduction trees remain
represented. Numerical ports and compiler decoding are modeling boundaries;
there is no deferred transformation law or split/FMA equality premise. -/
theorem layernorm_fwd_triton_native (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (R : NativeKernel.Reductions) (a : Arguments shape st eps)
    (abi : KernelABI.EpsilonBinding eps) (read : Nat → Torch.Word) (scratch : State NativeKernel.Term) :
    ∃ final, Source.launch (NativeKernel.algebra R) shape st abi.term
        (state (Physical.readTerm read) a scratch)=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        NativeKernel.observeCell I P (final.mem "Y" (yOffset st row col.val))=.mk .fp32
          (NativeKernel.reference I P R shape.columns
            (logicalX (Physical.readTerm read) a row) (logicalW (Physical.readTerm read) a row) abi.term col.val)) ∧
      (∀ region off, KernelGrid.Outside st (Source.programs shape) region off →
        final.mem region off=(state (Physical.readTerm read) a scratch).mem region off) ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns,
        (KernelABI.pointer a.x.dataPtr row.1.val row.2.val col.val st.xSeq st.xHead st.xDim).toNat=
          address a.x (xOffset st row col.val) ∧
        (KernelABI.pointer a.w.dataPtr 0 row.2.val col.val 0 st.wHead st.wDim).toNat=
          address a.w (wOffset st row col.val) ∧
        (KernelABI.pointer a.y.dataPtr row.1.val row.2.val col.val st.ySeq st.yHead st.yDim).toNat=
          address a.y (yOffset st row col.val)) := by
  obtain ⟨final,run,outputs,frame⟩ := KernelBoundary.complete (NativeKernel.algebra R) a (Physical.readTerm read) scratch abi.term
  refine ⟨final,run,?_,frame,KernelABI.active_pointers a⟩
  intro row col
  rw [outputs row col]
  rfl

/-- Native physical corollary at the actual supplied output pointer. The
ordinary per-store byte refinement gives every FP32 output byte, all exterior
bytes (including holes), and every original input backing byte. -/
theorem layernorm_fwd_triton_native_physical (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (R : NativeKernel.Reductions) (a : Arguments shape st eps)
    (abi : KernelABI.EpsilonBinding eps) (heap : Physical.Bytes) (scratch : State NativeKernel.Term) :
    ∃ final, Source.launch (NativeKernel.algebra R) shape st abi.term
        (state (Physical.readTerm (Physical.read32 heap)) a scratch)=some final ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns, ∀ b : Fin 4,
        Physical.outputBytes I P a heap final (address a.y (yOffset st row col.val)+b.val)=
          Physical.byteOf (NativeKernel.reference I P R shape.columns
            (logicalX (Physical.readTerm (Physical.read32 heap)) a row)
            (logicalW (Physical.readTerm (Physical.read32 heap)) a row) abi.term col.val) b.val) ∧
      (∀ byte, Physical.Outside a byte → Physical.outputBytes I P a heap final byte=heap byte) ∧
      (∀ t : Torch.Tensor, t=a.x ∨ t=a.w → ∀ i, i<t.storage.elements → ∀ b : Fin 4,
        Physical.outputBytes I P a heap final (t.storage.address i+b.val)=heap (t.storage.address i+b.val)) ∧
      (∀ row : Row shape, ∀ col : Fin shape.columns, ∀ b : Fin 4,
        (KernelABI.pointer a.y.dataPtr row.1.val row.2.val col.val st.ySeq st.yHead st.yDim+
          BitVec.ofNat 64 b.val).toNat=address a.y (yOffset st row col.val)+b.val) := by
  obtain ⟨final,run,outputs,frame⟩ := KernelBoundary.complete (NativeKernel.algebra R) a
    (Physical.readTerm (Physical.read32 heap)) scratch abi.term
  refine ⟨final,run,?_,Physical.frame I P a heap final,?_,Physical.output_pointer a⟩
  · intro row col b
    rw [Physical.coordinate_bytes,outputs row col]
    rfl
  · intro t ht i hi b
    exact Physical.input_frame I P a heap final t ht i hi b

#axiomsClean layernorm_fwd_triton_real
#axiomsClean layernorm_fwd_triton_native
#axiomsClean python_checks_iff_preconditions
#axiomsClean layernorm_fwd_triton_native_physical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.TensorBridge
