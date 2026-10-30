import bench.repairs.tritonbench_g.rope_transform.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rope_transform.Correct
import bench.repairs.tritonbench_g.rope_transform.FPEquiv
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelBytes
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelAccess

namespace VeriTile.Bench.Repairs.RopeTransform.TensorBridge

open VeriTile.Triton FP.Structural KernelContract
noncomputable section

/-- Kernel-only complete forward rotation on the actual prepared Q/K Tensors.
The same successful full source grid supplies all values and the full frame. -/
theorem rope_forward_kernel_correct (read : Nat→Word) (interpret : Word→ℝ)
    (c : Call) (admit : Admitted c) :
    InputBinding c read ∧ Access c ∧
    ∃ final,sourceRun realAlgebra c (entry id (interpret ∘ read) c) c.geometry.rows=some final ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        logicalRead (physical id c (interpret ∘ read) final.mem) (c.q key) (c.geometry.offset key row head col)=
          realValue c (interpret ∘ read) key row head col) ∧
      (∀ name index,¬WrittenRegion c name index →final.mem name index=memory id (interpret ∘ read) c name index) ∧
      (∀ address,¬WrittenWord c Bool.false address ∧ ¬WrittenWord c Bool.true address →
        physical id c (interpret ∘ read) final.mem address=(interpret ∘ read) address) := by
  refine ⟨input_binding c admit read,access c admit,
    Grid.run realAlgebra c.geometry (entry id (interpret ∘ read) c) c.geometry.rows,
    source_success realAlgebra c _,?_,grid_frame realAlgebra id c (interpret ∘ read),physical_frame id c _ _⟩
  intro key row head col hr hh hc
  simpa only [real_binding,id_eq] using
    physical_values realAlgebra id id c admit (interpret ∘ read) key row head col hr hh hc

theorem rope_forward_kernel_no_launch {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (s : State α) (empty : c.geometry.batches=0 ∨ c.geometry.sequence=0) :
    c.geometry.rows=0 ∧ sourceRun M c s c.geometry.rows=some s := by
  have zero : c.geometry.rows=0 := by rcases empty with h|h <;> simp [Request.rows,h]
  exact ⟨zero,by rw [zero]; rfl⟩

#axiomsClean rope_forward_kernel_correct
#axiomsClean rope_forward_kernel_no_launch
end

open VeriTile.Triton FP.Structural KernelContract
noncomputable section

/-- Actual Tensor/byte inputs, complete typed grid success, all Q/K words and
four-byte readbacks, and full regional/exterior frame. Primitive cfg and the
observed coefficient-first forward FMA lowering are explicit model boundaries. -/
theorem rope_forward_kernel_fp32 (cfg : FP.Config) (c : Call) (admit : Admitted c) (before : Bytes) :
    InputBinding c (read32 before) ∧ Access c ∧
    ∃ final,sourceRun (nativeAlgebra cfg) c (entry leaf (read32 before) c) c.geometry.rows=some final ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        logicalRead (read32 (byteMemory c before final.mem)) (c.q key) (c.geometry.offset key row head col)=
          nativeValue cfg c (read32 before) key row head col) ∧
      (∀ key row head col,row<c.geometry.rows →head<c.geometry.heads key →col<c.geometry.dimension →
        ∀ byte : Fin 4,byteMemory c before final.mem
            ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val)=
          wordByte (nativeValue cfg c (read32 before) key row head col) byte) ∧
      (∀ name index,¬WrittenRegion c name index →final.mem name index=memory leaf (read32 before) c name index) ∧
      (∀ address,¬WrittenByte c Bool.false address ∧ ¬WrittenByte c Bool.true address →
        byteMemory c before final.mem address=before address) := by
  refine ⟨input_binding c admit (read32 before),access c admit,
    Grid.run (nativeAlgebra cfg) c.geometry (entry leaf (read32 before) c) c.geometry.rows,
    source_success _ c _,?_,?_,grid_frame _ leaf c (read32 before),bytes_frame c before _⟩
  · exact native_words cfg c admit before
  · exact native_bytes cfg c admit before

#axiomsClean rope_forward_kernel_fp32
end
end VeriTile.Bench.Repairs.RopeTransform.TensorBridge
