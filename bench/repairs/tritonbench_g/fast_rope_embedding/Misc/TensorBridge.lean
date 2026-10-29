import bench.repairs.tritonbench_g.fast_rope_embedding.Correct
import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.KernelBytes
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelAccess
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.RawReal
import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.RawNative

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.TensorBridge

open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open KernelContract
noncomputable section
open scoped Classical

/-- Kernel-only mathematical correctness. The same actual Tensor input words
supply the ideal scalar interpretation, successful whole-source execution,
every logical Q result, the complete regional/physical frame, and finite
machine-address bounds. The theorem applies to either source direction. -/
theorem fast_rope_kernel_correct (read : Nat→Word) (interpret : Word→ℝ)
    (c : Call) (admit : Admitted c) (row group : Nat) (launch : c.geometry.Launched row group) :
    InputBinding c read ∧ Access c row group ∧
    ∃ final, FP.Structural.exec ideal.backend c.source
        (programState (state id (interpret ∘ read) c.inputs) row group)=some final ∧
      Kernel.Post c.geometry row group (state id (interpret ∘ read) c.inputs) final
        (fun head col=>realValue c (interpret ∘ read) row head col) ∧
      (∀ peer head col,peer<c.geometry.rows → head<c.geometry.heads → col<c.geometry.dimension →
        logicalRead (physical id c.inputs (interpret ∘ read) final) c.inputs.q (c.geometry.offset peer head col)=
          if affected c row group peer head col then realValue c (interpret ∘ read) row head col
          else logicalRead (interpret ∘ read) c.inputs.q (c.geometry.offset peer head col)) ∧
      (∀ address,¬WrittenAddress c row group address →
        physical id c.inputs (interpret ∘ read) final address=(interpret ∘ read) address) := by
  refine ⟨input_binding c admit read,access c admit row group launch,?_⟩
  simpa only [real_binding,id_eq] using
    tensor_execution ideal id id (fun _=>rfl) c admit (interpret ∘ read) row group launch

/-- Every rotated Tensor coordinate is covered, including padded blocks and
partial head groups. This describes the kernel grid, not host launch code. -/
theorem fast_rope_kernel_coverage (c : Call) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads) (hc : col<2*c.geometry.half) :
    ∃ group,c.geometry.Launched row group ∧
      Kernel.Written c.geometry row group "Q" (c.geometry.offset row head col) := by
  obtain ⟨group,hg,written⟩ := Kernel.complete_coverage c.geometry row head col hr hh hc
  exact ⟨group,⟨by omega,hr,hg⟩,written⟩

/-- Empty payloads and D<2 have an empty declared kernel grid. -/
theorem fast_rope_no_launch (c : Call) (empty : c.geometry.size=0 ∨ c.geometry.dimension<2) :
    ∀ row group,¬c.geometry.Launched row group := by
  intro row group launch
  have head := group_head c.geometry group launch.2.2
  have half := launch.1
  have rows := launch.2.1
  rcases empty with empty | empty
  · unfold Request.size at empty
    have dim : 0<c.geometry.dimension := by unfold Request.half at half; omega
    have positive := Nat.mul_pos (Nat.mul_pos (show 0<c.geometry.rows by omega)
      (show 0<c.geometry.heads by omega)) dim
    omega
  · unfold Request.half at half; omega

#axiomsClean fast_rope_kernel_correct
#axiomsClean fast_rope_kernel_coverage
#axiomsClean fast_rope_no_launch
end

open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open KernelContract
noncomputable section
open scoped Classical

/-- One composed native endpoint, starting from an actual byte heap and actual
Tensor pointers. The whole typed source succeeds; every output word and all
four bytes are observations of its single final memory. Unwritten regional
cells and all bytes outside the actual store footprint are framed. Primitive FP32 semantics and the
observed source-specific lowering are the explicit backend boundary; no local
Real exactness, output callback, or successful execution is assumed. -/
theorem fast_rope_kernel_fp32 (cfg : FP.Config) (c : Call) (admit : Admitted c)
    (before : Bytes) (row group : Nat) (launch : c.geometry.Launched row group) :
    InputBinding c (read32 before) ∧ Access c row group ∧
    ∃ final, FP.Structural.exec (Numerics.operations (Numerics.softwarePrimitives cfg)).backend c.source
        (programState (state Numerics.leaf (read32 before) c.inputs) row group)=some final ∧
      (∀ peer head col,peer<c.geometry.rows → head<c.geometry.heads → col<c.geometry.dimension →
        logicalRead (read32 (byteMemory c row group before final)) c.inputs.q (c.geometry.offset peer head col)=
          if affected c row group peer head col then
            nativeValue (Numerics.softwarePrimitives cfg) c (read32 before) row head col
          else logicalRead (read32 before) c.inputs.q (c.geometry.offset peer head col)) ∧
      (∀ peer head col,peer<c.geometry.rows → head<c.geometry.heads → col<c.geometry.dimension →
        ∀ byte : Fin 4,byteMemory c row group before final
            (c.inputs.q.storage.address (c.inputs.q.physicalIndex (c.geometry.offset peer head col))+byte.val)=
          if affected c row group peer head col then
            wordByte (nativeValue (Numerics.softwarePrimitives cfg) c (read32 before) row head col) byte
          else before (c.inputs.q.storage.address (c.inputs.q.physicalIndex (c.geometry.offset peer head col))+byte.val)) ∧
      (∀ region offset,¬Kernel.Written c.geometry row group region offset →
        final.mem region offset=(state Numerics.leaf (read32 before) c.inputs).mem region offset) ∧
      (∀ address,¬WrittenByte c row group address →byteMemory c row group before final address=before address) := by
  refine ⟨input_binding c admit (read32 before),access c admit row group launch,?_⟩
  obtain ⟨final,ran,post,_,_⟩ := tensor_execution
    (Numerics.operations (Numerics.softwarePrimitives cfg)) Numerics.leaf Numerics.Tagged.word
    (fun _=>rfl) c admit (read32 before) row group launch
  refine ⟨final,ran,?_,?_,post.frame,bytes_frame c row group before final⟩
  · intro peer head col hr hh hc
    simpa only [native_binding] using tensor_byte_word c admit row group before final _ post peer head col hr hh hc
  · intro peer head col hr hh hc byte
    simpa only [native_binding] using tensor_bytes c admit row group before final _ post peer head col hr hh hc byte

#axiomsClean fast_rope_kernel_fp32
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.TensorBridge
