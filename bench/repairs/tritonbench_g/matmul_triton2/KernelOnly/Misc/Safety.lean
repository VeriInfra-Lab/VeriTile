import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Tensor
import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.FiniteSource

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Safety
open VeriTile.Triton FP.Structural Revision001 Model
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 8000000

theorem denseA (c : Arguments) (h : KernelContract c) (choice : Fin 8) :
    Core.DenseView (parameters c choice).rows (parameters c choice).innerSize
      (parameters c choice).inputRow (parameters c choice).inputInner c.a.metadata.numel := by
  rw [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract.matrix_numel c.a.metadata h.inputs.rank.1]
  exact ⟨le_refl _,by intro r col hr hc; simp [parameters]⟩

theorem denseB (c : Arguments) (h : KernelContract c) (choice : Fin 8) :
    Core.DenseView (parameters c choice).innerSize (parameters c choice).columns
      (parameters c choice).weightInner (parameters c choice).weightColumn c.b.metadata.numel := by
  rw [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract.matrix_numel c.b.metadata h.inputs.rank.2,←h.inputs.dimensions]
  exact ⟨le_refl _,by intro r col hr hc; simp [parameters]⟩

theorem denseC (c : Arguments) (h : KernelContract c) (choice : Fin 8) :
    Core.DenseView (parameters c choice).rows (parameters c choice).columns
      (parameters c choice).outputRow (parameters c choice).outputColumn c.output.metadata.numel := by
  rw [TensorBinding.output_numel c h]
  exact ⟨le_refl _,by intro r col hr hc; simp [parameters]⟩

theorem active_accesses {α : Type} (c : Arguments) (h : KernelContract c) (choice : Fin 8)
    (s : State α) (block : Nat) :
    (∀ i : Core.AIndex (parameters c choice),
      Core.row (parameters c choice) s i.1 < rows c.a.metadata →
      i.2.1.val < columns c.a.metadata-block*(Native.config choice).blockInner →
      Core.aOffset (parameters c choice) s block i < c.a.metadata.numel) ∧
    (∀ i : Core.BIndex (parameters c choice),
      i.1.val < columns c.a.metadata-block*(Native.config choice).blockInner →
      Core.column (parameters c choice) s i.2.1 < columns c.b.metadata →
      Core.bOffset (parameters c choice) s block i < c.b.metadata.numel) ∧
    (∀ i : Core.CIndex (parameters c choice),Core.active (parameters c choice) s i →
      Core.cOffset (parameters c choice) s i < c.output.metadata.numel) := by
  exact ⟨fun i hr hk => Core.active_a_bound _ _ _ _ (denseA c h choice) i hr hk,
    fun i hk hc => Core.active_b_bound _ _ _ _ (denseB c h choice) i hk hc,
    fun i hi => Core.active_c_bound _ _ _ (denseC c h choice) i hi⟩

theorem pointer_windows (c : Arguments) (h : KernelContract c) :
    ∀ t,t=c.a ∨ t=c.b ∨ t=c.output → ∀ i,i < t.metadata.numel →
      (t.metadata.dataPtr+VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word (i*4)).toNat=address t i ∧
      (t.metadata.dataPtr+VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word (i*4)).toNat%4=0 := by
  intro t selected i hi
  have one (t : Tensor) (laws : Laws t) (contiguous : t.metadata.isContiguous=Bool.true)
      (alignment : 0 < t.metadata.numel → t.metadata.dataPtr.toNat%4=0)
      (i : Nat) (hi : i < t.metadata.numel) :
      (t.metadata.dataPtr+VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word (i*4)).toNat=address t i ∧
      (t.metadata.dataPtr+VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word (i*4)).toNat%4=0 := by
    have window := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding.pointer_window
      t laws contiguous alignment i hi
    simp only [VeriTile.Torch.FP32.Tensor.dataPtr] at window
    refine ⟨?_,window.2.1⟩
    rw [window.1,address,TensorBinding.pointer_base t laws contiguous (by omega)]
    rfl
  rcases selected with rfl|rfl|rfl
  · exact one c.a c.leftLaws h.inputs.contiguous.1 h.inputs.alignment.1 i hi
  · exact one c.b c.rightLaws h.inputs.contiguous.2 h.inputs.alignment.2 i hi
  · exact one c.output c.outputLaws h.output.contiguous h.output.alignment i hi

theorem storage (c : Arguments) (h : KernelContract c) (choice : Fin 8) :
    VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.StorageBounds (parameters c choice) := by
  have a := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding.span c.a c.leftLaws h.inputs.contiguous.1
  have b := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding.span c.b c.rightLaws h.inputs.contiguous.2
  have out := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.TensorBinding.span c.output c.outputLaws h.output.contiguous
  rw [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract.matrix_numel c.a.metadata h.inputs.rank.1] at a
  rw [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.SharedContract.matrix_numel c.b.metadata h.inputs.rank.2,←h.inputs.dimensions] at b
  rw [TensorBinding.output_numel c h] at out
  exact ⟨a,b,out⟩

/-- The inherited finite executor models a representable CUDA grid. This is
an explicit auxiliary compatibility boundary, not a new input assertion or a
premise of the complete quoted-source correctness endpoints. -/
theorem ready (c : Arguments) (h : KernelContract c) (choice : Fin 8)
    (representable : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.count (parameters c choice) < 2^31) :
    VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.Ready (parameters c choice) := by
  refine ⟨h.inputs.extent.1,h.inputs.extent.2.2,h.inputs.extent.2.1,?_,?_,?_,?_,representable⟩ <;>
    fin_cases choice <;> norm_num [parameters,Native.config]

theorem canonical (c : Arguments) (choice : Fin 8) : FiniteSource.Canonical (parameters c choice) :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem finite_source {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (c : Arguments) (h : KernelContract c) (choice : Fin 8)
    (representable : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.count (parameters c choice) < 2^31)
    (before : State α) (active : before.pids 0 < VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.count (parameters c choice)) :
    ∃ actual typed,
      VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteExecution.program P
        (parameters c choice) Bool.true before=some actual ∧
      FP.Structural.exec (Core.backend rest P) (kernel (parameters c choice)) before=some typed ∧
      actual.mem=typed.mem :=
  FiniteSource.program_source rest P _ (canonical c choice) (ready c h choice representable)
    (storage c h choice) before active

theorem scalar_ABI (n : Nat) : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.scalarABI n=VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word n := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.scalarABI_word n
theorem padded_pointer_residue (offset : BitVec 64) :
    BitVec.ofInt 64 (4*offset.toInt)=VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.word 4*offset := VeriTile.Bench.Repairs.MatmulTriton1.Revision001.FiniteIndex.signed_pointer_offset offset

#axiomsClean active_accesses
#axiomsClean pointer_windows
#axiomsClean storage
#axiomsClean finite_source
#axiomsClean scalar_ABI
#axiomsClean padded_pointer_residue
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Safety
