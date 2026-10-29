import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Host

/-! Finite address/mask certificate for both actual tensors in the launch.
No address, stride or output-coverage obligation is a numerical assumption. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.Host
open VeriTile.Triton FP.Structural
noncomputable section

/-- The allocator's output satisfies the same relevant access contract; these
facts are derived, not additional Python input checks. -/
theorem output_contract {α : Type} [Inhabited α] (x : Tensor) (h : InputContract x.metadata)
    (before : State α) (a : EmptyLikeResult x h before) : InputContract a.output.metadata where
  layout := by rw [a.metadata];exact h.layout
  dtype := by rw [a.metadata];exact h.dtype
  cuda := by rw [a.metadata];exact h.cuda
  contiguous := a.contiguous
  noNeg := by rw [a.metadata]
  extent := by rw [a.numel];exact h.extent
  alignment := a.alignment

/-- The native load/store address includes the Tensor's actual storage offset.
Each complete FP32 access is aligned and lies within its backing allocation. -/
structure Access (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (pid lane : Nat) : Prop where
  exact : (x.dataPtr +
    (VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset32 pid lane).signExtend 64 *
      BitVec.ofNat 64 4).toNat = base x laws h.contiguous+4*(pid*1024+lane)
  aligned : (base x laws h.contiguous+4*(pid*1024+lane))%4=0
  inStorage : x.storage.base≤base x laws h.contiguous+4*(pid*1024+lane) ∧
    base x laws h.contiguous+4*(pid*1024+lane)+4≤x.storage.base+4*x.storage.elements
  noWrap : base x laws h.contiguous+4*(pid*1024+lane)+4≤2^64

theorem access_safe (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (pid lane : Nat) (hp : pid<gridCount x.metadata.numel) (hl : lane<1024)
    (active : pid*1024+lane<x.metadata.numel) : Access x laws h pid lane := by
  obtain ⟨he,ha,hb⟩:=native_pointer x laws h pid lane hp hl active
  rw [he] at ha hb
  exact ⟨he,ha,data_inside x laws h.contiguous _ active,hb⟩

/-- Every scheduled lane has exact signed-int32 indexing/mask, and every active
load and store has its actual aligned, bounded 64-bit address. -/
theorem launch_safe {α : Type} [Inhabited α] (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (before : State α) (a : EmptyLikeResult x h before)
    (schedule : Structural.Schedule x.metadata.numel) (pid lane : Nat)
    (hp : pid∈schedule.programs) (hl : lane<1024) :
    (VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset32 pid lane).toInt=
      (pid*1024+lane : Nat) ∧
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.mask32 x.metadata.numel pid lane=
      decide (pid*1024+lane<x.metadata.numel) ∧
    (pid*1024+lane<x.metadata.numel →
      Access x laws h pid lane ∧ Access a.output a.laws (output_contract x h before a) pid lane) := by
  have launched:=List.mem_range.mp (schedule.complete.mem_iff.mp hp)
  obtain ⟨index,mask⟩:=native_indices x.metadata h pid lane launched hl
  refine ⟨index,mask,?_⟩
  intro active
  exact ⟨access_safe x laws h pid lane launched hl active,
    access_safe a.output a.laws (output_contract x h before a) pid lane
      (by rw [a.numel];exact launched) hl (by rw [a.numel];exact active)⟩

/-- Separate programs cannot race on a live output word. -/
theorem unique_writer (pid other lane otherLane : Nat) (hl : lane<1024) (ho : otherLane<1024)
    (equal : pid*1024+lane=other*1024+otherLane) : pid=other ∧ lane=otherLane := by omega

#axiomsClean launch_safe
#axiomsClean unique_writer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.Host
