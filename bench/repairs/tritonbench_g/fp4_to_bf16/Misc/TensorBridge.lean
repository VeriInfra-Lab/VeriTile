import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.KernelResult
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Contract

/-! Optional Tensor input bindings and Python-check correspondence. Primary
mathematical and native kernel entries do not import this bridge. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host RealSpec
open Bytes KernelBinding KernelResult
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem python_checks_iff_preconditions (x : VeriTile.Torch.ArgumentMetadata) :
    Checked.checkInputs x=.ok () ↔ Checked.AdmittedArguments x := Checked.checkInputs_iff x

theorem fp4_to_bf16_kernel_real (rest : Algebra Word) (h : Input) (x : PackedTensor)
    (checked : Checked.InputContract x.metadata) (z : Bf16Tensor) (binding : Binding h x z)
    (heap : Heap) (scratch : State Word) (jobs : List Nat)
    (gridSafe : grid h≤2147483647) (order : jobs.Perm (List.range (grid h))) :
    VeriTile.Spec.Real (∃ t,
      Launch.run rest h jobs (initial binding checked heap scratch)=some t ∧
      (∀ j,j<z.metadata.numel →
        finiteReal (z.value (physical h (binding.storage checked) heap t.mem) j)=
          some (mathematical (x.value heap) j)) ∧
      (∀ region o,region≠h.output ∨ 2*x.metadata.numel≤o →
        t.mem region o=(initial binding checked heap scratch).mem region o) ∧
      (∀ address,address<z.viewBase ∨ z.viewBase+4*x.metadata.numel≤address →
        physical h (binding.storage checked) heap t.mem address=heap address) ∧
      (∀ o,o<x.capacity →
        physical h (binding.storage checked) heap t.mem (x.base+o)=heap (x.base+o))) := by
  obtain ⟨t,run,values,frame,exterior,input⟩ :=
    native_complete rest h x checked z binding heap scratch jobs gridSafe order
  refine ⟨t,run,?_,frame,exterior,input⟩
  intro j hj
  rw [values j hj]
  exact expected_real _

theorem fp4_to_bf16_kernel_native (rest : Algebra Word) (h : Input) (x : PackedTensor)
    (checked : Checked.InputContract x.metadata) (z : Bf16Tensor) (binding : Binding h x z)
    (heap : Heap) (scratch : State Word) (jobs : List Nat)
    (gridSafe : grid h≤2147483647) (order : jobs.Perm (List.range (grid h))) :
    ∃ t,Launch.run rest h jobs (initial binding checked heap scratch)=some t ∧
      (∀ j,j<z.metadata.numel →
        z.value (physical h (binding.storage checked) heap t.mem) j=nativeTarget (x.value heap) j) ∧
      (∀ region o,region≠h.output ∨ 2*x.metadata.numel≤o →
        t.mem region o=(initial binding checked heap scratch).mem region o) ∧
      (∀ address,address<z.viewBase ∨ z.viewBase+4*x.metadata.numel≤address →
        physical h (binding.storage checked) heap t.mem address=heap address) ∧
      (∀ o,o<x.capacity →
        physical h (binding.storage checked) heap t.mem (x.base+o)=heap (x.base+o)) :=
  native_complete rest h x checked z binding heap scratch jobs gridSafe order

/-- Every padded native signed64 lane mask agrees with the shared source.
The count bound follows from the actual output allocation, not a user guard. -/
theorem native_lane_refinement (h : Input) (x : PackedTensor)
    (checked : Checked.InputContract x.metadata) (z : Bf16Tensor) (binding : Binding h x z)
    (pid li lo : Nat) (hp : pid<grid h) (hi : li<512) (ho : lo<1024) :
    Kernels.inputMask h.shape.count pid li=decide (pid*512+li<h.shape.count) ∧
    Kernels.outputMask h.shape.count pid lo=decide (pid*1024+lo<2*h.shape.count) :=
  Finite.masks_exact _ _ _ _ (Finite.storage_count_bound (binding.storage checked)) hp hi ho

#axiomsClean fp4_to_bf16_kernel_real
#axiomsClean fp4_to_bf16_kernel_native
#axiomsClean native_lane_refinement
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.TensorBridge
