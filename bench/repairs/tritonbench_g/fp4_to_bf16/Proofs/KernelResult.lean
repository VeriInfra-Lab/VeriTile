import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.RawKernel
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.KernelBinding
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Launch
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPReal

namespace VeriTile.Bench.Repairs.Fp4ToBf16.KernelResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule RealSpec
open Bytes KernelBinding
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Independent signed E2M1 specification, high nibble then low nibble.
Exponent bias is one; the exponent-zero magnitude is m/2, otherwise
(1+m/2)*2^(e-1). No kernel state, output allocation or conversion is used. -/
def mathematical (packed : Nat → BitVec 8) (j : Nat) : ℝ :=
  e2m1 (logicalNibble packed j)

/-- Independent native BF16 format table, retaining both signs of zero. -/
def nativeTarget (packed : Nat → BitVec 8) (j : Nat) : FP.Value .bf16 :=
  ⟨Bits.expectedBits (logicalNibble packed j)⟩

/-- Representation of arbitrary actual input and torch.empty output bytes.
The existing roundtrip proof shows every heap has this initial state. -/
def initial (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (heap : Heap) (scratch : State Word) : State Word :=
  represent h (b.storage checked) heap scratch

theorem initial_realization (b : Binding h x z) (checked : Checked.InputContract x.metadata)
    (heap : Heap) (scratch : State Word) :
    physical h (b.storage checked) heap (initial b checked heap scratch).mem=heap :=
  physical_initial h (b.storage checked) heap _
    (represent_correct h (b.storage checked) heap scratch b.regions)

/-- The complete actual grid executes the shared typed conversion source.
Finite input/count/pid/index/mask/address refinement is reused from Launch
and Finite. Grid representability is a launch API premise, not a new guard. -/
theorem native_complete (rest : Algebra Word) (h : Input) (x : PackedTensor)
    (checked : Checked.InputContract x.metadata) (z : Bf16Tensor) (b : Binding h x z)
    (heap : Heap) (scratch : State Word) (jobs : List Nat)
    (gridSafe : grid h≤2147483647) (order : jobs.Perm (List.range (grid h))) :
    ∃ t,Launch.run rest h jobs (initial b checked heap scratch)=some t ∧
      (∀ j,j<z.metadata.numel →
        z.value (physical h (b.storage checked) heap t.mem) j=nativeTarget (x.value heap) j) ∧
      (∀ region o,region≠h.output ∨ 2*x.metadata.numel≤o →
        t.mem region o=(initial b checked heap scratch).mem region o) ∧
      (∀ address,address<z.viewBase ∨ z.viewBase+4*x.metadata.numel≤address →
        physical h (b.storage checked) heap t.mem address=heap address) ∧
      (∀ o,o<x.capacity → physical h (b.storage checked) heap t.mem (x.base+o)=heap (x.base+o)) := by
  have rep : StorageRep h (b.storage checked) heap (initial b checked heap scratch) :=
    represent_correct h (b.storage checked) heap scratch b.regions
  obtain ⟨t,run,memory,frame⟩ :=
    RawKernel.complete rest h (b.storage checked) jobs
      (initial b checked heap scratch) b.regions
      (packed_of_rep h (b.storage checked) heap _ rep.input) gridSafe order
  refine ⟨t,?_,?_,?_,?_,input_storage_frame b checked heap t.mem⟩
  · exact run
  · intro j inside
    have hj : j<2*h.shape.count := by simpa only [b.output_count] using inside
    have bits : (z.value (physical h (b.storage checked) heap t.mem) j).bits=
        (nativeTarget (x.value heap) j).bits := by
      rw [output_read b checked heap t.mem j hj,memory]
      have bound : j<h.shape.count*2 := by omega
      simp only [expectedMemory,bound,and_self,if_true,cell,value]
      rw [read_embed,nibble_from_bytes h (b.storage checked) heap _ rep.input j hj]
      rw [logical_nibble b checked heap j (by simpa only [b.input_count] using hj)]
      rfl
    cases left : z.value (physical h (b.storage checked) heap t.mem) j
    cases right : nativeTarget (x.value heap) j
    simpa only [left,right] using congrArg (fun w : BitVec 16 => (⟨w⟩ : FP.Value .bf16)) bits
  · intro region o outside
    apply frame
    simpa only [b.input_count,Nat.mul_comm] using outside
  · intro address outside
    apply physical_frame
    simpa only [b.input_count,Binding.storage] using outside

#axiomsClean initial_realization
#axiomsClean native_complete
#specNonCircular mathematical avoiding [source,Kernels.source,Launch.run,initial,physical,nativeTarget]
#specNonCircular nativeTarget avoiding [source,Kernels.source,Launch.run,initial,physical]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4ToBf16.KernelResult
