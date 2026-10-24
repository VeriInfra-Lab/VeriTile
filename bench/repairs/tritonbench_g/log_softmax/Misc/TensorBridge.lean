import bench.repairs.tritonbench_g.log_softmax.Correct
import bench.repairs.tritonbench_g.log_softmax.FPEquiv
import bench.repairs.tritonbench_g.log_softmax.Misc.NativeEquivalence
import bench.repairs.tritonbench_g.log_softmax.Proofs.RawCorrect
import bench.repairs.tritonbench_g.log_softmax.Proofs.Machine
import bench.repairs.tritonbench_g.log_softmax.Proofs.TensorBinding

/-! Optional Tensor applications and Python/check-address correspondences for
the independent forward/backward kernel specifications. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.TensorBridge
open VeriTile.Triton FP.Structural Address
open Correct
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section
open Classical

local instance : Inhabited VeriTile.Torch.FP32.Word := ⟨⟨0⟩⟩

theorem forward_kernel_correctness (a : Arguments) (h : InputContract a)
    (BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan a.N (Grid.blockColumns a.N)) (initial : State (WithBot ℝ))
    (xs : Nat → ℝ) (binding : KernelMath.InputValues initial "input" a.M a.N a.K xs) :
    Spec.Real (∃ final,
      KernelGrid.run .forward (KernelMath.forwardEngine RealProof.ops plan) Bool.false
        a.M a.N a.K BM (KernelGrid.programs a.M a.N a.K BM) initial=some final ∧
      (∀ q : Coord a.M a.N a.K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetForward
          (fun j : Fin a.N => xs (offset a.N a.K q.1.val j.val q.2.2.val)) q.2.1))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.input.numel≤ i → final.mem r i=initial.mem r i)) := by
  obtain ⟨final,run,values,frame⟩ := RawCorrect.forward_grid_correctness a.M a.N a.K BM retained plan initial xs binding
  refine ⟨final,run,values,?_⟩
  intro r i outside
  apply frame r i
  simpa only [input_numel a h.2.2] using outside

theorem backward_kernel_correctness (a : BackwardArguments) (h : BackwardContract a)
    (BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan a.saved.N (Grid.blockColumns a.saved.N)) (initial : State (WithBot ℝ))
    (ys gs : Nat → ℝ)
    (saved : KernelMath.InputValues initial "input" a.saved.M a.saved.N a.saved.K ys)
    (upstream : KernelMath.InputValues initial "gradient" a.saved.M a.saved.N a.saved.K gs) :
    Spec.Real (∃ final,
      KernelGrid.run .backward (KernelMath.backwardEngine RealProof.ops plan) Bool.false
        a.saved.M a.saved.N a.saved.K BM (KernelGrid.programs a.saved.M a.saved.N a.saved.K BM)
        initial=some final ∧
      (∀ q : Coord a.saved.M a.saved.N a.saved.K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetBackward a.saved.N
          (fun j => ys (offset a.saved.N a.saved.K q.1.val j q.2.2.val))
          (fun j => gs (offset a.saved.N a.saved.K q.1.val j q.2.2.val)) q.2.1.val))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.saved.input.numel≤ i → final.mem r i=initial.mem r i)) := by
  obtain ⟨final,run,values,frame⟩ := RawCorrect.backward_grid_correctness a.saved.M a.saved.N a.saved.K BM retained plan initial ys gs saved upstream
  refine ⟨final,run,values,?_⟩
  intro r i outside
  apply frame r i
  simpa only [input_numel a.saved h.2.2.2.2] using outside

/-- Public Real endpoint bound to the original Tensor's logical values. The
initial mathematical state is constructed from actual source FP32 words; only
the existing contiguous() element law and ordinary pointer reads are premises.
No host execution, successful kernel, output formula or finite guard is assumed. -/
theorem forward_tensor_correctness (a : Arguments) (h : InputContract a)
    (BM : Nat) (retained : Grid.RetainedBlock BM) (plan : Plan a.N (Grid.blockColumns a.N))
    (input : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws input)
    (metadata : input.metadata=a.input) (before after : Nat → VeriTile.Torch.FP32.Word)
    (prepared : Binding.ContiguousResult input laws before after)
    (words : State VeriTile.Torch.FP32.Word)
    (binding : Binding.PointerInput words "input" after prepared.tensor)
    (interpret : VeriTile.Torch.FP32.Word → ℝ) :
    Spec.Real (∃ final,
      KernelGrid.run .forward (KernelMath.forwardEngine RealProof.ops plan) Bool.false
        a.M a.N a.K BM (KernelGrid.programs a.M a.N a.K BM) (Binding.realState interpret words)=some final ∧
      (∀ q : Coord a.M a.N a.K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetForward
          (fun j : Fin a.N => Binding.logicalReal interpret before input laws
            (offset a.N a.K q.1.val j.val q.2.2.val)) q.2.1))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.input.numel≤ i →
        final.mem r i=(Binding.realState interpret words).mem r i)) := by
  apply forward_kernel_correctness a h BM retained plan (Binding.realState interpret words)
    (Binding.logicalReal interpret before input laws)
  intro i hi
  apply Binding.prepared_real_read interpret words "input" input laws before after prepared binding i
  rw [metadata,←input_numel a h.2.2]
  exact hi

/-- Saved values are read from the actual saved Tensor at kernel entry; the
gradient formula uses the caller's logical gradient before its existing
contiguous() call. Both source inputs share the same absolute entry memory.
The unchanged checked contract supplies saved contiguity, resolved views and
alignment; no extra input guard or whole-output hypothesis is introduced. -/
theorem backward_tensor_correctness (a : BackwardArguments) (h : BackwardContract a)
    (BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan a.saved.N (Grid.blockColumns a.saved.N))
    (saved gradient : VeriTile.Torch.FP32.Tensor)
    (savedLaws : VeriTile.Torch.FP32.TensorLaws saved)
    (gradientLaws : VeriTile.Torch.FP32.TensorLaws gradient)
    (savedMetadata : saved.metadata=a.saved.input) (gradientMetadata : gradient.metadata=a.gradient)
    (before after : Nat → VeriTile.Torch.FP32.Word)
    (prepared : Binding.ContiguousResult gradient gradientLaws before after)
    (words : State VeriTile.Torch.FP32.Word)
    (savedInput : Binding.PointerInput words "input" after saved)
    (gradientInput : Binding.PointerInput words "gradient" after prepared.tensor)
    (interpret : VeriTile.Torch.FP32.Word → ℝ) :
    Spec.Real (∃ final,
      KernelGrid.run .backward (KernelMath.backwardEngine RealProof.ops plan) Bool.false
        a.saved.M a.saved.N a.saved.K BM (KernelGrid.programs a.saved.M a.saved.N a.saved.K BM)
        (Binding.realState interpret words)=some final ∧
      (∀ q : Coord a.saved.M a.saved.N a.saved.K,final.mem "output" (linear q)=.mk .real
        (some (RealProof.targetBackward a.saved.N
          (fun j => Binding.logicalReal interpret after saved savedLaws
            (offset a.saved.N a.saved.K q.1.val j q.2.2.val))
          (fun j => Binding.logicalReal interpret before gradient gradientLaws
            (offset a.saved.N a.saved.K q.1.val j q.2.2.val)) q.2.1.val))) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.saved.input.numel≤ i →
        final.mem r i=(Binding.realState interpret words).mem r i)) := by
  apply backward_kernel_correctness a h BM retained plan (Binding.realState interpret words)
    (Binding.logicalReal interpret after saved savedLaws)
    (Binding.logicalReal interpret before gradient gradientLaws)
  · intro i hi
    have valid : MaxReduction.TensorValid saved.metadata Bool.true := by
      simpa only [savedMetadata] using h.1
    apply Binding.direct_real_read interpret words "input" after saved savedLaws
      (valid.2.2.2.2.1 rfl) valid.2.2.2.1 valid.2.2.2.2.2.2 savedInput i
    rw [savedMetadata,←input_numel a.saved h.2.2.2.2]
    exact hi
  · intro i hi
    apply Binding.prepared_real_read interpret words "gradient" gradient gradientLaws before after prepared
      gradientInput i
    have elements : gradient.metadata.numel=a.saved.input.numel := by
      simp only [VeriTile.Torch.TensorMetadata.numel,gradientMetadata,h.2.2.1]
    rw [elements,←input_numel a.saved h.2.2.2.2]
    exact hi

/-- The exact address expressions of the canonical source obey the retained
signed64 masks and four-byte backing bounds, including inactive clamped lanes.
The descriptor is bound to observed Tensor storage by Binding.dense. -/
theorem kernel_native_access (a : Arguments) (h : InputContract a)
    (nonempty : 0<a.input.numel) (BM p k : Nat) (retained : Grid.RetainedBlock BM)
    (hp : p<Grid.groups a.M BM) (hk : k<a.K)
    (buffer : Dense (a.M*a.N*a.K)) (i : TileIndex [BM,Grid.blockColumns a.N]) :
    ((Machine.word p*Machine.word BM+Machine.word i.1.val).slt (Machine.word a.M)=
        decide (p*BM+i.1.val<a.M) ∧
      (Machine.word i.2.1.val).slt (Machine.word a.N)=decide (i.2.1.val<a.N) ∧
      (Machine.word k).slt (Machine.word a.K)=Bool.true) ∧
    (Machine.word buffer.base+Machine.word (Kernel.address a.M a.N a.K BM p k i)*Machine.word 4).toNat=
      buffer.address (Kernel.address a.M a.N a.K BM p k i) ∧
    (∀ b,b<4 → buffer.contains (buffer.address (Kernel.address a.M a.N a.K BM p k i)+b)) := by
  have positive := launch_bounds a h nonempty BM retained
  have bound : Kernel.address a.M a.N a.K BM p k i<a.M*a.N*a.K := by
    simpa only [Kernel.address,safe,offset] using
      masked_offset_bound positive.1 positive.2.1 positive.2.2.1 (p*BM+i.1.val) i.2.1.val k
  exact ⟨Machine.native_masks a h nonempty BM p k i.1.val i.2.1.val retained hp hk i.1.isLt i.2.1.isLt,
    Machine.pointer32 buffer _ bound⟩

theorem forward_python_checks (a : Arguments) : checkInputs a=.ok () ↔ InputContract a :=
  checkInputs_iff a

theorem backward_python_checks (a : BackwardArguments) : checkBackward a=.ok () ↔ BackwardContract a :=
  checkBackward_iff a

abbrev Word := VeriTile.Torch.FP32.Word

/-- The whole canonical FP32 source grids generate identical native-observable
results for every word, under a fixed selected reduction layout, instruction
interpretation and fusion schedule. No finite-value guard or primitive numeric
law is required. Graphs retain all source precision and cast boundaries. -/
theorem forward_fp32_grid (a : Arguments) (h : InputContract a) (BM : Nat)
    (_retained : Grid.RetainedBlock BM) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) (memory : RegionName → Nat → Word) :
    ∃ original commuted,
      Observed.run .forward (Native.algebra R) Bool.false a.M a.N a.K BM (Grid.blockColumns a.N)
        (KernelGrid.programs a.M a.N a.K BM) (NativeGrid.entry memory)=some original ∧
      Observed.run .forward (Native.algebra R) Bool.true a.M a.N a.K BM (Grid.blockColumns a.N)
        (KernelGrid.programs a.M a.N a.K BM) (NativeGrid.entry memory)=some commuted ∧
      (∀ (r : RegionName) i,Native.observe I S original r i=Native.observe I S commuted r i) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.input.numel≤ i →
        Native.observe I S original r i=(memory r i).bits ∧
        Native.observe I S commuted r i=(memory r i).bits) := by
  obtain ⟨original,commuted,ra,rb,equal,frame⟩ := NativeGrid.whole .forward R I S
    a.M a.N a.K BM (Grid.blockColumns a.N) (block_positive a.N) memory
  refine ⟨original,commuted,ra,rb,equal,?_⟩
  intro r i outside
  apply frame r i
  simpa only [input_numel a h.2.2] using outside

/-- Native observation retains the default enabled FMA contraction. In
particular NativeGrid.backward_fused interprets the actual adjacent sub/mul
site with fmaRN, without equating that schedule to split rounded operations. -/
theorem backward_fp32_grid (a : BackwardArguments) (h : BackwardContract a) (BM : Nat)
    (_retained : Grid.RetainedBlock BM) (R : Native.Reductions)
    (I : Native.Instructions) (S : Native.Schedule) (memory : RegionName → Nat → Word) :
    ∃ original commuted,
      Observed.run .backward (Native.algebra R) Bool.false a.saved.M a.saved.N a.saved.K BM
        (Grid.blockColumns a.saved.N) (KernelGrid.programs a.saved.M a.saved.N a.saved.K BM)
        (NativeGrid.entry memory)=some original ∧
      Observed.run .backward (Native.algebra R) Bool.true a.saved.M a.saved.N a.saved.K BM
        (Grid.blockColumns a.saved.N) (KernelGrid.programs a.saved.M a.saved.N a.saved.K BM)
        (NativeGrid.entry memory)=some commuted ∧
      (∀ (r : RegionName) i,Native.observe I S original r i=Native.observe I S commuted r i) ∧
      (∀ (r : RegionName) i,r≠"output" ∨ a.saved.input.numel≤ i →
        Native.observe I S original r i=(memory r i).bits ∧
        Native.observe I S commuted r i=(memory r i).bits) := by
  obtain ⟨original,commuted,ra,rb,equal,frame⟩ := NativeGrid.whole .backward R I S
    a.saved.M a.saved.N a.saved.K BM (Grid.blockColumns a.saved.N) (block_positive a.saved.N) memory
  refine ⟨original,commuted,ra,rb,equal,?_⟩
  intro r i outside
  apply frame r i
  simpa only [input_numel a.saved h.2.2.2.2] using outside

#axiomsClean forward_kernel_correctness
#axiomsClean backward_kernel_correctness
#axiomsClean forward_fp32_grid
#axiomsClean backward_fp32_grid
#axiomsClean forward_tensor_correctness
#axiomsClean backward_tensor_correctness
#axiomsClean kernel_native_access
#axiomsClean forward_python_checks
#axiomsClean backward_python_checks
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogSoftmax.TensorBridge
