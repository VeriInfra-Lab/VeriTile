import bench.repairs.tritonbench_g.softmax_triton2.Correct
import bench.repairs.tritonbench_g.softmax_triton2.Misc.RealApplication
import bench.repairs.tritonbench_g.softmax_triton2.Misc.NativeCorrect
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.Native

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.TensorBridge
noncomputable section

section Mathematical
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Every valid row of the actual Tensor computes the independent unshifted
softmax. The input binding uses original backing storage, and the frame covers
all cells outside the active output row. No wrapper execution is a premise. -/
theorem softmax_checked_kernel_correctness
    (x : Object) (h : InputContract x.metadata) (out : RegionName) (ob : Nat)
    (read : Nat → ℝ) (s : BlockState) (binding : InputMemory RealProof.access read x s)
    (pid : s.pid < rows x.metadata) :
    Spec.Real (∃ t,
      exec (SoftmaxTriton2.softmax_kernel out x.region x.tensor.storageOffset ob
        (cols x.metadata) 1 (cols x.metadata) 1 (cols x.metadata)
        (blockSize (cols x.metadata))).toAlgKernel s = some t ∧
      (∀ c, c < cols x.metadata →
        t.readMem out (ob+s.pid*cols x.metadata+c) =
          reference (cols x.metadata) (logicalValue Neg.neg read x s.pid) c) ∧
      (∀ r a, (∀ c, c < cols x.metadata →
        ¬(out=r ∧ ob+s.pid*cols x.metadata+c=a)) → t.mem r a=s.mem r a)) :=
  RealProof.checked x h out ob read s binding pid

/-- The exact source pointer, its active load, the independent logical value,
and allocation bounds all name the same actual Tensor storage cell. -/
theorem real_input_binding
    (x : Object) (h : InputContract x.metadata) (read : Nat → ℝ)
    (s : BlockState) (binding : InputMemory RealProof.access read x s)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    RealProof.access.read s x.region ((descriptor x).index r c) =
        read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat ∧
      RealProof.access.read s x.region ((descriptor x).index r c) = logicalValue Neg.neg read x r c ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat =
        x.tensor.storage.address (x.tensor.physicalIndex (r * cols x.metadata + c)) ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat + 4 ≤
        x.tensor.storage.base + 4 * x.tensor.storage.elements ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat % 4 = 0 :=
  Binding.active_load RealProof.access Neg.neg read x h s binding r c hr hc

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x=.ok () ↔ AdmittedArgument x := checkInput_iff x

#axiomsClean softmax_checked_kernel_correctness
#axiomsClean real_input_binding
#axiomsClean python_checks_iff_preconditions
end Mathematical

section Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous NativeGraph

/-- Successful execution of this task's typed source, every output column,
and equality of every full memory cell outside the active output row.
All raw FP32 bit patterns are admitted, including exceptional values. -/
theorem softmax_checked_fp32 (inst : Instructions)
    (x : Object) (h : InputContract x.metadata) (y : Tensor)
    (shape : SameShape (descriptor x) y) (injective : y.Injective)
    (p : Nat) (hp : p < rows x.metadata)
    (read : Nat → Word) (s : State Word) (binding : InputMemory Floating.access read x s) :
    ∃ t, NativeProof.evaluate inst (descriptor x) y (blockSize (cols x.metadata)) p s = some t ∧
      (∀ c, c < cols x.metadata →
        (t.mem y.region (y.index p c)).read .real =
          FPEquiv.reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue NativeProof.negateWord read x p) c) ∧
      FrameRows Floating.access y [p] s t :=
  NativeProof.checked inst x h y shape injective p hp read s binding

/-- Both input and output are actual Tensor objects. Output values are read
at the original logical-to-physical indices; the exterior frame includes the
prefix/suffix and every other row of the output backing allocation. -/
theorem softmax_tensor_fp32 (inst : Instructions)
    (x y : Object) (hx : InputContract x.metadata) (hy : InputContract y.metadata)
    (shape : SameShape (descriptor x) (descriptor y))
    (p : Nat) (hp : p < rows x.metadata)
    (read : Nat → Word) (s : State Word) (binding : InputMemory Floating.access read x s) :
    ∃ t, NativeProof.evaluate inst (descriptor x) (descriptor y)
        (blockSize (cols x.metadata)) p s = some t ∧
      (∀ c, c < cols x.metadata →
        (t.mem y.region (y.tensor.physicalIndex (p * cols y.metadata + c))).read .real =
          FPEquiv.reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue NativeProof.negateWord read x p) c) ∧
      FrameRows Floating.access (descriptor y) [p] s t := by
  obtain ⟨t,run,values,frame⟩ := softmax_checked_fp32 inst x hx (descriptor y)
    shape (descriptor_injective y) p hp read s binding
  refine ⟨t,run,?_,frame⟩
  intro c hc
  have hr : rows y.metadata = rows x.metadata := shape.rows
  have hn : cols y.metadata = cols x.metadata := shape.cols
  rw [←Binding.physical_index y hy p c (by rwa [hr]) (by rwa [hn])]
  exact values c hc

/-- Actual source loads use the reader at the widened sequential pointer;
logical Tensor indices and the active four-byte storage bounds agree. -/
theorem native_input_binding
    (x : Object) (h : InputContract x.metadata) (read : Nat → Word)
    (s : State Word) (binding : InputMemory Floating.access read x s)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < cols x.metadata) :
    Floating.access.read s x.region ((descriptor x).index r c) =
        read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat ∧
      Floating.access.read s x.region ((descriptor x).index r c) = logicalValue NativeProof.negateWord read x r c ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat =
        x.tensor.storage.address (x.tensor.physicalIndex (r * cols x.metadata + c)) ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat + 4 ≤
        x.tensor.storage.base + 4 * x.tensor.storage.elements ∧
      (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat % 4 = 0 :=
  Binding.active_load Floating.access NativeProof.negateWord read x h s binding r c hr hc

#axiomsClean softmax_checked_fp32
#axiomsClean softmax_tensor_fp32
#axiomsClean native_input_binding
end Native

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.TensorBridge
