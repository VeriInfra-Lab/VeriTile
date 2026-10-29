import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.KernelStages
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Boundary
import VeriTile.Torch.FP32

/-! Actual Tensor storage bindings and byte observations for the kernel result.
Tensor API representation laws are model boundaries. No allocator, wrapper
execution, computed value or successful source run is assumed by the binding.
-/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.KernelTensor
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
open KernelProof SourceReader Admission

/-- Existing descriptor fields are observations of this actual FP32 Tensor.
The ordinary flag includes the Python no-neg/no-conj checks; shared FP32
logical reads need only its no-neg consequence. Storage offsets stay arbitrary.
-/
structure Binding (d : Admission.Tensor) (t : VeriTile.Torch.FP32.Tensor) : Prop where
  laws : VeriTile.Torch.FP32.TensorLaws t
  object : d.object = t.object
  dtype : d.dtype = .fp32
  layout : d.strided = t.metadata.isStrided
  cuda : d.cuda = t.metadata.isCuda
  device : d.device = t.metadata.device.index
  shape : d.shape = t.metadata.shape
  pointer : d.base = t.dataPtr.toNat
  contiguous : d.contiguous = t.metadata.isContiguous
  ordinary : d.ordinary = Bool.true → t.metadata.isNeg = Bool.false

theorem element_count (binding : Binding d t) : d.elements = t.metadata.numel := by
  simp [Admission.Tensor.elements, VeriTile.Torch.TensorMetadata.numel,
    binding.shape,List.prod_eq_foldl]

theorem tensor_flags (h : tensorCheck d .fp32 device = Bool.true) :
    d.contiguous = Bool.true ∧ d.ordinary = Bool.true := by
  simp only [tensorCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  rcases h with ⟨⟨⟨⟨⟨_,_⟩,_⟩,hc⟩,ho⟩,_⟩
  exact ⟨hc,ho⟩

def wordReader (d : Admission.Tensor) (values : Nat → VeriTile.Torch.FP32.Word) (address : Nat) : VeriTile.Torch.FP32.Word :=
  values ((address-d.base)/4)

theorem shared_read (binding : Binding d t) (checked : tensorCheck d .fp32 device = Bool.true)
    (values : Nat → VeriTile.Torch.FP32.Word) (i : Nat) (hi : i < d.elements) :
    VeriTile.Torch.FP32.logicalValue (wordReader d values) t binding.laws i = values i := by
  have flags := tensor_flags checked
  have hc : t.metadata.isContiguous = Bool.true := binding.contiguous.symm.trans flags.1
  have hn : t.metadata.isNeg = Bool.false := binding.ordinary flags.2
  have hi' : i < t.metadata.numel := by simpa only [element_count binding] using hi
  let view := VeriTile.Torch.FP32.dataView t binding.laws hc
  have hv : view.base = d.base := by
    have ptr := congrArg BitVec.toNat (VeriTile.Torch.FP32.actual_pointer t binding.laws hc (by omega))
    have span := view.spanFits
    have bound : view.base < 2^64 := by
      change view.base + 4*t.metadata.numel ≤ 2^64 at span
      omega
    simp only [VeriTile.Torch.FP32.DataView.address, Nat.mul_zero, Nat.add_zero,
      BitVec.toNat_ofNat] at ptr
    change view.base % 2^64 = t.dataPtr.toNat at ptr
    rw [Nat.mod_eq_of_lt bound,←binding.pointer] at ptr
    exact ptr
  rw [←VeriTile.Torch.FP32.direct_read_eq_logical (wordReader d values) t binding.laws hc hn i hi']
  change wordReader d values (view.base+4*i) = values i
  rw [hv]
  simp [wordReader]

def SharedReads (p c : Admission.Tensor) (pt ct : VeriTile.Torch.FP32.Tensor)
    (pb : Binding p pt) (cb : Binding c ct) : Prop :=
  (∀ i, i<p.elements → VeriTile.Torch.FP32.logicalValue (wordReader p (fun j => ⟨p.word j⟩)) pt pb.laws i = ⟨p.word i⟩) ∧
  (∀ i, i<c.elements → VeriTile.Torch.FP32.logicalValue (wordReader c (fun j => ⟨c.word j⟩)) ct cb.laws i = ⟨c.word i⟩)

theorem shared_reads (ok : check s p c l = Bool.true)
    (pb : Binding p pt) (cb : Binding c ct) : SharedReads p c pt ct pb cb :=
  ⟨fun i hi => shared_read pb (checked_parts ok).2.1 _ i hi,
   fun i hi => shared_read cb (checked_parts ok).2.2.1 _ i hi⟩

/-- The actual caller Tensor observes the same final word encoded by `heap`.
The target is independent of the source interpreter and its traces. -/
def NativeResult (s : Scalars) (p c l : Admission.Tensor) (ct : VeriTile.Torch.FP32.Tensor)
    (cb : Binding c ct) (stage : Phased.Stage) (before : ByteHeap)
    (result : List SourceEvent × Wide.State (FP.Value .fp32)) : Prop :=
  (∀ i, i<c.elements →
    VeriTile.Torch.FP32.logicalValue (wordReader c (fun j => result.2.values.mem.output j)) ct cb.laws i =
      outputTarget fp32 s.args stage (rawFP p c l) i) ∧
  (∀ i j, i<c.elements → j<4 →
    heap before p c l result.2.values.mem (c.base+4*i+j) =
      byteOf (outputTarget fp32 s.args stage (rawFP p c l) i).bits j) ∧
  (∀ a, Untouched (result.1.map project) p c l a →
    heap before p c l result.2.values.mem a = before a)

theorem byte_frame (ok : check s p c l = Bool.true) (before : ByteHeap)
    (input : InputBytes before p c l)
    (facts : Facts fp32 s.args (checkedBuffers ok) stage xs (rawFP p c l) result)
    (a : Nat) (untouched : Untouched (result.1.map project) p c l a) :
    heap before p c l result.2.values.mem a = before a := by
  unfold heap
  split_ifs with hp hc hl
  · rw [facts.partialFrame _ (untouched.1 hp)]
    simpa [rawFP,memory,slot] using (input.partials a hp).symm
  · rw [facts.outputFrame _ (untouched.2.1 hc)]
    simpa [rawFP,memory,slot] using (input.output a hc).symm
  · rw [facts.lockFrame _ (untouched.2.2 hl)]
    have hslot : (a-l.base)/4<l.elements := by
      have hh := hl.2
      have hb := hl.1
      apply (Nat.div_lt_iff_lt_mul (by omega : 0<4)).mpr
      omega
    simpa only [rawFP,memory,slot,Int.toNat_natCast,Int.natCast_nonneg,true_and,if_pos hslot]
      using (input.locks a hl).symm
  · rfl

theorem native_result (ok : check s p c l = Bool.true) (cb : Binding c ct)
    (before : ByteHeap) (input : InputBytes before p c l)
    (facts : Facts fp32 s.args (checkedBuffers ok) stage xs (rawFP p c l) result) :
    NativeResult s p c l ct cb stage before result := by
  refine ⟨?_,?_,?_⟩
  · intro i hi
    rw [shared_read cb (checked_parts ok).2.2.1 _ i hi,facts.output]
  · intro i j hi hj
    rw [heap_output_bytes ok before _ i j hi hj,facts.output]
  · exact byte_frame ok before input facts

#specNonCircular Binding avoiding [runKernel, SourceReader.runPlan, Wide.nativeRun, outputTarget]
#specNonCircular NativeResult avoiding [runKernel, SourceReader.runPlan, Wide.nativeRun]
#axiomsClean shared_read
#axiomsClean shared_reads
#axiomsClean native_result
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.KernelTensor
