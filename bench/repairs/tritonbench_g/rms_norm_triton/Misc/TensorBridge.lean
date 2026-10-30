import bench.repairs.tritonbench_g.rms_norm_triton.Correct
import bench.repairs.tritonbench_g.rms_norm_triton.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.KernelPublic

namespace VeriTile.Bench.Repairs.RmsNormTriton.TensorBridge.Real
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural
noncomputable section

/-- Checked repair: successful execution of the unchanged typed JIT, every
logical output of the actual Tensor, and equality of complete physical memory
to the independent RMSNorm output update. Strided/broadcast/offset originals
are admitted through the existing ordinary contiguous-copy law. -/
theorem kernel_correct
    (x w xc wc y : ActualTensor) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x.info w.info n eps)
    (a : Whole.Prepared ℝ input weight entry live)
    (binding : KernelTensorBinding a x w xc wc y)
    (api : PreparedAPI a n.dimensions) (checked : PointerContract x.info n.dimensions api.pointers)
    (physical : KernelStorage api x w xc wc y) (schedule : Whole.Schedule input n.dimensions) :
    Spec.Real (checkTensorInput x.info w.info n eps = .ok () ∧
      checkPointers x.info n.dimensions ⟨xc.tensor.dataPtr,wc.tensor.dataPtr,y.tensor.dataPtr⟩ = .ok () ∧
      Whole.IndexCorrespondence a n.dimensions (api.finite x.info w.info n eps c binding.entry a checked) ∧
      Whole.ActualSourceAddresses a n.dimensions (api.finite x.info w.info n eps c binding.entry a checked)
        xc.view wc.view y.view ∧
      ∃ final, a.entryRun Whole.realAlgebra n.dimensions eps.value schedule = some final ∧
        Whole.KernelResult (mathematicalResult n.dimensions eps.value x w entry)
          y.view a.output.memory (a.commit final) ∧
        (∀ k, k < y.tensor.metadata.numel → y.read (a.commit final) k =
          mathematicalResult n.dimensions eps.value x w entry k) ∧
        a.commit final = Whole.referenceMemory (mathematicalResult n.dimensions eps.value x w entry)
          y.view a.output.memory) := by
  let finite := api.finite x.info w.info n eps c binding.entry a checked
  refine ⟨(checkTensorInput_iff _ _ _ _).2 c,by rw [← physical.pointers]; exact (checkPointers_iff _ _ _).2 checked,
    Whole.index_correspondence a n.dimensions finite,
    Whole.actual_source_addresses a binding.views (binding.entry.domain n eps c) finite physical.storage,?_⟩
  obtain ⟨final,ran,result⟩ := logical_tensor_result a Whole.realAlgebra x w xc wc y n eps c binding schedule
  simp only [Whole.rowValue_real,Whole.Prepared.layout] at result
  exact ⟨final,ran,result,fun k hk => (y.read_view (a.commit final) k hk).symm.trans (result.1 k hk),
    Whole.kernelResult_memory binding.views.output a.output.contiguous result⟩

#guard_msgs (drop info) in
#specNonCircular mathematicalResult avoiding [kernel,Whole.rowValue,Whole.sourceRun,Whole.Prepared.entryRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.TensorBridge.Real

namespace VeriTile.Bench.Repairs.RmsNormTriton.TensorBridge.Native
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural
open Whole Observed
noncomputable section
set_option maxHeartbeats 6000000

/-- Independent native instruction recurrence on actual logical input words.
Runtime epsilon, primitive IEEE/FMA/conversion semantics and compiler reduction
trees are model boundaries, never numerical experiment assumptions. -/
def nativeResult (M : NativeModel (BitVec 32)) (profile : ReductionFamily)
    (n : List Nat) (x w : ActualTensor) (entry : Memory (BitVec 32)) (k : Nat) : BitVec 32 :=
  reference M profile (columns n) (block (columns n))
    (fun j => x.read entry ((k/columns n)*columns n+j)) (w.read entry) (k%columns n)

/-- Successful source graph execution and all-memory equivalence to the
independent native recurrence, including N=1 fused x*x+runtime-epsilon. The
actual checked epsilon word is explicitly connected to the native model. -/
theorem kernel_native_equivalence
    (x w xc wc y : ActualTensor) (n : NormalizedInfo) (eps : EpsilonInfo)
    (c : InputContract x.info w.info n eps)
    (M : NativeModel (BitVec 32)) (epsilonArgument : M.epsilon=eps.runtimeWord)
    (profile : ReductionFamily)
    (a : Prepared (Term (BitVec 32)) input weight (entryGraph entry) live)
    (binding : KernelTensorBinding a x w xc wc y)
    (api : PreparedAPI a n.dimensions) (checked : PointerContract x.info n.dimensions api.pointers)
    (physical : KernelStorage api x w xc wc y) (schedule : Schedule input n.dimensions)
    (initial : Memory (BitVec 32)) (heap : a.output.memory=entryGraph initial) :
    checkTensorInput x.info w.info n eps=.ok () ∧
    checkPointers x.info n.dimensions ⟨xc.tensor.dataPtr,wc.tensor.dataPtr,y.tensor.dataPtr⟩=.ok () ∧ M.epsilon=eps.runtimeWord ∧
    IndexCorrespondence a n.dimensions (api.finite x.info w.info n eps c binding.entry a checked) ∧
    ActualSourceAddresses a n.dimensions (api.finite x.info w.info n eps c binding.entry a checked)
      xc.view wc.view y.view ∧
    ∃ final, a.entryRun graphAlgebra n.dimensions eps.value schedule=some final ∧
      KernelResult (nativeResult M profile n.dimensions x w entry) y.view initial
        (projectMemory M profile (columns n.dimensions) a.output.tensor.base (a.commit final)) ∧
      (∀ k, k < y.tensor.metadata.numel →
        y.read (projectMemory M profile (columns n.dimensions) a.output.tensor.base (a.commit final)) k =
          nativeResult M profile n.dimensions x w entry k) ∧
      projectMemory M profile (columns n.dimensions) a.output.tensor.base (a.commit final) =
        referenceMemory (nativeResult M profile n.dimensions x w entry) y.view initial := by
  let finite := api.finite x.info w.info n eps c binding.entry a checked
  refine ⟨(checkTensorInput_iff _ _ _ _).2 c,by rw [← physical.pointers]; exact (checkPointers_iff _ _ _).2 checked,epsilonArgument,
    index_correspondence a n.dimensions finite,
    actual_source_addresses a binding.views (binding.entry.domain n eps c) finite physical.storage,?_⟩
  obtain ⟨final,ran,result⟩ := logical_tensor_result a graphAlgebra x w xc wc y n eps c binding schedule
  have projected : KernelResult (nativeResult M profile n.dimensions x w entry) y.view initial
      (projectMemory M profile (columns n.dimensions) a.output.tensor.base (a.commit final)) := by
    constructor
    · intro k hk
      have offset := a.output_offset binding.views k hk
      have lane : (y.view.offset k-a.output.tensor.base)%columns n.dimensions=k%columns n.dimensions := by
        rw [offset,Nat.add_sub_cancel_left]
      have value := congrArg (interpret M profile (k%columns n.dimensions)) (result.1 k hk)
      rw [show w.read (entryGraph entry) = (fun j => Term.input (w.read entry j)) from rfl] at value
      change interpret M profile ((y.view.offset k-a.output.tensor.base)%columns n.dimensions)
        (a.commit final y.view.allocation (y.view.offset k)) = _
      rw [lane]
      exact value.trans (source_graph_lowering M profile (a.layout n.dimensions) eps.value
        (fun j => x.read entry ((k/columns n.dimensions)*columns n.dimensions+j))
        (w.read entry) (k%columns n.dimensions))
    · intro allocation offset outside
      have unchanged := result.2 allocation offset outside
      rw [heap] at unchanged
      exact congrArg (interpret M profile ((offset-a.output.tensor.base)%columns n.dimensions)) unchanged
  exact ⟨final,ran,projected,fun k hk => (y.read_view _ k hk).symm.trans (projected.1 k hk),
    kernelResult_memory binding.views.output a.output.contiguous projected⟩

#guard_msgs (drop info) in
#specNonCircular nativeResult avoiding [kernel,lower,interpret,rowValue,sourceRun,Prepared.entryRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.TensorBridge.Native
