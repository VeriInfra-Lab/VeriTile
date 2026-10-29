import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelPhysical

/-! Optional Python-check correspondence. The complete mathematical and native
kernel specifications remain in Correct.lean and FPEquiv.lean. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open KernelDomain KernelBinding KernelModel KernelPhysical KernelAddresses
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- The pre-allocation checks admit exactly builtin nonnegative integers. -/
theorem python_checks_iff_preconditions (a : Contract.Arguments) :
    (Contract.checkInputs a).isSome=Bool.true ↔ Contract.InputContract a :=
  Contract.python_checks_iff_preconditions a

/-- Separate successful remainder/assert stage, after the three allocations.
Zero blocks produce the original modulo errors; they are not new entry guards. -/
theorem kernel_checks_iff_preconditions (a : Arguments) :
    checkKernel a=Bool.true ↔ KernelContract a := checkKernel_iff a

/-- The independent formula is the full tensor dot product. The conclusion
includes complete source and finite-int64 execution, actual output coordinates,
actual pointer/access binding and tagged/physical memory frames. The physical
Real heap is the mathematical interpretation, not rounded native output. -/
theorem batched_vecmat_kernel_real (decode : ValueProofs.Word → ℝ)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector)
    (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (binding : Binding a x y z) (heap : Heap) (scratch : State ValueProofs.Word) :
    Spec.Real (AddressBinding a x y z heap ∧
      Realizes a x y z heap (inputMemory (x.value heap) (y.value heap) (allocatedScratch z heap scratch).mem) ∧
      ∃ final, SourceGraph.runGrid rest (KernelProfile.toPlans profile) (parameters a)
        (initial (x.value heap) (y.value heap) (allocatedScratch z heap scratch))=some final ∧
      FiniteExecution.grid (KernelProfile.toPlans profile) (parameters a)
        (initial (x.value heap) (y.value heap) (allocatedScratch z heap scratch))=some final.mem ∧
      (∀ m : Fin a.M,∀ n : Fin a.N,
        z.observe (physical z (fun addr => decode (heap addr)) (realValues decode final)) (m.val*a.N+n.val)=
          mathematical decode a (x.value heap) (y.value heap) m.val n.val) ∧
      (∀ r i,SourceGraph.GridOutside (parameters a) r i →
        final.mem r i=(initial (x.value heap) (y.value heap) (allocatedScratch z heap scratch)).mem r i) ∧
      (∀ addr,addr<z.base ∨ z.base+z.count≤addr →
        physical z (fun addr => decode (heap addr)) (realValues decode final) addr=decode (heap addr)) ∧
      (∀ i,i<x.tensor.storage.elements → physical z (fun addr => decode (heap addr)) (realValues decode final)
        (x.tensor.storage.base/4+i)=decode (heap (x.tensor.storage.base/4+i))) ∧
      (∀ i,i<y.tensor.storage.elements → physical z (fun addr => decode (heap addr)) (realValues decode final)
        (y.tensor.storage.base/4+i)=decode (heap (y.tensor.storage.base/4+i)))) := by
  exact ⟨addresses a native x y z heap,initial_realizes a x y z heap scratch,
    KernelResult.real_correct decode rest (KernelProfile.toPlans profile) a h native x y z binding heap (allocatedScratch z heap scratch)⟩

/-- Every FP32 word is admitted. The independent operand-only graph is evaluated
using FP.mul/add/fma; no arbitrary local dot operator or PrimitiveLaw premise
stands in for execution. Source, finite indices and actual Tensor memory are
connected explicitly. The consumed profile sees payload-erased source structure;
input bits, NaNs and intermediate FP32 values cannot select a different tree.
This does not certify arbitrary compiler/backend profiles. -/
theorem batched_vecmat_kernel_native (cfg : FP.Config)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector)
    (a : Arguments) (h : KernelContract a) (native : LaunchABI a)
    (x : Allocated [a.M,a.K]) (y : Allocated [a.M,a.N,a.K]) (z : Allocated [a.M,a.N])
    (binding : Binding a x y z) (heap : Heap) (scratch : State ValueProofs.Word) :
    AddressBinding a x y z heap ∧
      Realizes a x y z heap (inputMemory (x.value heap) (y.value heap) (allocatedScratch z heap scratch).mem) ∧
      ∃ final, SourceGraph.runGrid rest (KernelProfile.toPlans profile) (parameters a)
        (initial (x.value heap) (y.value heap) (allocatedScratch z heap scratch))=some final ∧
      FiniteExecution.grid (KernelProfile.toPlans profile) (parameters a)
        (initial (x.value heap) (y.value heap) (allocatedScratch z heap scratch))=some final.mem ∧
      (∀ m : Fin a.M,∀ n : Fin a.N,
        z.value (physical z heap (nativeValues cfg final)) (m.val*a.N+n.val)=
          nativeTarget cfg (KernelProfile.toPlans profile) a (x.value heap) (y.value heap) m.val n.val) ∧
      (∀ r i,SourceGraph.GridOutside (parameters a) r i →
        SourceGraph.Native.decodeMemory cfg final.mem r i=
          inputMemory (x.value heap) (y.value heap) (allocatedScratch z heap scratch).mem r i) ∧
      (∀ addr,addr<z.base ∨ z.base+z.count≤addr →
        physical z heap (nativeValues cfg final) addr=heap addr) ∧
      (∀ i,i<x.tensor.storage.elements → physical z heap (nativeValues cfg final)
        (x.tensor.storage.base/4+i)=heap (x.tensor.storage.base/4+i)) ∧
      (∀ i,i<y.tensor.storage.elements → physical z heap (nativeValues cfg final)
        (y.tensor.storage.base/4+i)=heap (y.tensor.storage.base/4+i)) ∧
      Realizes a x y z (physical z heap (nativeValues cfg final))
        (SourceGraph.Native.decodeMemory cfg final.mem) := by
  obtain ⟨final,he,hfinite,hc,hf,hp,ha,hb⟩ := KernelResult.native_correct rest cfg (KernelProfile.toPlans profile) a h native
    x y z binding heap (allocatedScratch z heap scratch)
  exact ⟨addresses a native x y z heap,initial_realizes a x y z heap scratch,
    final,he,hfinite,hc,hf,hp,ha,hb,final_realizes cfg a x y z binding heap scratch final hf
      (output_typed rest (KernelProfile.toPlans profile) a h native x y z heap (allocatedScratch z heap scratch) final he)⟩

#axiomsClean batched_vecmat_kernel_real
#axiomsClean batched_vecmat_kernel_native
#axiomsClean python_checks_iff_preconditions
#axiomsClean kernel_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.TensorBridge
