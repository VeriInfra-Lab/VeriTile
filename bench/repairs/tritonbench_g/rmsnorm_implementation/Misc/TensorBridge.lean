import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.RawCorrect
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.TensorSource
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.KernelValues

/-! Optional actual Tensor applications of the two primary Real kernel
specifications and correspondence with the Python input checks. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.TensorBridge
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation
noncomputable section

/-- The same successful canonical kernel, now bound to actual Tensor objects.
Every logical output and every physical cell outside this program's row are
covered, including storage-offset views and singleton zero strides. -/
theorem tensor_kernel_correctness (x w out : TensorInfo) (eps : EpsilonInfo)
    (p : Parameters) (base : Nat → Nat) (binding : Binding x w out eps p base)
    (views : TensorBinding x w out p base) (before : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (_separation : Whole.Separation p) (unique : Whole.OutputUnique p) :
    Spec.Real (∃ final,
      exec (TensorSource.source p).eraseDType.toAlgKernel (TensorSource.realState p before b m)=some final ∧
      (∀ k,k<p.width → tensorRead (TensorSource.result p before final) views.output (flatIndex p b m k)=
        mathematical p.width p.eps
          (fun j=>tensorRead before views.input (flatIndex p b m j)) (tensorRead before views.weight) k) ∧
      (∀ allocation offset,(∀ k,k<p.width →
        ¬(allocation=p.output.allocation ∧ offset=Whole.outputOffset p b m k)) →
        TensorSource.result p before final allocation offset=before allocation offset) ∧
      (∀ k,k<p.width →
        views.input.storage.address (views.input.physicalIndex (flatIndex p b m k))=
          views.input.dataPtr.toNat+4*(b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2) ∧
        views.weight.storage.address (views.weight.physicalIndex k)=
          views.weight.dataPtr.toNat+4*(k*p.weights.stride 0) ∧
        views.output.storage.address (views.output.physicalIndex (flatIndex p b m k))=
          views.output.dataPtr.toNat+4*(b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2))) := by
  obtain ⟨final,ran,values,frame⟩ := RawCorrect.rmsnorm_kernel_correctness
    (p.input.stride 0) (p.input.stride 1) (p.input.stride 2) (p.weights.stride 0)
    (p.output.stride 0) (p.output.stride 1) (p.output.stride 2) p.width p.eps
    (TensorSource.realState p before b m)
    (fun j=>before p.input.allocation (Whole.inputOffset p b m j))
    (fun j=>before p.weights.allocation (Whole.weightOffset p j))
    (fun k _=>TensorSource.input_read p before b m k)
    (fun k _=>TensorSource.weight_read p before b m k)
    (TensorSource.unique p before b m hb hm unique)
  refine ⟨final,ran,?_,?_,?_⟩
  · intro k hk
    rw [views.output_read _ b m k hb hm hk,TensorSource.result_read]
    have value := values ⟨k,hk⟩
    rw [TensorSource.output_address] at value
    rw [value]
    exact TensorSource.mathematical_congr p.width p.eps _ _ _ _ k hk
      (fun j hj=>(views.input_read before b m j hb hm hj).symm)
      (fun j hj=>(views.weight_read before j hj).symm)
  · apply TensorSource.result_frame p before final b m
    intro address outside
    exact frame "out" address (Or.inr (by simpa only [TensorSource.output_address] using outside))
  · intro k hk
    exact ⟨views.input_pointer binding b m k hb hm hk,views.weight_pointer binding k hk,
      views.output_pointer binding b m k hb hm hk⟩

theorem tensor_index_commuted_correctness (x w out : TensorInfo) (eps : EpsilonInfo)
    (p : Parameters) (base : Nat → Nat) (binding : Binding x w out eps p base)
    (views : TensorBinding x w out p base) (before : Memory ℝ) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (_separation : Whole.Separation p) (unique : Whole.OutputUnique p) :
    Spec.Real (∃ final,
      exec (TensorSource.commutedSource p).eraseDType.toAlgKernel (TensorSource.realState p before b m)=some final ∧
      (∀ k,k<p.width → tensorRead (TensorSource.result p before final) views.output (flatIndex p b m k)=
        mathematical p.width p.eps
          (fun j=>tensorRead before views.input (flatIndex p b m j)) (tensorRead before views.weight) k) ∧
      (∀ allocation offset,(∀ k,k<p.width →
        ¬(allocation=p.output.allocation ∧ offset=Whole.outputOffset p b m k)) →
        TensorSource.result p before final allocation offset=before allocation offset) ∧
      (∀ k,k<p.width →
        views.input.storage.address (views.input.physicalIndex (flatIndex p b m k))=
          views.input.dataPtr.toNat+4*(b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2) ∧
        views.weight.storage.address (views.weight.physicalIndex k)=
          views.weight.dataPtr.toNat+4*(k*p.weights.stride 0) ∧
        views.output.storage.address (views.output.physicalIndex (flatIndex p b m k))=
          views.output.dataPtr.toNat+4*(b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2))) := by
  obtain ⟨final,ran,rest⟩ := tensor_kernel_correctness x w out eps p base binding views before b m hb hm _separation unique
  exact ⟨final,(TensorSource.commuted_run p _).trans ran,rest⟩

theorem python_checks_iff_preconditions (x w : Argument) (eps : EpsilonInfo) :
    checkInput x w eps = .ok () ↔ Admitted x w eps := checkInput_iff x w eps


open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole NativeKernel in
/-- Retained complete native Tensor application. Its values/frame use the same
Whole.row_value and row_frame certificates as NativeInterface.observed_kernel;
Tensor binding supplies the physical offsets and logical input correspondence. -/
theorem observed_tensor_kernel (x w out : TensorInfo) (eps : EpsilonInfo) (p : Parameters)
    (base : Nat → Nat) (binding : Binding x w out eps p base)
    (views : TensorBinding x w out p base) (ops : Primitives) (red : Reduction)
    (sep : Separation p) (unique : OutputUnique p) (b m : Nat)
    (hb : b<p.batches) (hm : m<p.rows) (before : Memory Word) (undefinedWeight : Nat → Word) :
    ∃ final, final=executeRow (numerics ops) red p b m undefinedWeight before ∧
      (∀ k,k<p.width → tensorRead final views.output (flatIndex p b m k)=
        reference ops red p.width p.eps
          (fun j=>tensorRead before views.input (flatIndex p b m j)) (tensorRead before views.weight) k) ∧
      (∀ allocation offset,(∀ k,k<p.width →
        ¬(allocation=p.output.allocation ∧ offset=outputOffset p b m k)) →
        final allocation offset=before allocation offset) ∧
      (∀ k,k<p.width →
        views.input.storage.address (views.input.physicalIndex (flatIndex p b m k))=
          views.input.dataPtr.toNat+4*(b*p.input.stride 0+m*p.input.stride 1+k*p.input.stride 2) ∧
        views.weight.storage.address (views.weight.physicalIndex k)=
          views.weight.dataPtr.toNat+4*(k*p.weights.stride 0) ∧
        views.output.storage.address (views.output.physicalIndex (flatIndex p b m k))=
          views.output.dataPtr.toNat+4*(b*p.output.stride 0+m*p.output.stride 1+k*p.output.stride 2)) := by
  exact NativeKernel.tensor_row x w out eps p base binding views ops red sep unique b m hb hm before undefinedWeight

#axiomsClean observed_tensor_kernel

#axiomsClean tensor_kernel_correctness
#axiomsClean tensor_index_commuted_correctness
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.TensorBridge
