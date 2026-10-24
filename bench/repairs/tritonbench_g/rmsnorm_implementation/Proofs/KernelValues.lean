import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.TensorBinding
import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.NativePrimitives

namespace VeriTile.Bench.Repairs.RmsnormImplementation.NativeKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section

/-- Every output word of one full two-pass program and its entire physical
memory frame. The scalar primitives are total on all FP32 payloads, including
NaNs and infinities. No finiteness premise or numerical rule is required. -/
theorem tensor_row (x w out : TensorInfo) (eps : EpsilonInfo) (p : Parameters)
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
  refine ⟨_,rfl,?_,?_,?_⟩
  · intro k hk
    rw [views.output_read _ b m k hb hm hk,row_value (numerics ops) red p sep unique b m hb hm undefinedWeight before k hk]
    exact value_congr (numerics ops) red p.width p.eps _ _ _ _ k hk
      (fun j hj=>(views.input_read before b m j hb hm hj).symm)
      (fun j hj=>(views.weight_read before j hj).symm)
  · exact row_frame (numerics ops) red p b m undefinedWeight before
  · intro k hk
    exact ⟨views.input_pointer binding b m k hb hm hk,views.weight_pointer binding k hk,
      views.output_pointer binding b m k hb hm hk⟩

#specNonCircular reference avoiding [executeRow,outputPass,varianceStep]
#axiomsClean tensor_row
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.NativeKernel
