import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Physical
import bench.repairs.tritonbench_g.swiglu_backward.Proofs.Bytes
import bench.repairs.tritonbench_g.swiglu_backward.Correct
import bench.repairs.tritonbench_g.swiglu_backward.FPEquiv

namespace VeriTile.Bench.Repairs.SwigluBackward.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole
noncomputable section

/-- The complete selected kernel grid succeeds for every legal configuration
and order. All three effective outputs equal the independent analytic VJP or
forward formula; exterior physical memory is unchanged. Logical Tensor reads
are tied to the actual finite pointers used by the source. -/
theorem swiglu_backward_kernel_correct (p : Prepared ℝ call entry preexisting)
    (storage : Storage p) (shapes : checks p = Bool.true)
    (admitted : checkLaunch p storage.byteBase = Bool.true)
    (config : Config) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid p.layout config.block)) :
    VeriTile.Spec.Real (∃ final,
      KernelProof.run realAlgebra p config tiles p.state = some final ∧
      KernelProof.PhysicalResult formula p storage final ∧
      (∀ region a, WholeOutside p.layout call.recompute region a →
        final.mem region a = p.state.mem region a) ∧
      KernelProof.InputBinding p storage ∧ CheckedIndices p storage) := by
  obtain ⟨final,ran,result,indices⟩ := KernelProof.success realAlgebra p storage shapes admitted config tiles schedule
  have math : scalar realAlgebra = formula := by funext ch x y d; exact scalar_real ch x y d
  have h := (checkLaunch_iff _ _).mp admitted
  refine ⟨final,ran,?_,result.2.1,KernelProof.input_binding p storage shapes h,indices⟩
  simpa only [math] using KernelProof.physical_result (scalar realAlgebra) p storage shapes h final result

#axiomsClean swiglu_backward_kernel_correct
#guard_msgs (drop info) in
#specNonCircular formula avoiding [kernel,KernelProof.run,KernelProof.physicalCommit]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.TensorBridge


namespace VeriTile.Bench.Repairs.SwigluBackward.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward
open Whole
noncomputable section

/-- Successful complete-source execution equals an independently specified
ordered native graph on every output bit pattern and preserves exterior memory.
The native compiler graph and scalar primitives are explicit model boundaries;
they are not deferred numerical identities. Native FMA, operand order, casts,
sigmoid and both recomputation branches remain intact. -/
theorem swiglu_backward_kernel_fp32 (M : Observed.NativeModel (BitVec 32))
    (p : Prepared (Observed.Term (BitVec 32)) call (Observed.entryGraph entry) preexisting)
    (storage : Storage p) (shapes : checks p = Bool.true)
    (admitted : checkLaunch p storage.byteBase = Bool.true)
    (byteEntry : Nat → Nat → BitVec 8)
    (bytes : KernelProof.ByteInputBinding M p storage byteEntry)
    (config : Config) (tiles : List (Nat × Nat))
    (schedule : tiles.Perm (grid p.layout config.block)) :
    ∃ final, KernelProof.run Observed.graphAlgebra p config tiles p.state = some final ∧
      KernelProof.NativeResult M p storage final ∧
      KernelProof.NativeBytes M p storage final byteEntry ∧
      KernelProof.LogicalInputBytes p storage byteEntry ∧
      (∀ region a, WholeOutside p.layout call.recompute region a →
        final.mem region a = p.state.mem region a) ∧
      KernelProof.InputBinding p storage ∧ CheckedIndices p storage := by
  obtain ⟨final,ran,result,indices⟩ := KernelProof.success Observed.graphAlgebra p storage shapes admitted config tiles schedule
  have h := (checkLaunch_iff _ _).mp admitted
  have native := KernelProof.native_result M p storage final
    (KernelProof.physical_result _ p storage shapes h final result)
  exact ⟨final,ran,native,KernelProof.native_bytes M p storage h final byteEntry native,
    KernelProof.logical_input_bytes M p storage shapes h byteEntry bytes,
    result.2.1,KernelProof.input_binding p storage shapes h,indices⟩

#axiomsClean swiglu_backward_kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluBackward.TensorBridge
