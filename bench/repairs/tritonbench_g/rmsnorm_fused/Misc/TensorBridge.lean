import bench.repairs.tritonbench_g.rmsnorm_fused.Correct
import bench.repairs.tritonbench_g.rmsnorm_fused.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rmsnorm_fused.FPEquiv
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.KernelIndex

/-! Complete actual-Tensor applications of the raw source-grid result. Binding.execution
and KernelInterface.complete use the same Runtime source/value/frame proofs. -/
namespace VeriTile.Bench.Repairs.RmsnormFused.TensorBridge
open VeriTile Triton FP.Structural Binding Ordered
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra)
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- Successful execution of the repaired typed kernel, mathematical RMSNorm
on actual logical Tensor inputs, every output, and complete physical memory.
The FP32→Real idealization is separate from native bit equivalence. -/
theorem kernel_correct (x w y : Tensor) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (out : Output x w y)
    (memory : Nat → ℝ) (schedule : List Nat) (all : schedule.Perm (geometry x.info).grid) :
    Spec.Real (checkTensorInput x.info w.info eps=.ok () ∧ Index.Correspondence x w y ∧
      (∀ row col, ∀ hr : row<(geometry x.info).rows, ∀ hc : col<width x.info,
        (Index.pointer x.actual.dataPtr (width x.info) row col).toNat=x.view.address (row*width x.info+col) ∧
        (Index.pointer y.actual.dataPtr (width x.info) row col).toNat=y.view.address (row*width x.info+col)) ∧
      ∃ final,Runtime.run realAlgebra "X" "Y" "W" (geometry x.info).rows (width x.info)
        (block (width x.info)) eps.value schedule (initial x w y memory)=some final ∧
        (∀ k,k<y.size → y.read (commit y memory final) k=Ideal.result x w memory (width x.info) eps.value k) ∧
        commit y memory final=reference y memory (Ideal.result x w memory (width x.info) eps.value) ∧
        (∀ address,¬footprint y address → commit y memory final address=memory address) ∧
        (∀ (region : RegionName) offset,region≠"Y" ∨ x.size≤offset →
          final.mem region offset=(initial x w y memory).mem region offset)) := by
  refine ⟨(checkTensorInput_iff _ _ _).mpr c,Index.correspondence x w y eps c out,?_,?_⟩
  · intro row col hr hc
    have addresses := Index.source_addresses x w y eps c out row col hr hc
    exact ⟨addresses.1,addresses.2.1⟩
  · obtain ⟨final,ran,values,frame⟩ := execution realAlgebra x w y eps c eps.value memory schedule all
    have flat : ∀ k,k<y.size → Runtime.read final "Y" k=Ideal.result x w memory (width x.info) eps.value k := by
      intro k hk
      have coords := flat_coordinates x w eps c k (by rw [←out.size]; exact hk)
      have result := (values (k/width x.info) (k%width x.info) coords.1 coords.2.1).trans
        (Ideal.logical_value x w y memory eps c _ _ coords.1 coords.2.1)
      simpa only [coords.2.2] using result
    refine ⟨final,ran,?_,commit_reference y memory final _ flat,commit_frame y memory final,frame⟩
    intro k hk
    rw [y.read_view _ k hk,commit_value y memory final k hk]
    exact flat k hk

/-- Runtime scalar ABI, all pointer bytes, successful actual source execution,
all returned words, and complete physical byte equality with the independent
ordered native recurrence. IEEE/conversion/compiler choices are explicit local
model boundaries, not numerical experiments or whole-result assumptions. -/
theorem kernel_native_equivalence (x w y : Tensor) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (out : Output x w y)
    (I : Instructions) (P : Policy) (R : Reductions)
    (epsilonWord : Word) (abi : I.scalarABI eps.value=epsilonWord)
    (memory : Bytes.Memory) (schedule : List Nat) (all : schedule.Perm (geometry x.info).grid) :
    checkTensorInput x.info w.info eps=.ok () ∧ I.scalarABI eps.value=epsilonWord ∧ Index.Correspondence x w y ∧
      (∀ row col, ∀ hr : row<(geometry x.info).rows, ∀ hc : col<width x.info,
        (Index.pointer x.actual.dataPtr (width x.info) row col).toNat=x.view.address (row*width x.info+col) ∧
        (Index.pointer y.actual.dataPtr (width x.info) row col).toNat=y.view.address (row*width x.info+col) ∧
        (w.actual.dataPtr+BitVec.ofNat 64 col*BitVec.ofNat 64 4).toNat=w.view.address col ∧
        ∀ byte,byte<4 → x.view.address (row*width x.info+col)+byte<2^64 ∧
          y.view.address (row*width x.info+col)+byte<2^64 ∧ w.view.address col+byte<2^64) ∧
      ∃ final,Runtime.run graphAlgebra "X" "Y" "W" (geometry x.info).rows (width x.info)
        (block (width x.info)) (.input epsilonWord) schedule (Bytes.initial x w y memory)=some final ∧
        Bytes.project I P R (width x.info) y memory final=
          Bytes.execute I P R (width x.info) (block (width x.info)) epsilonWord y schedule (Bytes.initial x w y memory) memory ∧
        (∀ k,k<y.size → y.read (Bytes.load (Bytes.project I P R (width x.info) y memory final)) k=
          Bytes.nativeResult I P R x w memory (width x.info) (block (width x.info)) epsilonWord k) ∧
        Bytes.project I P R (width x.info) y memory final=
          Bytes.reference y memory (Bytes.nativeResult I P R x w memory (width x.info) (block (width x.info)) epsilonWord) ∧
        (∀ address,¬footprint y address → Bytes.project I P R (width x.info) y memory final address=memory address) ∧
        ((∀ k,k<x.size → x.read (Bytes.load (Bytes.project I P R (width x.info) y memory final)) k=x.read (Bytes.load memory) k) ∧
         (∀ k,k<w.size → w.read (Bytes.load (Bytes.project I P R (width x.info) y memory final)) k=w.read (Bytes.load memory) k)) ∧
        (∀ (region : RegionName) offset,region≠"Y" ∨ x.size≤offset →
          final.mem region offset=(Bytes.initial x w y memory).mem region offset) := by
  refine ⟨(checkTensorInput_iff _ _ _).mpr c,abi,Index.correspondence x w y eps c out,Index.source_addresses x w y eps c out,?_⟩
  obtain ⟨final,ran,values,frame⟩ := execution graphAlgebra x w y eps c (.input epsilonWord)
    (fun address => .input (Bytes.load memory address)) schedule all
  have flat : ∀ k,k<y.size → interpret I P R (k%width x.info) (Runtime.read final "Y" k)=
      Bytes.nativeResult I P R x w memory (width x.info) (block (width x.info)) epsilonWord k := by
    intro k hk
    have coords := flat_coordinates x w eps c k (by rw [←out.size]; exact hk)
    have sourceValue := values (k/width x.info) (k%width x.info) coords.1 coords.2.1
    rw [coords.2.2] at sourceValue
    rw [sourceValue]
    have result := Bytes.logical_value I P R x w y memory eps c epsilonWord _ _ coords.1 coords.2.1
    simpa only [coords.2.2] using result
  have physical : Bytes.project I P R (width x.info) y memory final=
      Bytes.reference y memory (Bytes.nativeResult I P R x w memory (width x.info) (block (width x.info)) epsilonWord) := by
    funext address
    simp only [Bytes.project,Bytes.reference]
    split
    · rename_i active
      rw [flat _ (by unfold footprint at active; omega)]
    · rfl
  refine ⟨final,ran,Bytes.run_bytes I P R x w y eps c out epsilonWord memory schedule all final ran,
    ?_,physical,?_,?_,frame⟩
  · intro k hk
    rw [physical,y.read_view _ k hk]
    exact Bytes.reference_load _ _ _ k hk
  · intro address outside
    rw [physical]
    exact Bytes.reference_frame _ _ _ address outside
  · rw [physical]
    exact ⟨Bytes.disjoint_read y x memory _ out.inputFresh,Bytes.disjoint_read y w memory _ out.weightFresh⟩

#axiomsClean kernel_correct
#axiomsClean kernel_native_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.TensorBridge
