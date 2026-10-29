import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteNativeExamples

namespace QuantTransposeFinite
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000

/-- Complete admitted finite source execution, with the actual metadata
widths, signed coordinates/masks, byte addresses and concrete stores. No Safe
or arithmetic-refinement premise excludes wrapped behavior. All primitive
numerical operations remain the exact explicitly bound Numerics ports.

Interpreter observations do not claim native exceptions or post-failure
rollback. Successful memory has a byte-store replay and independent pointwise
result. The logical transpose formula is a separately conditional refinement. -/
specification quant_transpose_finite_acceptance
    (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations) (m : Memory)
    (x : String) (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p)
    (nonempty : 0<a.numel) (rank : a.shape.length=2) (stride : a.sam=1 ∨ a.san=1)
    (cuda : a.isCuda=Bool.true) (trials : List Config) (last : Config) :
    (∃ maximum, produced b d logical=some maximum ∧
      externalMemory d a alloc p (allocated d a alloc m maximum (b.reciprocal d maximum))=
        hostPhysical d p maximum (b.reciprocal d maximum) ∧
      ((¬gridSupported a ∧ executeNativeAPI b (fun _ => none) (dtype d) a alloc m x p trials last=
          .rejected ⟨.gridParse,.gridLimit (gridM a.M*gridN a.N)⟩) ∨
       (gridSupported a ∧ ∃ issue,
          nativeTrialsPlan b d a alloc p (hostPhysical d p maximum (b.reciprocal d maximum))
            (trials++[last])=.error issue ∧
          executeNativeAPI b (fun _ => none) (dtype d) a alloc m x p trials last=
            .attempted d (.error (.observation issue))) ∨
       (gridSupported a ∧ ∃ ret,
          executeNativeAPI b (fun _ => none) (dtype d) a alloc m x p trials last=.attempted d (.ok ret) ∧
          NativeResult b d a alloc p maximum trials last ret))) ∧
    profiles.map (fun dt => nativeAdmission (fun _ => none) .cuda dt 3 2 Bool.true 3 1)=observedCuda ∧
    (∀ env dt input why, nativeAdmission env (if input.isCuda then .cuda else .cpu) dt input.numel input.shape.length
      (decide (input.sam=1 ∨ input.san=1)) input.M input.N=.error why →
      executeNativeAPI b env dt input alloc m x p trials last=.rejected why) ∧
    (∀ maximum ret, NativeResult b d a alloc p maximum trials last ret →
      (∀ k, k<bytes d → ret.memory (p.maximum.base+k)=some (byteOf maximum k)) ∧
      (∀ k, k<bytes (compute d) → ret.memory (p.inverse.base+k)=some (byteOf (b.reciprocal d maximum) k))) ∧
    (ArithmeticRefines a d p → ∀ maximum ret,
      NativeResult b d a alloc p maximum trials last ret →
      ∀ i, i<a.numel → ret.memory (p.output.base+i)=result b d (xs i) (b.reciprocal d maximum)) := by
  refine ⟨?_,native_small_dtype_inventory,?_,?_,?_⟩
  · obtain ⟨maximum,hm,outcome⟩ := native_host_all_outcomes b d a alloc m x logical xs p
      D P nonempty rank stride cuda trials last
    exact ⟨maximum,hm,host_physical_preparation d a alloc m x p D.fresh P maximum
      (b.reciprocal d maximum),outcome⟩
  · intro env dt input why h
    exact native_rejected_source_outcome b env dt input alloc m x p trials last why h
  · intro maximum ret R
    exact native_returned_scalar_bytes b d a alloc m x p P maximum trials last ret R
  · intro refines maximum ret R
    exact native_logical_transpose_refinement b d a alloc m x logical xs p D P refines maximum trials last ret R

#axiomsClean quant_transpose_finite_acceptance
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantTransposeFinite
