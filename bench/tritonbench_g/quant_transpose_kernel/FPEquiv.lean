import bench.tritonbench_g.quant_transpose_kernel.Kernels
import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteNativeExamples
import bench.tritonbench_g.quant_transpose_kernel.Proofs.IntegerSource
import bench.tritonbench_g.quant_transpose_kernel.Proofs.TypedShape

/-! Existing floating execution contracts with selected FP32 input and int8
output. General finite routes remain available for compatibility; no new dtype
matrix is required. Numerical primitive assumptions remain explicit and this
file does not claim completed native numerical equivalence to Real. -/
local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

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

namespace QuantizeIntegerStage.Transpose
open VeriTile.Triton FP.Structural QuantizeIntegerStage.FP32
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

specification quant_transpose_fp32_llrint_int8_correctness (rest : Algebra Word) (round : Llrint64 Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat)
    (hOutInj : Function.Injective (outputOffset s sbn sbm BM BN)) :
    ∃ t, QuantizeIntegerStage.exec (backend rest cfg) (some round) (kernel A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      (∀ i, active s M N BM BN i →
        t.mem B (outputOffset s sbn sbm BM BN i) = .mk .int (word round cfg s A AbsmaxInv sam san BM BN i).toInt) ∧
      (∀ i, active s M N BM BN i →
        BitVec.ofInt 8 ((t.mem B (outputOffset s sbn sbm BM BN i)).read .int) =
          word round cfg s A AbsmaxInv sam san BM BN i) ∧
      (∀ r o, (r ≠ B ∨ ∀ i, active s M N BM BN i → o ≠ outputOffset s sbn sbm BM BN i) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hm⟩ := source_run rest round cfg s A AbsmaxInv B sam san sbn sbm M N BM BN
  have readback (i) (hi : active s M N BM BN i) :
      t.mem B (outputOffset s sbn sbm BM BN i) =
        .mk .int (word round cfg s A AbsmaxInv sam san BM BN i).toInt := by
    rw [hm]
    exact State.masked_scatter_readback B (outputOffset s sbn sbm BM BN) _
      (active s M N BM BN) s i hi hOutInj
  refine ⟨t, he, readback, ?_, ?_⟩
  · intro i hi
    rw [readback i hi]
    simp only [Cell.read_same, BitVec.ofInt_toInt]
  · intro r o hmiss
    rw [hm]
    unfold written
    apply State.masked_scatter_frame
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho i hi)

#axiomsClean quant_transpose_fp32_llrint_int8_correctness
end
end QuantizeIntegerStage.Transpose

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel.FPCorrect
open VeriTile.Triton FP.Structural TypedScaled
noncomputable section

/-- The accepted pre-rounding stage, with two separately rounded fp32
multiplications, exact fp32 literal127, masked transposed stores and full frame.
This does not claim the llrint/int8 output or the grouped-grid launch pipeline. -/
specification quant_transpose_scaled_store_fp32_correctness
    (rest : Algebra Word) (cfg : FP.Config) (s : State Word)
    (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat)
    (hOutInj : Function.Injective (outputOffset s sbn sbm BM BN)) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source A AbsmaxInv B sam san sbn sbm M N BM BN) s = some t ∧
      (∀ i, TypedScaled.active s M N BM BN i →
        t.mem B (outputOffset s sbn sbm BM BN i) = .mk .real (value cfg s A AbsmaxInv sam san BM BN i)) ∧
      (∀ r o, (r ≠ B ∨ ∀ i, TypedScaled.active s M N BM BN i → o ≠ outputOffset s sbn sbm BM BN i) →
        t.mem r o = s.mem r o) :=
  TypedScaled.source_correct rest cfg s A AbsmaxInv B sam san sbn sbm M N BM BN hOutInj

#axiomsClean quant_transpose_scaled_store_fp32_correctness
end
end VeriTile.Bench.TritonBenchG.QuantTransposeKernel.FPCorrect
