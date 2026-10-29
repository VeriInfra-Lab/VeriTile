import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNative
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceNativeStorage
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalResult

/-! Complete raw source/native relations. The source builds expression trees;
primitive instructions and structural compiler schedules determine native words.
Tensor/ABI applications and physical store/load composition are in TensorBridge. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
open VeriTile.Triton FP.Structural SourceModel SourceNative SourceNative.Native
noncomputable section

/-- The full primitive model also supports the original strict UInt32 source
interpretation; both source and candidate terminate in the same state. -/
alias layer_norm_forward_source_fp32 := SourceForwardResult.executes
alias layer_norm_grouped_source_fp32 := SourceGroupedResult.executes
alias layer_norm_final_source_fp32 := SourceFinalResult.executes

theorem layer_norm_forward_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (stride N B : Nat) (eps : ℝ) (s : State Word) (positive : 0<N) (covers : N≤B) :
    ∃ t,FP.Structural.exec (engine (model R))
        (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) (seed s)=some t ∧
      FP.Structural.exec (engine (model R)) (SourceCandidate.forward stride N B eps) (seed s)=some t ∧
      (∀ r o,read I S t r o=
        read I S (SourceForwardGraph.final (model R) stride N B eps (seed s)) r o) ∧
      (∀ r o,SourceForwardResult.Outside stride N (seed s) r o →
        t.mem r o=(seed s).mem r o ∧ read I S t r o=(s.mem r o).read .real) := by
  obtain ⟨t,executed,candidate,memory,_,_,_,frame⟩ :=
    SourceForwardResult.executes (model R) stride N B eps positive covers (seed s)
  refine ⟨t,executed,candidate,read_congr I S _ _ memory,?_⟩
  intro r o outside
  refine ⟨frame r o outside,?_⟩
  simp only [SourceNative.read,frame r o outside,SourceNative.seed,
    FastLayernorm.KernelNative.observe,Cell.read_same,
    FastLayernorm.KernelNative.normalize,FastLayernorm.KernelNative.eval]

theorem layer_norm_grouped_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (stride M N G B : Nat) (s : State Word) :
    ∃ t,FP.Structural.exec (engine (model R))
        (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) (seed s)=some t ∧
      (∀ r o,read I S t r o=
        read I S (SourceGroupedExecution.final (model R) stride M N G B (seed s)) r o) ∧
      (∀ r o,SourceGroupedResult.Outside stride M N (s.pids 0) r o →
        t.mem r o=(seed s).mem r o ∧ read I S t r o=(s.mem r o).read .real) := by
  obtain ⟨t,executed,memory,_,frame⟩ := SourceGroupedResult.executes (model R) stride M N G B (seed s)
  refine ⟨t,executed,read_congr I S _ _ memory,?_⟩
  intro r o outside
  refine ⟨frame r o outside,?_⟩
  simp only [SourceNative.read,frame r o outside,SourceNative.seed,
    FastLayernorm.KernelNative.observe,Cell.read_same,
    FastLayernorm.KernelNative.normalize,FastLayernorm.KernelNative.eval]

theorem layer_norm_final_native (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (M N BM BN : Nat) (s : State Word) :
    ∃ t,FP.Structural.exec (engine (model R))
        (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) (seed s)=some t ∧
      (∀ r o,read I S t r o=
        read I S (SourceFinalGraph.final (model R) (seed s) M N BM BN) r o) ∧
      (∀ r o,SourceFinalResult.Outside (seed s) N BN r o →
        t.mem r o=(seed s).mem r o ∧ read I S t r o=(s.mem r o).read .real) := by
  obtain ⟨t,executed,memory,_,frame⟩ := SourceFinalResult.executes (model R) (seed s) M N BM BN
  refine ⟨t,executed,read_congr I S _ _ memory,?_⟩
  intro r o outside
  refine ⟨frame r o outside,?_⟩
  simp only [SourceNative.read,frame r o outside,SourceNative.seed,
    FastLayernorm.KernelNative.observe,Cell.read_same,
    FastLayernorm.KernelNative.normalize,FastLayernorm.KernelNative.eval]

theorem layer_norm_native_candidate (p : SourceNativeStorage.Policy) (stride N B : Nat) (eps : ℝ)
    (s : State Word) (positive : 0<N) (covers : N≤B) :
    FP.Structural.exec (engine (model p.reductions)) (SourceCandidate.forward stride N B eps) (seed s)=
    FP.Structural.exec (engine (model p.reductions))
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) (seed s) :=
  SourceCandidate.forward_equiv (engine (model p.reductions)) stride N B eps positive covers (seed s)

#axiomsClean layer_norm_forward_source_fp32
#axiomsClean layer_norm_grouped_source_fp32
#axiomsClean layer_norm_final_source_fp32
#axiomsClean layer_norm_forward_native
#axiomsClean layer_norm_grouped_native
#axiomsClean layer_norm_final_native
#axiomsClean layer_norm_native_candidate
#specNonCircular SourceForwardGraph.final avoiding [FP.Structural.exec]
#specNonCircular SourceGroupedExecution.final avoiding [FP.Structural.exec]
#specNonCircular SourceFinalGraph.final avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.LayerNormTriton.Equivalence
