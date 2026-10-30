import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardReal
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedDX
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedSums
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceFinalReal
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceStrideRefinement

/-! Complete raw source quotations for all three genuine kernel stages.
Full Tensor launches and their saved-statistic composition are in TensorBridge. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Correctness
open VeriTile.Triton FP.Structural SourceModel
noncomputable section

alias layer_norm_forward_source_real := SourceForwardReal.executes

/-- Grouped backward computes every visited input gradient and both partial
parameter gradients, with the complete exterior typed-memory frame. -/
theorem layer_norm_grouped_source_real (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride M N G B : Nat) (s : State ℝ) (groups : 0<G) (covers : N≤B)
    (rowAddresses : ∀ row,row<M → row*stride=row*N) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s=some t ∧
      t.mem=(SourceGroupedExecution.final m stride M N G B s).mem ∧
      (∀ row,row∈groupRows M G (s.pids 0) → ∀ j,j<N →
        t.mem "DX" (row*N+j)=.mk .real
          (RealProof.Spec.gradient N (SourceGroupedReal.row s "X" N row)
            (SourceGroupedReal.row s "DY" N row) (SourceGroupedReal.weight s)
            (SourceGroupedReal.savedMean s row) (SourceGroupedReal.savedInverse s row) j)) ∧
      (∀ i : TileIndex [B],i.1.val<N →
        t.mem "DW" (s.pids 0*N+i.1.val)=.mk .real
          (((groupRows M G (s.pids 0)).map (SourceGroupedSums.weightTerm s stride i.1.val)).sum) ∧
        t.mem "DB" (s.pids 0*N+i.1.val)=.mk .real
          (((groupRows M G (s.pids 0)).map (SourceGroupedSums.biasTerm s stride i.1.val)).sum)) ∧
      (∀ r o,SourceGroupedResult.Outside stride M N (s.pids 0) r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,executed,memory,partials,frame⟩ := SourceGroupedResult.executes m stride M N G B s
  refine ⟨t,executed,memory,?_,?_,frame⟩
  · intro row hr j hj
    rw [memory,SourceStrideRefinement.grouped_memory m stride M N G B s rowAddresses,
      SourceGroupedDX.final_dx m M N G B row j s groups covers hj hr,
      SourceGroupedReal.dx_real m laws s N N B row j covers hj]
  · intro i hi
    have sums := SourceGroupedSums.fold_group m laws stride M N G B (s.pids 0) s groups i hi
    exact ⟨(partials i hi).1.trans (congrArg (Cell.mk .real) sums.1),
      (partials i hi).2.trans (congrArg (Cell.mk .real) sums.2)⟩

/-- Final backward reduces all supplied partial rows to independent exact
sums, preserving every cell outside the selected output columns. -/
theorem layer_norm_final_source_real (m : Model ℝ) (laws : SourceFinalReal.Laws m)
    (s : State ℝ) (M N BN : Nat) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N 32 BN) s=some t ∧
      t.mem=(SourceFinalGraph.final m s M N 32 BN).mem ∧
      (∀ i : TileIndex [BN],SourceFinalGraph.columns s BN i<N →
        t.mem "FINAL_DW" (SourceFinalGraph.columns s BN i)=.mk .real
          (∑ row ∈ Finset.range M,(s.mem "DW" (row*N+SourceFinalGraph.columns s BN i)).read .real) ∧
        t.mem "FINAL_DB" (SourceFinalGraph.columns s BN i)=.mk .real
          (∑ row ∈ Finset.range M,(s.mem "DB" (row*N+SourceFinalGraph.columns s BN i)).read .real)) ∧
      (∀ r o,SourceFinalResult.Outside s N BN r o → t.mem r o=s.mem r o) := by
  obtain ⟨t,executed,memory,values,frame⟩ := SourceFinalResult.executes m s M N 32 BN
  refine ⟨t,executed,memory,?_,frame⟩
  intro i hi
  have sums := SourceFinalReal.reduced_values m laws s M N BN i hi
  exact ⟨(values i hi).1.trans (congrArg (Cell.mk .real) sums.1),
    (values i hi).2.trans (congrArg (Cell.mk .real) sums.2)⟩

#axiomsClean layer_norm_forward_source_real
#axiomsClean layer_norm_grouped_source_real
#axiomsClean layer_norm_final_source_real
end
end VeriTile.Bench.Repairs.LayerNormTriton.Correctness
