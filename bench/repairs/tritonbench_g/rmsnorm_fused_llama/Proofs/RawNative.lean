import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeReference
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeCandidate

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RawNative
open VeriTile Triton FP.Structural
noncomputable section

/-- Full raw typed source and candidate grids, with the independent ordered
native graph, runtime epsilon word, source FP16 round trip and full frame. -/
theorem rmsnorm_kernel_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (X Y W : RegionName) (rows columns : Nat) (epsilon : Native.Word)
    (initial : State Native.Term) (positive : 0<columns) (limit : columns≤16384)
    (xy : X≠Y) (wy : W≠Y) :
    ∃ final,
      RuntimeGrid.run (Native.algebra R) X Y W rows columns 16384 (.leaf "epsilon" 0 epsilon) initial rows=some final ∧
      RuntimeCandidate.grid (Native.algebra R) X Y W rows columns 16384 (.leaf "epsilon" 0 epsilon) initial rows=some final ∧
      (∀ row col,row<rows → col<columns →
        Native.observe I S final Y (row*columns+col)=
          RuntimeReference.native I R S (fun j=>RuntimeRow.read initial X (row*columns+j))
            (RuntimeRow.read initial W) columns 16384 (.leaf "epsilon" 0 epsilon) col) ∧
      RuntimeGrid.Frame Y (rows*columns) initial final := by
  obtain ⟨final,ran,memory⟩ := RuntimeGrid.complete (Native.algebra R) X Y W rows columns 16384
    (.leaf "epsilon" 0 epsilon) initial positive limit rows
  obtain ⟨values,frame⟩ := RuntimeGrid.graph_result (Native.algebra R) X Y W columns 16384
    (.leaf "epsilon" 0 epsilon) initial limit xy wy rows
  refine ⟨final,ran,?_,?_,?_⟩
  · rw [RuntimeCandidate.grid_eq]
    exact ran
  · intro row col hr hc
    unfold Native.observe
    rw [memory,values row hr col hc,Cell.read_same]
    apply congrArg (Native.eval I S)
    exact RuntimeReference.source_value R initial X W row columns 16384 (.leaf "epsilon" 0 epsilon)
      _ _ (fun _ _=>rfl) (fun _ _=>rfl) col hc
  · intro r i outside
    rw [memory]
    exact frame r i outside

alias source_candidate_equivalence := RuntimeCandidate.typed
#axiomsClean rmsnorm_kernel_native
#axiomsClean source_candidate_equivalence
#specNonCircular RuntimeReference.native avoiding [runtimeKernel,RuntimeGrid.run,RuntimeGrid.graph,RuntimeRow.value,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RawNative
