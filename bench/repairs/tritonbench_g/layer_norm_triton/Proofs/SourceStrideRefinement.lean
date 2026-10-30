import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceStride
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceForwardStorage
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceStrideRefinement
open VeriTile.Triton FP.Structural SourceModel
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section

/-- The actual flattened source view shares its backing and observed data
pointer with the admitted Tensor. Its strides are not required to be canonical. -/
structure MatrixView (input : KernelMemory.Input) (M N : Nat) where
  dense : Dense
  shape : dense.tensor.view.shape=[M,N]
  backing : dense.tensor.backing=input.dense.tensor.backing
  pointer : dense.address 0=input.dense.address 0

def MatrixView.stride (v : MatrixView input M N) := v.dense.tensor.view.strides.headD 0

theorem MatrixView.row_effect (v : MatrixView input M N) (hn : 0<N) (row : Nat) (hr : row<M) :
    row*v.stride=row*N := SourceStride.dense_row_offset v.dense M N row v.shape hn hr

theorem forward_memory [Inhabited α] (model : Model α) (s : State α) (stride N B : Nat) (eps : ℝ)
    (h : s.pids 0*stride=s.pids 0*N) :
    (SourceForwardGraph.final model stride N B eps s).mem=(SourceForwardGraph.final model N N B eps s).mem := by
  have hr : SourceForwardGraph.row s stride=SourceForwardGraph.row s N := by
    funext col
    simp only [SourceForwardGraph.row,h]
  simp only [SourceForwardGraph.final,hr,h]

theorem forward_actual [Inhabited α] (model : Model α) (B : Nat) (eps : ℝ)
    (api : Memory.API α Store Byte) (neg : α → α) (c : KernelContract.Forward M N)
    (memory : World Store) (view : MatrixView c.x M N) (hn : 0<N) (hb : N≤B) (row : Nat) (hr : row<M) :
    ∃ t,FP.Structural.exec (engine model)
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" view.stride N B eps)
      (SourceForwardStorage.initial api neg c memory row)=some t ∧
      t.mem=(SourceForwardGraph.final model N N B eps (SourceForwardStorage.initial api neg c memory row)).mem := by
  let s := SourceForwardStorage.initial api neg c memory row
  obtain ⟨t,ht⟩ := SourceForward.source_executes model view.stride N B eps hn hb s
  refine ⟨t,ht,(SourceForwardGraph.source_memory model view.stride N B eps hn hb s t ht).trans ?_⟩
  apply forward_memory
  simpa [s,SourceForwardStorage.initial] using view.row_effect hn row hr

theorem grouped_step [Inhabited α] (model : Model α) (s : State α) (stride N B row group : Nat)
    (a : SourceGroupedGraph.Carries α B) (h : row*stride=row*N) :
    SourceGroupedGraph.storeDX model s stride N B row=SourceGroupedGraph.storeDX model s N N B row ∧
    SourceGroupedGraph.nextCarries model s stride N B group row a=
      SourceGroupedGraph.nextCarries model s N N B group row a := by
  simp only [SourceGroupedGraph.storeDX,SourceGroupedGraph.dx,SourceGroupedGraph.first,SourceGroupedGraph.second,
    SourceGroupedGraph.xhat,SourceGroupedGraph.wdy,SourceGroupedGraph.loaded,SourceGroupedGraph.nextCarries,h]
  trivial

theorem grouped_fold [Inhabited α] (model : Model α) (stride M N G B group cur : Nat)
    (s : State α) (a : SourceGroupedGraph.Carries α B) (h : ∀ row,row<M → row*stride=row*N) :
    SourceGroupedExecution.fold model stride M N G B group cur s a=
      SourceGroupedExecution.fold model N M N G B group cur s a := by
  conv_lhs => rw [SourceGroupedExecution.fold]
  conv_rhs => rw [SourceGroupedExecution.fold]
  split
  · rfl
  · split
    · rw [(grouped_step model s stride N B cur group a (h cur ‹cur<M›)).1,
        (grouped_step model s stride N B cur group a (h cur ‹cur<M›)).2]
      exact grouped_fold model stride M N G B group (cur+G) _ _ h
    · rfl
termination_by M-cur
decreasing_by omega

theorem grouped_memory [Inhabited α] (model : Model α) (stride M N G B : Nat) (s : State α)
    (h : ∀ row,row<M → row*stride=row*N) :
    (SourceGroupedExecution.final model stride M N G B s).mem=(SourceGroupedExecution.final model N M N G B s).mem := by
  simp only [SourceGroupedExecution.final,grouped_fold model stride M N G B _ _ _ _ h]

#axiomsClean forward_actual
#axiomsClean grouped_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceStrideRefinement
