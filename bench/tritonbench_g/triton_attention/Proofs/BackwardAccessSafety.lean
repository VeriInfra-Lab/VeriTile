import bench.tritonbench_g.triton_attention.Proofs.BackwardAccessGeometry
import bench.tritonbench_g.triton_attention.Proofs.BackwardAccessLoops

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000

/-- Both original backward loops, including all DQ updates and final DK/DV
stores, use the finite declared source buffers. -/
theorem backward_safe (rest : Algebra α) (P : Primitives α)
    (c : Shape) (scale : ℝ) (initial : State α) (legal : initial.pids 0<c.groups)
    (bounds : RegionName→Nat) (backed : Bounds c bounds) :
    TraceSafe (backend rest P) bounds (backwardKernel c scale) initial := by
  let p := backwardParameters c scale
  have bound (r : RegionName) (hr : r∈(["Q","K","V","NewDO","DQ","DK","DV"] : List RegionName))
      (b : Nat) (live : b<c.blocks) (i : BLane p) : backwardOffset p initial b i<bounds r := by
    apply lt_of_lt_of_le (backward_element_bound c scale initial legal b live i)
    apply backed.1 r
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr ⊢
    tauto
  have row (r : RegionName) (hr : r="M" ∨ r="Delta")
      (b : Nat) (live : b<c.blocks) (i : BRow p) : backwardRowOffset p initial b i<bounds r := by
    apply lt_of_lt_of_le (backward_row_bound c scale initial legal b live i)
    apply backed.2 r
    rcases hr with rfl|rfl <;> simp
  have safeLoop := backward_outer_loop_safe rest P p initial (backwardPrologueResult p initial) 0
    (by change 0<128;decide) (backwardPrologueInvariant p initial) bounds
    (fun b live i=>⟨bound "Q" (by simp) b live i,bound "NewDO" (by simp) b live i,
      bound "DQ" (by simp) b live i⟩)
    (fun b live i=>⟨row "M" (by simp) b live i,row "Delta" (by simp) b live i⟩)
    (fun b live i=>⟨bound "K" (by simp) b live i,bound "V" (by simp) b live i,
      bound "DK" (by simp) b live i,bound "DV" (by simp) b live i⟩)
  change Safety.traceSafeList _ _ (backwardSource p).surfaceBody initial
  rw [backwardDecomposition,Safety.traceSafeList_append]
  refine ⟨backward_prologue_safe (backend rest P) p initial bounds,?_⟩
  intro next ran
  rw [backwardPrologueExecution rest P p initial] at ran
  cases ran
  simpa only [AccessSafety.singleton_safe,Safety.stepSafe_unfold] using safeLoop

#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
