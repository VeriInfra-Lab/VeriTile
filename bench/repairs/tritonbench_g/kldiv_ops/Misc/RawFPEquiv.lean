import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Native
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.BackwardCoverage
import bench.repairs.tritonbench_g.kldiv_ops.Kernels

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.RawFPEquiv
open VeriTile.Triton FP.Structural
open KldivTriton.Native (Word Node atom)
noncomputable section

/-- Complete source execution over raw FP32 words, lowered to the independent
ordered native graph. Epsilon conversion and reduction plans remain explicit. -/
theorem forward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (red : Nat) (eps : BitVec 64) (s : State Node)
    (prediction target : Nat → Word)
    (predictionReads : ∀ i,(s.mem p.Y i).read .real=atom (prediction i))
    (targetReads : ∀ i,(s.mem p.T i).read .real=atom (target i))
    (block : p.block=Host.blockSize p.cols) (predictionSeparate : p.Y≠p.O)
    (targetSeparate : p.T≠p.O) :
    ∃ final,Program.grid (Contiguous.Native.algebra ports plans) p flag red
        (atom (ports.bindEpsilon eps)) s=some final ∧
      (∀ row,row<p.rows → ∀ col,col<p.cols → red=0 →
        ((final.mem p.O (row*p.cols+col)).read .real).value=
          Contiguous.Native.loss ports flag (ports.bindEpsilon eps)
            (prediction (row*p.cols+col)) (target (row*p.cols+col))) ∧
      (∀ row,row<p.rows → red≠0 → ((final.mem p.O row).read .real).value=
        Contiguous.Native.row ports plans p.cols flag (ports.bindEpsilon eps)
          (fun i => prediction (row*p.cols+i)) (fun i => target (row*p.cols+i))) ∧
      (∀ r i,(∀ row,row<p.rows → ¬Program.Writes p red row r i) → final.mem r i=s.mem r i) := by
  have positive : 0<p.block := block ▸ Host.block_positive p.cols
  obtain ⟨final,executed,memory⟩ := Program.grid_complete (Contiguous.Native.algebra ports plans)
    p flag red (atom (ports.bindEpsilon eps)) s positive
  refine ⟨final,executed,?_,?_,?_⟩
  · intro row hr col hc hz
    subst red
    rw [memory,Program.grid_value _ p flag _ _ s.mem row col (List.mem_range.mpr hr)
      positive hc predictionSeparate targetSeparate]
    simp only [Cell.read_same,predictionReads,targetReads,Contiguous.Native.scalar_lowering]
    rfl
  · intro row hr hn
    rw [memory,FPGraph.grid_binding _ p flag red _ _ s.mem row (List.mem_range.mpr hr)
      hn predictionSeparate targetSeparate]
    simp only [Cell.read_same,predictionReads,targetReads,block]
    exact Contiguous.Native.row_lowering ports plans p.cols flag (ports.bindEpsilon eps) _ _
  · intro r i outside
    rw [memory]
    exact Program.grid_frame _ p flag red _ _ s.mem r i
      (fun row member => outside row (List.mem_range.mp member))

theorem backward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (s : State Node) (target : Nat → Word)
    (targetReads : ∀ i,(s.mem p.T i).read .real=atom (target i))
    (block : 0<p.block) (targetSeparate : p.T≠p.O) :
    ∃ final,Backward.grid (Contiguous.Native.algebra ports plans) p flag s=some final ∧
      (∀ row,row<p.rows → ∀ col,col<p.cols →
        ((final.mem p.O (row*p.cols+col)).read .real).value=
          Contiguous.Native.derivative ports flag (target (row*p.cols+col))) ∧
      (∀ r i,(∀ row,row<p.rows → r ≠ p.O ∨ i < row*p.cols ∨ row*p.cols+p.cols ≤ i) →
        final.mem r i=s.mem r i) := by
  obtain ⟨final,executed,memory⟩ := Backward.grid_complete (Contiguous.Native.algebra ports plans) p flag s block
  refine ⟨final,executed,?_,?_⟩
  · intro row hr col hc
    rw [memory,Backward.grid_value _ p flag _ s.mem row col (List.mem_range.mpr hr) block hc targetSeparate]
    simp only [Cell.read_same,targetReads,Contiguous.Native.backward_lowering]
    rfl
  · intro r i outside
    rw [memory]
    exact Backward.grid_frame _ p flag _ s.mem r i
      (fun row member => outside row (List.mem_range.mp member))

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.RawFPEquiv
