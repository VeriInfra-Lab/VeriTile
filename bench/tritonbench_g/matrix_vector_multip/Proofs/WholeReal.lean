import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeHost
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeMath

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeReal
open VeriTile.Triton FP.Structural WholeSource WholeGrid WholeHost
noncomputable section
set_option maxHeartbeats 6000000

/-- Independent standard dot-product specification over the original logical
matrix and vector coordinates, before allocation or any autotune trial. -/
def mathematical (x : Inputs) (entry : Nat → ℝ) (i : Fin x.rows) : ℝ :=
  WholeMath.matrixVector x.rows x.columns
    (fun r k => entry (x.matrixAddress r.val k.val))
    (fun k => entry (x.vectorAddress k.val)) i

def family : Config → Nat → Algebra ℝ := fun _ _ => WholeMath.algebra

theorem row_mathematical (x : Inputs) (entry : Nat → ℝ) (preexisting : Nat → Prop)
    (compatible : x.columns=x.vectorLength) (a : Allocated ℝ x entry preexisting)
    (c : Config) (i : Fin x.rows) :
    rowValue (family c) (x.parameters c) c.rows_pos a.state.mem i.val=mathematical x entry i := by
  have rowEq : row (x.parameters c) (i.val/c.blockRows) (i.val%c.blockRows)=i.val := by
    simpa only [row,Inputs.parameters,Nat.mul_comm] using (Nat.div_add_mod i.val c.blockRows)
  unfold rowValue family
  rw [WholeMath.tile_mathematical (x.parameters c) c.columns_pos _ _ _ (by
    change row (x.parameters c) (i.val/c.blockRows) (i.val%c.blockRows)<x.rows
    rw [rowEq]; exact i.isLt)]
  unfold mathematical WholeMath.matrixVector
  apply Finset.sum_congr rfl
  intro k _
  unfold WholeMath.product
  change a.memory (x.matrixBase+(row (x.parameters c) (i.val/c.blockRows)
      (i.val%c.blockRows)*x.strideRow+k.val*x.strideColumn)) *
    a.memory (x.vectorBase+k.val*x.strideVector)=_
  rw [rowEq]
  have hm := a.protected_frame (x.matrixAddress i.val k.val)
    (a.matrixExisting i.val k.val i.isLt k.isLt)
  have hv := a.protected_frame (x.vectorAddress k.val)
    (a.vectorExisting k.val (by simpa only [←compatible] using k.isLt))
  simpa only [Inputs.matrixAddress,Inputs.vectorAddress,Nat.add_assoc] using congrArg₂ (·*·) hm hv

/-- Complete original host return, including all autotune trials, original
strided input coordinates and the physical preservation of preexisting words. -/
theorem execute_returns (x : Inputs) (entry : Nat → ℝ) (preexisting : Nat → Prop)
    (rt : Runtime ℝ x entry preexisting) (compatible : x.columns=x.vectorLength)
    (a : Allocated ℝ x entry preexisting) (allocated : rt.allocate=some a)
    (ready : rt.launchReady a=Bool.true) :
    ∃ out,WholeHost.execute family rt=.returned out ∧ out.length=x.rows ∧ out.dtype=.fp32 ∧
      out.device=x.device ∧ out.stride=1 ∧ out.base=a.base ∧
      (∀ i : Fin x.rows,out.memory (out.base+i.val)=mathematical x entry i) ∧
      (∀ addr,(∀ i,i<x.rows → addr≠out.base+i) → out.memory addr=a.memory addr) ∧
      (∀ addr,preexisting addr → out.memory addr=entry addr) := by
  obtain ⟨out,ran,shape,dtype,device,stride,base,values,exterior,frame⟩ :=
    WholeHost.execute_returns family rt compatible a allocated ready
  refine ⟨out,ran,shape,dtype,device,stride,base,?_,exterior,frame⟩
  intro i
  rw [values i.val i.isLt]
  exact row_mathematical x entry preexisting compatible a _ i

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeReal
