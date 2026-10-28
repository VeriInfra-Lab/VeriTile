import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.NativeIO
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RealPrimitives

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
open scoped BigOperators Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def inputs (a : NativeInterface.Inputs ℝ) : Inputs :=
  ⟨a.x,a.x1,a.residual,a.weight,a.bias,a.weight1,a.bias1,a.rowscale,a.seeds⟩

theorem combined (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) :
    NativeInterface.combined algebra comparisons f a M N p row block =
      fun j => if j.1.val < N then Formula.combined f (inputs a) M p row j.1.val else 0 := by
  funext j
  by_cases hj : j.1.val < N
  all_goals cases hxf : f.x1 <;> cases hr : f.residual <;>
    cases hsc : f.rowscale <;> cases hd : f.dropout
  all_goals simp [NativeInterface.combined,NativeInterface.stream,NativeInterface.loaded,
    TypedSource.op,TypedSource.lit,TypedSource.cast,TypedSource.scalarLoad,
    Formula.combined,Formula.stream,Formula.dropout,inputs,hj,hxf,hr,hsc,hd,
    keep,algebra,engine,Algebra.withDefaultPrecision]

theorem mean (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.meanValue algebra N
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      Formula.mean f (inputs a) M N p row := by
  rw [combined]
  simp only [TypedSource.meanValue,count,TypedSource.op,algebra,engine,
    Algebra.withDefaultPrecision,realSum,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,Formula.mean]
  congr 1
  exact sum_prefix N block coverage (Formula.combined f (inputs a) M p row)

theorem centered (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.centeredValues algebra f.rms N
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      fun j => if j.1.val < N then Formula.centered f (inputs a) M N p row j.1.val else 0 := by
  funext j
  simp only [TypedSource.centeredValues,mean f a M N p row block coverage,combined,
    TypedSource.op,TypedSource.lit,algebra,engine,Algebra.withDefaultPrecision,Formula.centered]
  by_cases hj : j.1.val < N <;> simp [hj]

theorem variance (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.varianceValue algebra f.rms N
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      Formula.variance f (inputs a) M N p row := by
  rw [TypedSource.varianceValue,centered f a M N p row block coverage]
  have square : (fun j : TileIndex [block] => TypedSource.op algebra .mul
      (if j.1.val<N then Formula.centered f (inputs a) M N p row j.1.val else 0)
      (if j.1.val<N then Formula.centered f (inputs a) M N p row j.1.val else 0)) =
      (fun j => if j.1.val<N then (Formula.centered f (inputs a) M N p row j.1.val)^2 else 0) := by
    funext j
    by_cases hj : j.1.val<N <;> simp [hj,TypedSource.op,algebra,engine,Algebra.withDefaultPrecision,pow_two]
  rw [square]
  simp only [TypedSource.meanValue,count,TypedSource.op,algebra,engine,
    Algebra.withDefaultPrecision,realSum,TileShape.axisDim,TileShape.eraseAxis,
    TileShape.insertAxisIndex,Formula.variance]
  congr 1
  exact sum_prefix N block coverage (fun j => (Formula.centered f (inputs a) M N p row j)^2)

theorem rstd (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (eps p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.rstdValue algebra f.rms N eps
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      Formula.rstd f (inputs a) M N eps p row := by
  rw [TypedSource.rstdValue,variance f a M N p row block coverage]
  rfl

theorem normalized (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (eps p : ℝ) (row block : Nat) (coverage : N ≤ block)
    (j : TileIndex [block]) (active : j.1.val<N) :
    TypedSource.normalizedValues algebra f.rms N eps
      (NativeInterface.combined algebra comparisons f a M N p row block) j =
      Formula.normalized f (inputs a) M N eps p row j.1.val := by
  simp only [TypedSource.normalizedValues,mean f a M N p row block coverage,
    rstd f a M N eps p row block coverage,combined,active,if_pos]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem input_coordinates (r : Regions) (M N : Nat)
    (a : NativeInterface.Arrays r M N ℝ) :
    inputs (NativeInterface.logicalInputs a) =
      RealInterface.logicalInputs M N (ActualLogical.inputArrays r M N a) := rfl

theorem target (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (width : N ≤ tile N) :
    NativeInterface.target algebra comparisons f r stride M N eps p width =
      fun a => (ActualLogical.outputArrays r M N).symm
        (RealInterface.formula f M N eps p (ActualLogical.inputArrays r M N a)) := by
  funext a port j
  fin_cases port
  · have active : j.1.val % N < N := Nat.mod_lt _ (by have := j.1.isLt; nlinarith)
    simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula]
    rw [normalized f _ M N eps p _ _ width _ active]
    simp [TypedSource.affineValue,NativeInterface.loaded,active,TypedSource.cast,
      TypedSource.defaultOp,algebra,engine,Algebra.withDefaultPrecision,
      Formula.y,input_coordinates]
  · have active : j.1.val % N < N := Nat.mod_lt _ (by have := j.1.isLt; nlinarith)
    simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula]
    rw [normalized f _ M N eps p _ _ width _ active]
    simp [TypedSource.affineValue,NativeInterface.loaded,active,TypedSource.cast,
      TypedSource.defaultOp,algebra,engine,Algebra.withDefaultPrecision,
      Formula.y1,input_coordinates]
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula]
    rw [mean f _ M N p _ _ width,input_coordinates]
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula]
    rw [rstd f _ M N eps p _ _ width,input_coordinates]
  · have active : j.1.val % N < N := Nat.mod_lt _ (by have := j.1.isLt; nlinarith)
    simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,combined,active,if_pos,input_coordinates]
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,keep]
    rfl
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,keep]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.ActualMathematics
