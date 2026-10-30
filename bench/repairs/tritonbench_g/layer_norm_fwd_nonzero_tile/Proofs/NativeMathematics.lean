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
    literal,cast,op_add,op_sub,op_mul,op_div,
    Formula.combined,Formula.stream,Formula.dropout,inputs,hj,hxf,hr,hsc,hd,
    keep]

theorem mean (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.meanValue algebra N
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      Formula.mean f (inputs a) M N p row := by
  rw [combined]
  simp only [TypedSource.meanValue,count]
  change (∑ j : Fin block, if j.val < N then Formula.combined f (inputs a) M p row j.val else 0) / (N : ℝ) = _
  rw [sum_prefix N block coverage]
  rfl

theorem centered (f : Flags) (a : NativeInterface.Inputs ℝ)
    (M N : Nat) (p : ℝ) (row block : Nat) (coverage : N ≤ block) :
    TypedSource.centeredValues algebra f.rms N
      (NativeInterface.combined algebra comparisons f a M N p row block) =
      fun j => if j.1.val < N then Formula.centered f (inputs a) M N p row j.1.val else 0 := by
  funext j
  simp only [TypedSource.centeredValues]
  rw [mean f a M N p row block coverage,combined]
  by_cases hj : j.1.val < N <;> cases h : f.rms <;>
    simp [hj,h,Formula.centered,literal,op_sub]

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
    by_cases hj : j.1.val<N <;> simp [hj,op_mul,pow_two]
  rw [square]
  simp only [TypedSource.meanValue,count]
  change (∑ j : Fin block, if j.val < N then (Formula.centered f (inputs a) M N p row j.val)^2 else 0) / (N : ℝ) = _
  rw [sum_prefix N block coverage (fun j => (Formula.centered f (inputs a) M N p row j)^2)]
  rfl

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
  simp only [TypedSource.normalizedValues]
  rw [mean f a M N p row block coverage,rstd f a M N eps p row block coverage]
  rw [combined]
  simp only [active,if_pos]
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
  funext a port
  change Fin 7 at port
  fin_cases port <;> funext j
  · have active : j.1.val % N < N := Nat.mod_lt _ (by have := j.1.isLt; nlinarith)
    simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,List.length_cons,List.length_nil]
    rw [normalized f _ M N eps p _ _ width _ active]
    rw [←input_coordinates]
    simp [TypedSource.affineValue,NativeInterface.loaded,active,cast,
      default_mul,default_add,Formula.y,inputs]
    rfl
  · have active : j.1.val % N < N := Nat.mod_lt _ (by have := j.1.isLt; nlinarith)
    simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,List.length_cons,List.length_nil]
    rw [normalized f _ M N eps p _ _ width _ active]
    rw [←input_coordinates]
    simp [TypedSource.affineValue,NativeInterface.loaded,active,cast,
      default_mul,default_add,Formula.y1,inputs]
    rfl
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,List.length_cons,List.length_nil]
    simpa only [input_coordinates] using
      (mean f (NativeInterface.logicalInputs a) M N p j.1.val (tile N) width)
  · simp only [NativeInterface.target,ActualLogical.outputArrays,Equiv.coe_fn_symm_mk,
      RealInterface.formula,List.length_cons,List.length_nil]
    simpa only [input_coordinates] using
      (rstd f (NativeInterface.logicalInputs a) M N eps p j.1.val (tile N) width)
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
