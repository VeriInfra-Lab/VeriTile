import bench.tritonbench_g.layer_norm_triton.Proofs.LogicalIO

/-! Independent Real formulas for all three genuine FP32 source kernels.
IO.lean and GroupedIO.lean expose channels, windows, masks and source identity.
RealEndpoints retains the old operational results for proof consumers. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Correctness
open VeriTile.Triton FP.Structural SourceModel Interface
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO VeriTile.Triton.LogicalIO
noncomputable section

namespace Mathematics
def meanSpec (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j ∈ Finset.range N,x j)/(N : ℝ)
def varianceSpec (N : Nat) (x : Nat → ℝ) : ℝ := (∑ j ∈ Finset.range N,(x j-meanSpec N x)^2)/(N : ℝ)
def invSpec (N : Nat) (x : Nat → ℝ) (eps : ℝ) : ℝ := 1/Real.sqrt (varianceSpec N x+eps)
def outputSpec (N : Nat) (x w bias : Nat → ℝ) (eps : ℝ) (col : Nat) : ℝ :=
  ((x col-meanSpec N x)*invSpec N x eps)*w col+bias col

def gradientSpec (N : Nat) (x dy w : Nat → ℝ) (avg inv : ℝ) (col : Nat) : ℝ :=
  let q := fun j => dy j*w j
  let z := fun j => (x j-avg)*inv
  (q col-(∑ j ∈ Finset.range N,q j)/(N : ℝ)-
    z col*(∑ j ∈ Finset.range N,q j*z j)/(N : ℝ))*inv

end Mathematics

/-- Independent pure forward mathematics on the three named input tensors;
all three outputs, including both saved statistics, remain explicit. -/
def forwardFormula (N : Nat) (eps : ℝ) (xs : Logical.ForwardInputs N) : Logical.ForwardOutputs N :=
  { y := fun j => Mathematics.outputSpec N xs.xAt xs.weightAt xs.biasAt eps j.val
    mean := Mathematics.meanSpec N xs.xAt
    inverse := Mathematics.invSpec N xs.xAt eps }

/-- Independent DX derivative and both group partial sums. The public IO
observes DX only on visited rows, while preserving its original broad writable
window and all prior-DX input coordinates. -/
def groupedFormula (M N G B group : Nat) (xs : Logical.GroupedInputs M N B) : Logical.GroupedOutputs M N B :=
  { dx := fun row j => Mathematics.gradientSpec N (Logical.extend (xs.xRow row.val))
      (Logical.extend (xs.dyRow row.val)) (Logical.extend xs.weight)
      (xs.mean row) (xs.inverse row) j.val
    dw := fun j => ((groupRows M G group).map (fun row => xs.dyRow row j *
      ((xs.xRow row j-xs.meanAt row)*xs.inverseAt row))).sum
    db := fun j => ((groupRows M G group).map (fun row => xs.dyRow row j)).sum }

/-- Y = ((X-mean(X))*rsqrt(variance(X)+epsilon))*W+Bias;
saved mean and inverse standard deviation are both public outputs. -/
specification forward_correct (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride N B row : Nat) (eps : ℝ) (positive : 0<N) (covers : N≤B) :
    VeriTile.Spec.Real (Logical.forwardIO m stride N B row eps ⊨ forwardFormula N eps) :=
  LogicalProof.forward_implements m laws stride N B row eps positive covers

/-- Every visited DX gradient and both partial parameter-gradient sums. The
existing row-address condition remains explicit; no input range is narrowed. -/
specification grouped_correct (m : Model ℝ) (laws : SourceForwardReal.Laws m)
    (stride M N G B group : Nat) (groups : 0<G) (covers : N≤B)
    (rowAddresses : ∀ row,row<M → row*stride=row*N) :
    VeriTile.Spec.Real (Logical.groupedIO m stride M N G B group ⊨ groupedFormula M N G B group) :=
  LogicalProof.grouped_implements m laws stride M N G B group groups covers rowAddresses

/-- Final DW and DB are independent sums of every supplied partial row. -/
specification final_correct (m : Model ℝ) (laws : SourceFinalReal.Laws m)
    (M N BN pid : Nat) :
    VeriTile.Spec.Real (Logical.finalIO m M N BN pid ⊨ fun xs =>
      ({ dw := fun col => ∑ row : Fin M, xs.dw row col
         db := fun col => ∑ row : Fin M, xs.db row col } : Logical.FinalOutputs BN)) :=
  LogicalProof.final_implements m laws M N BN pid

#specNonCircular forwardFormula avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.forward]
#specNonCircular groupedFormula avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.groupedBackward]
#specNonCircular FinalRealIO.formula avoiding [FP.Structural.exec,FP.Structural.run,TypedSource.finalBackward]
#axiomsClean forward_correct
#axiomsClean grouped_correct
#axiomsClean final_correct
end
end VeriTile.Bench.Repairs.LayerNormTriton.Correctness
