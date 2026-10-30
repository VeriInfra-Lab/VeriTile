import bench.repairs.tritonbench_g.fast_layernorm.Proofs.KernelReference
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Real

namespace VeriTile.Bench.Repairs.FastLayernorm.KernelMath
open VeriTile.Triton FP.Structural
noncomputable section
open scoped BigOperators
set_option maxHeartbeats 2000000

/-- Total real-number interpretation of the typed source. FP storage casts
become identities; each reduction is the mathematical sum of its axis. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .mul => x*y | .add => x+y | .sub => x-y | .div => x/y | _ => 0
  unary := fun _ op x => match op with | .rsqrt => 1/Real.sqrt x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {s} ax keep xs => match keep with
    | Bool.false => fun i => ∑ k,xs (TileShape.insertAxisIndex s ax i k)
    | Bool.true => fun i => ∑ k,xs (TileShape.replaceAxisIndex s ax i k)

theorem reduce_real (x : Fin B → ℝ) : KernelReference.reduce algebra x=∑ j,x j := by
  simp [KernelReference.reduce,algebra,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex]
  rfl

theorem sum_padding (N B : Nat) (covered : N≤B) (x : Nat → ℝ) :
    (∑ j : Fin B,if j.val<N then x j.val else 0)=∑ j∈Finset.range N,x j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<N then x j else 0)]
  exact VeriTile.Bench.TritonBenchG.LayerNormFwd.Formula.sum_padded x covered

theorem mean_real (N B : Nat) (covered : N≤B) (x : Nat → ℝ) :
    KernelReference.mean algebra N (fun j : Fin B => x j.val)=RealProof.meanSpec N x := by
  unfold KernelReference.mean
  rw [reduce_real]
  change (∑ j : Fin B,if j.val<N then x j.val else 0)/(N:ℝ)=_
  rw [sum_padding N B covered]
  rfl

theorem center_real (N B : Nat) (covered : N≤B) (x : Nat → ℝ) (j : Fin B) :
    KernelReference.centered algebra N (fun j : Fin B => x j.val) j=
      if j.val<N then x j.val-RealProof.meanSpec N x else 0 := by
  rw [KernelReference.centered,mean_real N B covered]
  by_cases hj : j.val<N <;> simp [hj,KernelReference.input,KernelReference.cast,KernelReference.sub,KernelReference.literal,algebra]

theorem inverse_real (N B : Nat) (covered : N≤B) (eps : ℝ) (x : Nat → ℝ) :
    KernelReference.inverse algebra N eps (fun j : Fin B => x j.val)=RealProof.invSpec N x eps := by
  unfold KernelReference.inverse
  rw [reduce_real]
  simp_rw [center_real N B covered]
  change 1/Real.sqrt (((∑ j : Fin B,(if j.val<N then x j.val-RealProof.meanSpec N x else 0)*
    (if j.val<N then x j.val-RealProof.meanSpec N x else 0))/(N:ℝ))+eps)=_
  have squares : (fun j : Fin B => (if j.val<N then x j.val-RealProof.meanSpec N x else 0)*
      (if j.val<N then x j.val-RealProof.meanSpec N x else 0))=
      fun j : Fin B => if j.val<N then (x j.val-RealProof.meanSpec N x)^2 else 0 := by
    funext j
    split <;> simp [pow_two]
  rw [squares,sum_padding N B covered (fun j => (x j-RealProof.meanSpec N x)^2)]
  rfl

theorem output_real (N B : Nat) (covered : N≤B) (eps : ℝ) (bias : Bool)
    (x w b : Nat → ℝ) (j : Fin B) (active : j.val<N) :
    KernelReference.output algebra N eps bias (fun j : Fin B => x j.val)
      (fun j => w j.val) (fun j => b j.val) j=
      Cell.mk .real (RealProof.outputSpec N x w (fun k => if bias then b k else 0) eps j.val) := by
  unfold KernelReference.output
  rw [center_real N B covered,inverse_real N B covered]
  cases bias <;> simp [active,KernelReference.cast,KernelReference.add,KernelReference.mul,
    KernelReference.input,KernelReference.literal,algebra,RealProof.outputSpec]

theorem gradient_real (N B : Nat) (covered : N≤B) (x dy w : Nat → ℝ) (avg inv : ℝ)
    (j : Fin B) (active : j.val<N) :
    KernelReference.gradient algebra N (fun j : Fin B => x j.val) (fun j => dy j.val)
      (fun j => w j.val) avg inv j=Cell.mk .real (RealProof.gradientSpec N x dy w avg inv j.val) := by
  have weighted : KernelReference.weighted algebra N (fun j : Fin B => dy j.val) (fun j => w j.val)=
      fun j : Fin B => if j.val<N then dy j.val*w j.val else 0 := by
    funext j
    by_cases hj : j.val<N <;> simp [KernelReference.weighted,KernelReference.input,KernelReference.cast,
      KernelReference.mul,KernelReference.literal,algebra,hj]
  have product : KernelReference.product algebra N (fun j : Fin B => x j.val)
      (fun j => dy j.val) (fun j => w j.val) avg inv=
      fun j : Fin B => if j.val<N then dy j.val*w j.val*((x j.val-avg)*inv) else 0 := by
    funext j
    rw [KernelReference.product,weighted]
    by_cases hj : j.val<N <;> simp [hj,KernelReference.normalized,KernelReference.input,
      KernelReference.cast,KernelReference.mul,KernelReference.sub,KernelReference.literal,algebra]
  unfold KernelReference.gradient
  rw [weighted,product]
  simp only [reduce_real]
  rw [sum_padding N B covered (fun j => dy j*w j),
    sum_padding N B covered (fun j => dy j*w j*((x j-avg)*inv))]
  simp [active,KernelReference.normalized,KernelReference.input,KernelReference.cast,KernelReference.mul,
    KernelReference.sub,KernelReference.div,KernelReference.literal,algebra,RealProof.gradientSpec]

theorem affine_real (active : Bool) (x dy avg inv : Nat → ℝ) (rows : Nat) :
    KernelReference.affineCarry algebra active x dy avg inv rows=
      (if active then ∑ row∈Finset.range rows,dy row*((x row-avg row)*inv row) else 0,
       if active then ∑ row∈Finset.range rows,dy row else 0) := by
  induction rows with
  | zero => simp [KernelReference.affineCarry,KernelReference.literal,algebra]
  | succ rows ih =>
    rw [KernelReference.affineCarry,ih]
    cases active <;> simp [KernelReference.affineTerm,KernelReference.add,KernelReference.sub,
      KernelReference.mul,KernelReference.cast,KernelReference.literal,algebra,Finset.sum_range_succ]

#axiomsClean output_real
#axiomsClean gradient_real
#axiomsClean affine_real
end
end VeriTile.Bench.Repairs.FastLayernorm.KernelMath
