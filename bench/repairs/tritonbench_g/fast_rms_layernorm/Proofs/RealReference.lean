import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv

/-! Independent logical RMS normalization and both input and weight gradients.
No source execution, output extraction, packed address or source-gradient
helper appears in these definitions. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealReference
noncomputable section
open scoped BigOperators

def weight (gemma : Bool) (w : ℝ) : ℝ := if gemma then w+1 else w

def variance {N : Nat} (eps : ℝ) (x : Fin N → ℝ) : ℝ :=
  (∑ j,x j^2)/(N:ℝ)+eps

def inv {N : Nat} (eps : ℝ) (x : Fin N → ℝ) : ℝ := 1/Real.sqrt (variance eps x)

def output {N : Nat} (gemma : Bool) (eps : ℝ) (x w : Fin N → ℝ) (j : Fin N) : ℝ :=
  x j*inv eps x*weight gemma (w j)

def inputGradient {N : Nat} (gemma : Bool) (eps : ℝ) (x w g : Fin N → ℝ) (j : Fin N) : ℝ :=
  let r:=inv eps x
  r*(g j*weight gemma (w j))-(r^3/(N:ℝ))*x j*(∑ k,g k*weight gemma (w k)*x k)

def weightGradient {M N : Nat} (eps : ℝ) (x g : Fin M → Fin N → ℝ) (j : Fin N) : ℝ :=
  ∑ i,g i j*x i j*inv eps (x i)

def perturb {N : Nat} (x v : Fin N → ℝ) (t : ℝ) : Fin N → ℝ := fun j => x j+t*v j

theorem variance_directional {N : Nat} (eps : ℝ) (x v : Fin N → ℝ) :
    HasDerivAt (fun t => variance eps (perturb x v t))
      ((2/(N:ℝ))*(∑ j,x j*v j)) 0 := by
  have each (j : Fin N) : HasDerivAt (fun t : ℝ => (x j+t*v j)^2) (2*(x j)*v j) 0 := by
    convert (((hasDerivAt_id (0:ℝ)).mul_const (v j)).const_add (x j)).pow 2 using 1 <;> simp <;> ring
  have total := ((HasDerivAt.sum (u:=Finset.univ) (fun j _ => each j)).div_const (N:ℝ)).add_const eps
  convert total using 1
  · funext t
    simp [variance,perturb,Finset.sum_apply]
  · simp only [div_eq_mul_inv,Finset.mul_sum,Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro j _
    ring

@[simp] theorem perturb_zero {N : Nat} (x v : Fin N → ℝ) : perturb x v 0=x := by
  funext j
  simp [perturb]

theorem inv_directional {N : Nat} (hN : 0<N) (eps : ℝ) (x v : Fin N → ℝ)
    (domain : 0<variance eps x) :
    HasDerivAt (fun t => inv eps (perturb x v t))
      (-((inv eps x)^3/(N:ℝ))*(∑ j,x j*v j)) 0 := by
  have root := (variance_directional eps x v).sqrt (by simpa using ne_of_gt domain)
  have rootNe : Real.sqrt (variance eps x)≠0 := ne_of_gt (Real.sqrt_pos.mpr domain)
  have nNe : (N:ℝ)≠0 := by exact_mod_cast ne_of_gt hN
  have quotient := (hasDerivAt_const (0:ℝ) (1:ℝ)).div root (by simpa using rootNe)
  convert quotient using 1
  simp only [perturb_zero]
  unfold inv
  field_simp [rootNe,nNe]
  <;> ring

theorem coordinate_input_directional {N : Nat} (hN : 0<N) (gemma : Bool) (eps : ℝ)
    (x w v : Fin N → ℝ) (j : Fin N) (domain : 0<variance eps x) :
    HasDerivAt (fun t => output gemma eps (perturb x v t) w j)
      ((v j*inv eps x-x j*((inv eps x)^3/(N:ℝ))*(∑ k,x k*v k))*weight gemma (w j)) 0 := by
  have lin := ((hasDerivAt_id (0:ℝ)).mul_const (v j)).const_add (x j)
  have norm := (lin.mul (inv_directional hN eps x v domain)).mul_const (weight gemma (w j))
  convert norm using 1
  simp only [id_eq,one_mul,zero_mul,add_zero,perturb_zero]
  ring

theorem coordinate_weight_directional {N : Nat} (gemma : Bool) (eps : ℝ)
    (x w v : Fin N → ℝ) (j : Fin N) :
    HasDerivAt (fun t => output gemma eps x (perturb w v t) j)
      (x j*inv eps x*v j) 0 := by
  have lin := ((hasDerivAt_id (0:ℝ)).mul_const (v j)).const_add (w j)
  cases gemma
  · simpa only [output,weight,perturb,Bool.false_eq_true,if_false,id_eq,one_mul] using lin.const_mul (x j*inv eps x)
  · simpa only [output,weight,perturb,if_true,id_eq,one_mul] using (lin.add_const 1).const_mul (x j*inv eps x)

theorem row_loss_input_directional {N : Nat} (hN : 0<N) (gemma : Bool) (eps : ℝ)
    (x w g v : Fin N → ℝ) (domain : 0<variance eps x) :
    HasDerivAt (fun t => ∑ j,g j*output gemma eps (perturb x v t) w j)
      (∑ j,inputGradient gemma eps x w g j*v j) 0 := by
  have total := HasDerivAt.fun_sum (u:=Finset.univ)
    (fun j _ => (coordinate_input_directional hN gemma eps x w v j domain).const_mul (g j))
  convert total using 1
  let r:=inv eps x
  let c:=r^3/(N:ℝ)
  let d:=∑ j,g j*weight gemma (w j)*x j
  let e:=∑ j,g j*weight gemma (w j)*v j
  let f:=∑ j,x j*v j
  calc
    _ = ∑ j,(r*(g j*weight gemma (w j)*v j)-c*d*(x j*v j)) := by
      apply Finset.sum_congr rfl
      intro j _
      dsimp [inputGradient,r,c,d]
      ring
    _ = r*e-c*d*f := by rw [Finset.sum_sub_distrib,←Finset.mul_sum,←Finset.mul_sum]
    _ = ∑ j,(r*(g j*weight gemma (w j)*v j)-c*f*(g j*weight gemma (w j)*x j)) := by
      rw [Finset.sum_sub_distrib,←Finset.mul_sum,←Finset.mul_sum]
      dsimp only [e,d]
      ring
    _ = _ := by
      apply Finset.sum_congr rfl
      intro j _
      dsimp [r,c,f]
      ring

theorem full_input_directional {M N : Nat} (hN : 0<N) (gemma : Bool) (eps : ℝ)
    (x g v : Fin M → Fin N → ℝ) (w : Fin N → ℝ)
    (domain : ∀ i,0<variance eps (x i)) :
    HasDerivAt (fun t => ∑ i,∑ j,g i j*output gemma eps (perturb (x i) (v i) t) w j)
      (∑ i,∑ j,inputGradient gemma eps (x i) w (g i) j*v i j) 0 :=
  HasDerivAt.fun_sum (fun i _ => row_loss_input_directional hN gemma eps (x i) w (g i) (v i) (domain i))

theorem full_weight_directional {M N : Nat} (gemma : Bool) (eps : ℝ)
    (x g : Fin M → Fin N → ℝ) (w v : Fin N → ℝ) :
    HasDerivAt (fun t => ∑ i,∑ j,g i j*output gemma eps (x i) (perturb w v t) j)
      (∑ j,weightGradient eps x g j*v j) 0 := by
  have total := HasDerivAt.fun_sum (u:=Finset.univ) (fun i _ =>
    HasDerivAt.fun_sum (u:=Finset.univ) (fun j _ =>
      (coordinate_weight_directional gemma eps (x i) w v j).const_mul (g i j)))
  convert total using 1
  simp only [weightGradient,Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

#axiomsClean full_input_directional
#axiomsClean full_weight_directional

#axiomsClean inv_directional
#axiomsClean coordinate_input_directional
#axiomsClean coordinate_weight_directional

#axiomsClean variance_directional
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealReference
