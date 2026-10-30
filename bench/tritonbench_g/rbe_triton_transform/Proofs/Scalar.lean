import bench.tritonbench_g.rbe_triton_transform.Proofs.Spec
namespace VeriTile.Bench.Repairs.RbeTritonTransform
open VeriTile.Triton FP.Structural
noncomputable section

/-- Only local scalar operations. FP32 storage-to-register and identity casts
do not change the carrier. Compiler contraction is an explicit separate port. -/
structure Operations (α : Type) where
  literal : ℝ → α
  natural : Nat → α
  integer : Int → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  pow : α → α → α
  sin : α → α
  cos : α → α

def Operations.backend (ops : Operations α) : Algebra α where
  literal := fun _ _ => ops.literal
  negInf := ops.literal 0
  binary := fun _ _ op => match op with
    | .add => ops.add | .sub => ops.sub | .mul => ops.mul
    | .div => ops.div | .pow => ops.pow | .max => fun a _ => a
  unary := fun _ op => match op with
    | .sin => ops.sin | .cos => ops.cos | _ => id
  cast := fun _ _ _ => id
  fromNat := fun _ => ops.natural
  fromInt := fun _ => ops.integer
  fp32Bits := fun _ => ops.literal 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => ops.literal 0
  reduceSum := fun _ {_} _ _ _ _ => ops.literal 0

def scalarAngle (ops : Operations α) (r : Request) (m n : Nat) : α :=
  ops.div (ops.integer (r.position+(m : Int)))
    (ops.pow (ops.literal 10000) (ops.div (ops.natural (n%128)) (ops.literal 128)))

def scalarPair (ops : Operations α) (r : Request) (x : Nat → α) (b m n : Nat) (imag : Bool) : α :=
  let u := x (r.offset b m n)
  let v := if n+1<r.columns then x (r.offset b m (n+1)) else ops.literal 0
  let a := scalarAngle ops r m n
  if imag then ops.add (ops.mul u (ops.sin a)) (ops.mul v (ops.cos a))
  else ops.sub (ops.mul u (ops.cos a)) (ops.mul v (ops.sin a))

def scalarResult (ops : Operations α) (r : Request) (x : Nat → α) (b m c : Nat) : α :=
  scalarPair ops r x b m (c/2*2) (decide (c%2=1))

def ideal : Operations ℝ :=
  ⟨id,fun n => n,fun n => n,(·+·),(·-·),(·*·),(·/·),Real.rpow,Real.sin,Real.cos⟩

theorem ideal_angle (r : Request) (m n : Nat) : scalarAngle ideal r m n = Spec.angle r m n := by
  simp [scalarAngle,ideal,Spec.angle]

theorem ideal_result (r : Request) (x : Nat → ℝ) (b m c : Nat) :
    scalarResult ideal r x b m c = Spec.result r x b m c := by
  simp [scalarResult,scalarPair,Spec.result,Spec.pair,scalarAngle,Spec.angle,ideal]

#axiomsClean ideal_result
#specNonCircular scalarResult avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RbeTritonTransform
