import bench.tritonbench_g.rms_matmul_rbe.Proofs.Reference
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- Only scalar conversions/operations and fixed eight-pair MMA are numerical
ports. Dot and reduction are expanded below from finite structural plans. -/
structure Arithmetic (α : Type) extends toScalar : Ops α where
  literal : FloatDType → ℝ → α
  cast : FloatDType → FloatDType → α → α
  e5Bits : FloatDType → Float8E5Bits → α
  fp32Bits : Float32Bits → α
  plan : RmsMatmulRbe.DotPlan
  reduction : RmsMatmulRbe.Reduction
  /-- One scalar realization identity connects HEAD_DIM's integer literal
  with the source literal callback; it imposes no runtime-conversion law. -/
  natLiteral : ∀ n : Nat,literal .real (n:ℝ)=toScalar.literalNat n

def kernelCast (a : Arithmetic α) (src dst : FloatDType) (x : α) : α :=
  if src=dst then x else a.cast src dst x

def binary (a : Arithmetic α) : Binary → α → α → α
  | .add => a.add
  | .sub => a.sub
  | .mul => a.mul
  | .div => a.divide
  | .pow => a.pow
  | .max => fun _ _ => a.zero

def chain (a : Arithmetic α) {k : Nat} (x y : Fin k → α) (seed : α) : α :=
  if hk : k=64 then
    a.plan.instructions.foldl (fun old lanes =>
      a.mma8 (fun j => x ⟨(lanes j).val,by have := (lanes j).isLt; omega⟩)
        (fun j => y ⟨(lanes j).val,by have := (lanes j).isLt; omega⟩) old) seed
  else a.zero

def tensor (a : Arithmetic α) {batch : TileShape} {m k n : Nat}
    (x : TileIndex (batch++[m,k]) → α) (y : TileIndex (batch++[k,n]) → α)
    (seed : TileIndex (batch++[m,n]) → α) : TileIndex (batch++[m,n]) → α :=
  match batch with
  | [] => fun i => chain a (fun j => x (i.1,j,PUnit.unit))
      (fun j => y (j,i.2.1,PUnit.unit)) (seed i)
  | _::rest => fun i => tensor a (batch:=rest)
      (fun j => x (i.1,j)) (fun j => y (i.1,j)) (fun j => seed (i.1,j)) i.2
termination_by batch.length

def reduceLine (a : Arithmetic α) {n : Nat} (v : Fin n → α) : α :=
  if hn : n=64 then a.reduction.tree.eval a.add
    (fun i => v ⟨i.val,by have := i.isLt; omega⟩) else a.zero

def reduceAxis (a : Arithmetic α) {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → Values α .real shape →
      Values α .real (TileShape.reduceShape shape ax keepDims)
  | Bool.false,v => fun out => reduceLine a
      (fun k => v (TileShape.insertAxisIndex shape ax out k))
  | Bool.true,v => fun out => reduceLine a
      (fun k => v (TileShape.replaceAxisIndex shape ax out k))

def Arithmetic.model (a : Arithmetic α) : Algebra α where
  literal := fun _ d x => a.literal d x
  negInf := a.zero
  binary := fun _ _ => binary a
  fma := fun _ => some a.fma
  unary := fun _ op => match op with
    | .neg _ => a.neg
    | .rsqrt => a.rsqrt
    | .cos => a.cos
    | .sin => a.sin
    | _ => fun _ => a.zero
  libdevicePowi := fun _ => some a.powi
  cast := fun _ => kernelCast a
  fromNat := fun _ n => a.ofInt n
  fromInt := fun _ => a.ofInt
  fp32Bits := a.fp32Bits
  float8E5Bits := some a.e5Bits
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => a.zero
  reduceSum := fun _ => reduceAxis a
  dot := fun _ {_} {_} {_} {_} x y => some (tensor a x y (fun _ => a.zero))
  dotTyped := fun _ left right out _ {_} {_} {_} {_} x y =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun _ => a.zero) i))
  dotAcc := fun _ left right out _ {_} {_} {_} {_} x y seed =>
    some (fun i => ofFloat out (tensor a (fun j => toFloat left (x j))
      (fun j => toFloat right (y j)) (fun j => toFloat out (seed j)) i))

theorem source_dot_seed (a : Arithmetic α)
    (x : Values α .real [16,64]) (w : Values α .real [64,64]) :
    a.model.dot (some .fp32) (batch:=[]) x w=
      some (tensor a x w (fun _ => a.zero)) := rfl

theorem chain_plan (a : Arithmetic α) (x y : Fin 64 → α) (seed : α) :
    chain a x y seed=a.plan.instructions.foldl
      (fun old lanes => a.mma8 (fun j => x (lanes j)) (fun j => y (lanes j)) old) seed := by
  simp [chain]

theorem reduction_plan (a : Arithmetic α) (v : Fin 64 → α) :
    reduceLine a v=a.reduction.tree.eval a.add v := by
  simp [reduceLine]

#axiomsClean source_dot_seed
#axiomsClean chain_plan
#axiomsClean reduction_plan
#specNonCircular Arithmetic avoiding [accumulation,moment,projection,value,referenceValue,wordTarget]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
