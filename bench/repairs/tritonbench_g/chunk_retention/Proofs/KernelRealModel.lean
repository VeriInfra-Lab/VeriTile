import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelFold
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.RealScalar

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal
open VeriTile Triton FP.Structural KernelModel
open scoped BigOperators
noncomputable section

/-- Exact Real interpretation of the local scalar operations present in the
selected source. It imposes no relation between native words and Reals. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b
    | .max => max a b | .pow => Real.rpow a b
  unary := fun _ op x => match op with
    | .exp2 => Real.exp (x*Real.log 2)
    | .log2 => Real.log x/Real.log 2
    | _ => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def dotValues (l r out : FloatDType) :
    (batch : TileShape) → {m k n : Nat} →
      Values ℝ l.toTileDType (batch++[m,k]) →
      Values ℝ r.toTileDType (batch++[k,n]) →
      Values ℝ out.toTileDType (batch++[m,n])
  | [], _, k, _, a, b, i => ofFloat out
      (∑ e : Fin k, toFloat l (a ⟨i.1,e,()⟩)*toFloat r (b ⟨e,i.2.1,()⟩))
  | _::ds, _, _, _, a, b, i => dotValues l r out ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2
def dot : DotPrimitive ℝ where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b

abbrev Memory := RegionName → Nat → Cell ℝ
def inputs (c : Shape) (m : Memory) (seed terminal : Bool) : Inputs ℝ where
  q := fun h t e => (m "q" (h*(c.T*c.K)+t*c.K+e)).read .real
  k := fun h t e => (m "k" (h*(c.T*c.K)+t*c.K+e)).read .real
  v := fun h t j => (m "v" (h*(c.T*c.V)+t*c.V+j)).read .real
  seed := fun h e j => if seed then (m "h0" (h*c.K*c.V+e*c.V+j)).read .real else 0
  incoming := fun h t j => (m "incoming" (h*(c.T*c.V)+t*c.V+j)).read .real
  finalIncoming := fun h e j => if terminal then (m "terminal" (h*c.K*c.V+e*c.V+j)).read .real else 0

theorem rate_eq (c : Shape) (head : Nat) :
    KernelModel.rate algebra c head=logRate realOps c head := by
  simp only [KernelModel.rate,algebra,logRate,realOps,mul_one]
  norm_cast
  simp only [Int.cast_sub,Int.cast_natCast]

theorem decay_eq (c : Shape) (head : Nat) (n : Int) :
    KernelModel.decay algebra c head n=Whole001.decay realOps c head n := by
  unfold KernelModel.decay
  rw [rate_eq]
  rfl

#specNonCircular inputs avoiding [run,FP.Structural.exec]
#axiomsClean rate_eq
#axiomsClean decay_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelReal
