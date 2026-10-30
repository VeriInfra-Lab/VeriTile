import VeriTile.Triton.Float.ScalarBackend
import bench.repairs.tritonbench_g.rope_embedding.Contract
import Mathlib.Tactic.Ring
import VeriTile.Meta.Specification
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural
noncomputable section

/-- FP32 load/cast sites are explicit local operations. There is no whole
rotation, execution or memory-result field. -/
structure Operations (α : Type) where
  zero : α
  load : α→α
  cast : α→α
  neg : α→α
  add : α→α→α
  sub : α→α→α
  mul : α→α→α
def Operations.backend (op : Operations α) : Algebra α where
  literal := fun _ _ _ => op.zero
  negInf := op.zero
  binary := fun _ _ code => match code with
    | .add => op.add | .sub => op.sub | .mul => op.mul | _ => fun a _ => a
  unary := fun _ code => match code with | .neg _ => op.neg | _ => id
  cast := fun _ _ _ => op.cast
  fromNat := fun _ _ => op.zero
  fromInt := fun _ _ => op.zero
  fp32Bits := fun _ => op.zero
  fp32Load := op.load
  reduceMax := fun _ {_} _ _ _ _ => op.zero
  reduceSum := fun _ {_} _ _ _ _ => op.zero
def scalarPair (op : Operations α) (backward : Bool) (a b c s : α) (second : Bool) : α :=
  let sine := if backward then op.neg (op.load s) else op.load s
  let cosine := op.load c
  let first := op.cast a
  let other := op.cast b
  if second then op.add (op.mul other cosine) (op.mul first sine)
    else op.sub (op.mul first cosine) (op.mul other sine)
def scalarResult (op : Operations α) (r : Request) (backward : Bool)
    (q cosine sine : Nat→α) (row head c : Nat) : α :=
  if c<2*r.half then
    let lane := if c<r.half then c else c-r.half
    scalarPair op backward (q (r.offset row head lane))
      (q (r.offset row head (lane+r.half)))
      (cosine (row%r.sequence*r.half+lane)) (sine (row%r.sequence*r.half+lane))
      (decide (r.half≤c))
  else q (r.offset row head c)
def ideal : Operations ℝ := ⟨0,id,id,(-·),(·+·),(·-·),(·*·)⟩
theorem ideal_pair (backward second : Bool) (a b c s : ℝ) :
    scalarPair ideal backward a b c s second =
      if second then (Spec.pair backward a b c s).2 else (Spec.pair backward a b c s).1 := by
  cases backward <;> cases second <;> simp [scalarPair,ideal,Spec.pair] <;> ring
theorem ideal_result (r : Request) (backward : Bool) (q c s : Nat→ℝ) (row head col : Nat) :
    scalarResult ideal r backward q c s row head col=Spec.result r backward q c s row head col := by
  unfold scalarResult Spec.result
  split
  · rw [ideal_pair]
    split_ifs <;> simp_all <;> omega
  · rfl

#specNonCircular Spec.result avoiding [scalarResult,FP.Structural.exec]
#axiomsClean ideal_result
#axiomsClean Spec.adjoint
end
end VeriTile.Bench.Repairs.RopeEmbedding
