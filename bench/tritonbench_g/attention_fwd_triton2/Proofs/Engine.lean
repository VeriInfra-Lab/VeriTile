import bench.tritonbench_g.attention_fwd_triton2.Sources.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
open VeriTile.Triton FP.Structural
noncomputable section

structure Strides where
  (batch head row dim : Nat)
  deriving DecidableEq
structure Parameters where
  (Q K V QS KS Out : RegionName)
  (q k v o : Strides)
  (batches heads length : Nat)

def kernel (p : Parameters) : ComputeKernel :=
  TypedSource.repaired p.Q p.K p.V p.QS p.KS p.Out
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.batches p.heads p.length 96 128 64 128 96 1

/-- Private total typed adapter for execution lemmas. Public endpoints must
instantiate both products using explicit fixed-instruction chains. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  {M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p options {_} {_} {_} {_} a b => some (D.apply p .real .real .real options a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b)}

def parts (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Source
