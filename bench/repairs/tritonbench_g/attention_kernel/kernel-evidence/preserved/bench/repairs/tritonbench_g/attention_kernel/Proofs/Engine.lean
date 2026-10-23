import bench.repairs.tritonbench_g.attention_kernel.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
open VeriTile.Triton FP.Structural
noncomputable section

structure Strides where
  (batch head row dim : Nat)
  deriving DecidableEq
structure Parameters where
  storage : TypedSource.Storage
  (Q K V Bias Out : RegionName)
  (q k v bias o : Strides)
  (batches heads length dim bm bn : Nat)
  scale : ℝ

def kernelAs (storage : TypedSource.Storage) (p : Parameters) : ComputeKernel :=
  TypedSource.repaired storage p.Q p.K p.V p.Bias p.Out p.scale
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.bias.batch p.bias.head p.bias.row
    p.batches p.heads p.length 0 64 128 p.dim p.bm p.bn

def kernel (p : Parameters) : ComputeKernel := kernelAs p.storage p

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

def partsAs (storage : TypedSource.Storage) (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernelAs storage p).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def parts (p : Parameters) := partsAs p.storage p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Source
