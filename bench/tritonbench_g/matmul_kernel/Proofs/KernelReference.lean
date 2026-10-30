import bench.tritonbench_g.matmul_kernel.Proofs.KernelDecomposition
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
abbrev BM (p : Config) := p.BM
abbrev BN (p : Config) := p.BN
abbrev BK (p : Config) := p.BK
abbrev AIndex (p : Config) := TileIndex [BM p,BK p]
abbrev BIndex (p : Config) := TileIndex [BK p,BN p]
abbrev CIndex (p : Config) := TileIndex [BM p,BN p]

structure DotAcc (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α) → (TileIndex (batch ++ [m,n]) → α)
def backend {α : Type} (rest : Algebra α) (D : DotAcc α) : Algebra α :=
  {rest with dotAcc := fun p l r o opt {batch} {m} {k} {n} a b c =>
    match l,r,o with
    | .real,.real,.real => some (D.apply p .real .real .real opt a b c)
    | l,r,o => rest.dotAcc p l r o opt a b c}

def pidM {α : Type} (_ : Config) (s : State α) := s.pids 0
def pidN {α : Type} (_ : Config) (s : State α) := s.pids 1
def row {α : Type} (p : Config) (s : State α) (i : Fin (BM p)) := pidM p s*BM p+i.val
def col {α : Type} (p : Config) (s : State α) (j : Fin (BN p)) := pidN p s*BN p+j.val
def aRow {α : Type} (p : Config) (s : State α) (i : Fin (BM p)) := row p s i % p.M
def bCol {α : Type} (p : Config) (s : State α) (j : Fin (BN p)) := col p s j % p.N

def aOffset {α : Type} (p : Config) (s : State α) (k : Nat) (i : AIndex p) :=
  aRow p s i.1*p.A.sr+i.2.1.val*p.A.sc+k*(BK p*p.A.sc)
def bOffset {α : Type} (p : Config) (s : State α) (k : Nat) (i : BIndex p) :=
  i.1.val*p.B.sr+bCol p s i.2.1*p.B.sc+k*(BK p*p.B.sr)
def cOffset {α : Type} (p : Config) (s : State α) (i : CIndex p) :=
  p.C.sr*row p s i.1+p.C.sc*col p s i.2.1

def active {α : Type} (p : Config) (s : State α) (i : CIndex p) : Prop :=
  row p s i.1<p.M ∧ col p s i.2.1<p.N
instance {α : Type} (p : Config) (s : State α) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_ ∧ _))
def read {α : Type} [Inhabited α] (c : Cell α) : α := c.read .real
def stored {α : Type} (x : α) : Cell α := .mk .real x

/-- Every K-padded lane stays in the source-ordered live dot. Inactive
loads supply literal zero; no floating zero-product identity is used. -/
def aValue {α : Type} [Inhabited α] (M : Algebra α) (p : Config)
    (s : State α) (k : Nat) (i : AIndex p) : α :=
  M.fp32Load (if i.2.1.val+k*BK p<p.K then read (s.mem "A" (aOffset p s k i))
    else M.literal (some .fp32) .real 0)
def bValue {α : Type} [Inhabited α] (M : Algebra α) (p : Config)
    (s : State α) (k : Nat) (i : BIndex p) : α :=
  M.fp32Load (if i.1.val+k*BK p<p.K then read (s.mem "B" (bOffset p s k i))
    else M.literal (some .fp32) .real 0)
def update {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (s : State α) (k : Nat) (acc : CIndex p → α) : CIndex p → α :=
  D.apply (some .fp32) .real .real .real {} (batch := [])
    (aValue M p s k) (bValue M p s k) acc
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (s : State α) : CIndex p → α :=
  foldRange (update M D p s) 0 p.steps 1 (fun _ => M.literal (some .fp32) .real 0)
def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (s : State α) (i : CIndex p) : α :=
  M.cast (some .fp32) .fp16 .real (M.cast none .real .fp16 (accumulation M D p s i))
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
