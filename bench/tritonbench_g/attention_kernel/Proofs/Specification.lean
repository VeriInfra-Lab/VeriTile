import bench.tritonbench_g.attention_kernel.Proofs.Model
import bench.tritonbench_g.attention_kernel.Proofs.FamilyNativeStaging
import bench.tritonbench_g.attention_kernel.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.AttentionKernel.KernelOnly.Specification
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def query (M : Algebra α) (p : Parameters) (read : RegionName → Nat → α)
    (block batch head : Nat) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) p.storage.float .real
        (if block*p.bm+i.1.val<p.length then
          read p.Q (offset p.q batch head (block*p.bm+i.1.val) i.2.1.val)
         else M.literal (some .fp32) p.storage.float 0))
      (M.binary (some .fp32) .real .mul
        (M.literal (some .fp32) .real p.scale) (M.literal (some .fp32) .real 1.44269504)))

def key (M : Algebra α) (p : Parameters) (read : RegionName → Nat → α)
    (_block batch head row : Nat) (i : TileIndex [p.dim,p.bn]) : α :=
  if row+i.2.1.val<p.length then
    read p.K (offset p.k batch head (row+i.2.1.val) i.1.val)
  else M.literal (some .fp32) p.storage.float 0

def value (M : Algebra α) (p : Parameters) (read : RegionName → Nat → α)
    (_block batch head row : Nat) (i : TileIndex [p.bn,p.dim]) : α :=
  if row+i.1.val<p.length then
    read p.V (offset p.v batch head (row+i.1.val) i.2.1.val)
  else M.literal (some .fp32) p.storage.float 0

def biasStatic (M : Algebra α) (p : Parameters) (read : RegionName → Nat → α)
    (block batch head : Nat) (i : TileIndex [p.bm,p.bn]) : α :=
  if block*p.bm+i.1.val<p.length then
    read p.Bias (batch*p.bias.batch+head*p.bias.head+(block*p.bm+i.1.val)*p.bias.row+(i.2.1.val%64+64))
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

def biasDynamic (M : Algebra α) (p : Parameters) (read : RegionName → Nat → α)
    (block batch head row : Nat) (i : Fin p.bm) : α :=
  if block*p.bm+i.val<p.length then
    read p.Bias (batch*p.bias.batch+head*p.bias.head+(block*p.bm+i.val)*p.bias.row+row/p.bn)
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (i : TileIndex [p.bm,p.bn]) : α :=
  if i.2.1.val<p.length-row then
    M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .add
        (M.cast (some .fp32) p.storage.float .real
          (M.cast (some .fp32) .real p.storage.float (zero M)))
        (M.cast (some .fp32) .fp32 .real
          (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
          (fun j=>ofFloat p.storage.float (query M p read block batch head j))
          (fun j=>ofFloat p.storage.float (key M p read block batch head row j)) i)))
      (M.cast (some .fp32) p.storage.float .real
        (M.binary (some .fp32) p.storage.float .mul
          (M.binary (some .fp32) p.storage.float .add
            (biasDynamic M p read block batch head row i.1) (biasStatic M p read block batch head i))
          (M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 1.44269504))))
  else M.negInf

def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false (scores M dot p read block batch head row) (i,()))

def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.bn]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores M dot p read block batch head row i) (maximum M dot p read block batch head row a i.1))

def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (a.maximum i) (maximum M dot p read block batch head row a i))

def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .fp32 .real
    (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
    (fun j=>ofFloat p.storage.float (M.cast (some .fp32) .real p.storage.float
      (probability M dot p read block batch head row a j)))
    (fun j=>ofFloat p.storage.float (value M p read block batch head row j)) i)

/-- Source-ordered recurrence. The numerical backend is opaque here; neither
associativity nor conversion identities are used in this execution layer. -/
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α) (block batch head : Nat)
    (row : Nat) (a : Acc p α) : Acc p α :=
  ⟨maximum M dot p read block batch head row a,
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p read block batch head row a i))
     (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (probability M dot p read block batch head row a) (i,())),
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.value i) (rescale M dot p read block batch head row a i.1))
     (contribution M dot p read block batch head row a i)⟩

def initial (M : Algebra α) (p : Parameters) : Acc p α :=
  ⟨fun _=>M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _=>zero M,fun _=>zero M⟩

/-- State-free typed recurrence on original input addresses. All tail operands,
casts, reduction steps, accumulator ordering and final division are explicit. -/
def algorithm (M : Algebra α) (dot : Dot α) (p : Parameters) (read : RegionName → Nat → α)
    (block batch head : Nat) (i : TileIndex [p.bm,p.dim]) : α :=
  let final := foldRange (update M dot p read block batch head) 0 p.length p.bn (initial M p)
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .div (final.value i) (final.normalizer i.1))

def nativeReader (c : Call) (words : Nat → BitVec 16) : RegionName → Nat → FamilyNative.Expr :=
  fun r off _=>.ready (.bits ((caller c words r off).zeroExtend 32))

def nativeTarget (plan : FamilyNative.Layout) (c : Call) (words : Nat → BitVec 16)
    (block batch head : Nat) (i : TileIndex [(parameters c).bm,(parameters c).dim]) : Native.Graph :=
  Native.lower (algorithm (FamilyNative.algebra plan) (FamilyNative.dotModel plan)
    (parameters c) (nativeReader c words) block batch head i
      (FamilyNative.physicalOwner plan i.2.1.val))

#specNonCircular nativeTarget avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update]
end
end VeriTile.Bench.Repairs.AttentionKernel.KernelOnly.Specification
