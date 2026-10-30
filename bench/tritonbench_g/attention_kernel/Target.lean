import bench.tritonbench_g.attention_kernel.IO
import bench.tritonbench_g.attention_kernel.Proofs.Specification

namespace VeriTile.Bench.Repairs.AttentionKernel.Target
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 20000

/-- Exact finite attention formula, including both relative-bias windows and
source coefficient. No recurrence or evaluator defines the mathematical result. -/
def mathematical (p : Parameters) (vals : ChannelValues ℝ (IO.inputs p)) :
    ChannelValues ℝ (IO.outputs p) :=
  IO.result p (fun i=>
    let x:=views p vals i.1 i.2.1
    RealMath.expected p.length p.dim p.bn p.scale x.q x.k x.v x.bias
      i.2.2.1.val i.2.2.2.1.val)

namespace Native
variable {α : Type} [Inhabited α]

def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def query (M : Algebra α) (p : Parameters) (inputs : Views α)
    (block : Nat) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) p.storage.float .real
        (if block*p.bm+i.1.val<p.length then
          inputs.q (block*p.bm+i.1.val) i.2.1.val
         else M.literal (some .fp32) p.storage.float 0))
      (M.binary (some .fp32) .real .mul
        (M.literal (some .fp32) .real p.scale) (M.literal (some .fp32) .real 1.44269504)))

def key (M : Algebra α) (p : Parameters) (inputs : Views α)
    (_block row : Nat) (i : TileIndex [p.dim,p.bn]) : α :=
  if row+i.2.1.val<p.length then
    inputs.k (row+i.2.1.val) i.1.val
  else M.literal (some .fp32) p.storage.float 0

def value (M : Algebra α) (p : Parameters) (inputs : Views α)
    (_block row : Nat) (i : TileIndex [p.bn,p.dim]) : α :=
  if row+i.1.val<p.length then
    inputs.v (row+i.1.val) i.2.1.val
  else M.literal (some .fp32) p.storage.float 0

def biasStatic (M : Algebra α) (p : Parameters) (inputs : Views α)
    (block : Nat) (i : TileIndex [p.bm,p.bn]) : α :=
  if block*p.bm+i.1.val<p.length then
    inputs.bias (block*p.bm+i.1.val) (i.2.1.val%64+64)
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

def biasDynamic (M : Algebra α) (p : Parameters) (inputs : Views α)
    (block row : Nat) (i : Fin p.bm) : α :=
  if block*p.bm+i.val<p.length then
    inputs.bias (block*p.bm+i.val) (row/p.bn)
  else M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 0)

def scores (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (i : TileIndex [p.bm,p.bn]) : α :=
  if i.2.1.val<p.length-row then
    M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .add
        (M.cast (some .fp32) p.storage.float .real
          (M.cast (some .fp32) .real p.storage.float (zero M)))
        (M.cast (some .fp32) .fp32 .real
          (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
          (fun j=>ofFloat p.storage.float (query M p inputs block j))
          (fun j=>ofFloat p.storage.float (key M p inputs block row j)) i)))
      (M.cast (some .fp32) p.storage.float .real
        (M.binary (some .fp32) p.storage.float .mul
          (M.binary (some .fp32) p.storage.float .add
            (biasDynamic M p inputs block row i.1) (biasStatic M p inputs block i))
          (M.cast (some .fp32) .real p.storage.float (M.literal (some .fp32) .real 1.44269504))))
  else M.negInf

def maximum (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false (scores M dot p inputs block row) (i,()))

def probability (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.bn]) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (scores M dot p inputs block row i) (maximum M dot p inputs block row a i.1))

def rescale (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (a : Acc p α) (i : Fin p.bm) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .sub (a.maximum i) (maximum M dot p inputs block row a i))

def contribution (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (a : Acc p α) (i : TileIndex [p.bm,p.dim]) : α :=
  M.cast (some .fp32) .fp32 .real
    (dot.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
    (fun j=>ofFloat p.storage.float (M.cast (some .fp32) .real p.storage.float
      (probability M dot p inputs block row a j)))
    (fun j=>ofFloat p.storage.float (value M p inputs block row j)) i)

/-- Source-ordered recurrence. The numerical backend is opaque here; neither
associativity nor conversion identities are used in this execution layer. -/
def update (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α) (block : Nat)
    (row : Nat) (a : Acc p α) : Acc p α :=
  ⟨maximum M dot p inputs block row a,
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p inputs block row a i))
     (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (probability M dot p inputs block row a) (i,())),
   fun i=>M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.value i) (rescale M dot p inputs block row a i.1))
     (contribution M dot p inputs block row a i)⟩

def initial (M : Algebra α) (p : Parameters) : Acc p α :=
  ⟨fun _=>M.binary (some .fp32) .real .add (zero M) M.negInf,
   fun _=>zero M,fun _=>zero M⟩

/-- Typed numerical recurrence on independent logical input arrays. All tail operands,
casts, reduction steps, accumulator ordering and final division are explicit. -/
def algorithm (M : Algebra α) (dot : Dot α) (p : Parameters) (inputs : Views α)
    (block : Nat) (i : TileIndex [p.bm,p.dim]) : α :=
  let final := foldRange (update M dot p inputs block) 0 p.length p.bn (initial M p)
  M.cast (some .fp32) .real p.storage.float
    (M.binary (some .fp32) .real .div (final.value i) (final.normalizer i.1))

end Native

def native (I : KernelOnly.Instructions) (layout : FamilyNative.Layout)
    (p : Parameters) (hbm : 0<p.bm) (vals : ChannelValues (BitVec 16) (IO.inputs p)) :
    ChannelValues (BitVec 16) (IO.outputs p) :=
  IO.result p (fun i=>
    let x:=views p vals i.1 i.2.1
    let terms : Views FamilyNative.Expr:=
      ⟨fun r d=>IO.encodeWord (x.q r d),fun r d=>IO.encodeWord (x.k r d),
       fun r d=>IO.encodeWord (x.v r d),fun r c=>IO.encodeWord (x.bias r c)⟩
    IO.observeWord I layout p i
      (Native.algorithm (FamilyNative.algebra layout) (FamilyNative.dotModel layout) p terms
        (i.2.2.1.val/p.bm) (lane p hbm i.2.2.1.val i.2.2.2.1.val i.2.2.2.1.isLt)))

#specNonCircular mathematical avoiding [State,run,FP.Structural.exec,Whole.answer,Source.update,IO.attentionIO]
#specNonCircular native avoiding [State,run,FP.Structural.exec,Whole.answer,Source.update,IO.attentionIO]
end
end VeriTile.Bench.Repairs.AttentionKernel.Target
