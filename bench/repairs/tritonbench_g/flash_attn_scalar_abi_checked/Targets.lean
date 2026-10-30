import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.flash_attn.Spec

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.Logical
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource PublicIO
noncomputable section

/-- Source-ordered arithmetic on logical tensor rows. Physical strides and
interpreter states do not enter these numerical expressions. -/
def query (M : Algebra A) (D N rowBlock : Nat) (q : Nat → Nat → A) (scale : A)
    (i : TileIndex [128,D]) : A :=
  M.cast (some .fp32) .real .fp16
    (M.binary (some .fp32) .real .mul
      (M.fp32Load (if rowBlock*128+i.1.val<N then q (rowBlock*128+i.1.val) i.2.1.val
        else M.literal (some .fp32) .real 0))
      (M.binary (some .fp32) .real .mul scale (M.literal (some .fp32) .real 1.44269504)))

def key (M : Algebra A) (D N row : Nat) (k : Nat → Nat → A) (i : TileIndex [D,64]) : A :=
  M.fp32Load (if row+i.2.1.val<N then k (row+i.2.1.val) i.1.val else M.literal (some .fp32) .real 0)

def value (M : Algebra A) (D N row : Nat) (v : Nat → Nat → A) (i : TileIndex [64,D]) : A :=
  M.fp32Load (if row+i.1.val<N then v (row+i.1.val) i.2.1.val else M.literal (some .fp32) .real 0)

def scores (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k : Nat → Nat → A) (scale : A) (row : Nat) (i : TileIndex [128,64]) : A :=
  M.binary (some .fp32) .real .add
    (if row+i.2.1.val<p.SEQLEN then
      if p.IS_CAUSAL then
        if row+i.2.1.val≤rowBlock*128+i.1.val then M.literal (some .fp32) .real 0 else M.negInf
      else M.literal (some .fp32) .real 0
     else M.negInf)
    (dot.apply (some .fp32) .real .real .real {} (batch := [])
      (fun j => M.cast (some .fp32) .fp16 .real (query M p.DIM p.SEQLEN rowBlock q scale j))
      (key M p.DIM p.SEQLEN row k) i)

def maximum (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k : Nat → Nat → A) (scale : A) (row : Nat) (a : Acc p A) (i : Fin 128) : A :=
  M.binary (some .fp32) .real .max (a.maximum i)
    (M.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false (scores M dot p rowBlock q k scale row) (i,()))

def probability (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k : Nat → Nat → A) (scale : A) (row : Nat) (a : Acc p A) (i : TileIndex [128,64]) : A :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub
    (scores M dot p rowBlock q k scale row i) (maximum M dot p rowBlock q k scale row a i.1))

def rescale (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k : Nat → Nat → A) (scale : A) (row : Nat) (a : Acc p A) (i : Fin 128) : A :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .sub
    (a.maximum i) (maximum M dot p rowBlock q k scale row a i))

def contribution (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k v : Nat → Nat → A) (scale : A) (row : Nat) (a : Acc p A) (i : TileIndex [128,p.DIM]) : A :=
  dot.apply (some .fp32) .real .real .real {} (batch := [])
    (fun j => M.cast (some .fp32) .fp16 .real (M.cast (some .fp32) .real .fp16
      (probability M dot p rowBlock q k scale row a j)))
    (value M p.DIM p.SEQLEN row v) i

def update (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k v : Nat → Nat → A) (scale : A) (row : Nat) (a : Acc p A) : Acc p A :=
  ⟨maximum M dot p rowBlock q k scale row a,
   fun i => M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.normalizer i) (rescale M dot p rowBlock q k scale row a i))
     (M.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false (probability M dot p rowBlock q k scale row a) (i,())),
   fun i => M.binary (some .fp32) .real .add
     (M.binary (some .fp32) .real .mul (a.output i)
       (M.binary (some .fp32) .real .add
         (M.binary (some .fp32) .real .mul (a.normalizer i.1) (M.literal (some .fp32) .real 0))
         (rescale M dot p rowBlock q k scale row a i.1)))
     (contribution M dot p rowBlock q k v scale row a i)⟩

def initial (M : Algebra A) (p : Parameters) : Acc p A :=
  ⟨fun _ => M.binary (some .fp32) .real .add (M.literal (some .fp32) .real 0) M.negInf,
   fun _ => M.literal (some .fp32) .real 0,fun _ => M.literal (some .fp32) .real 0⟩

def accumulated (M : Algebra A) (dot : Dot A) (p : Parameters) (rowBlock : Nat)
    (q k v : Nat → Nat → A) (scale : A) : Acc p A :=
  ActualSource.foldRange (update M dot p rowBlock q k v scale) 0
    (if p.IS_CAUSAL then min ((rowBlock+1)*128) p.SEQLEN else p.SEQLEN) 64 (initial M p)

def output (M : Algebra A) (p : Parameters) (a : Acc p A) (i : TileIndex [128,p.DIM]) : A :=
  M.cast (some .fp32) .fp16 .real (M.cast (some .fp32) .real .fp16
    (M.binary (some .fp32) .real .div (a.output i) (a.normalizer i.1)))

def log (M : Algebra A) (p : Parameters) (a : Acc p A) (i : Fin 128) : A :=
  M.binary (some .fp32) .real .add (a.maximum i) (M.unary (some .fp32) .log2 (a.normalizer i))

def rowAccumulator [Inhabited A] (M : Algebra B) (dot : Dot B) (encode : A → B)
    (x : Layout) (scale : B) (xs : Inputs A x) (job : Program) : Acc (parameters x) B :=
  accumulated M dot (parameters x) job.1
    (fun r d => encode (inputAt xs 0 (job.2/x.shape.heads) (job.2%x.shape.heads) r d))
    (fun r d => encode (inputAt xs 1 (job.2/x.shape.heads) (job.2%x.shape.heads) r d))
    (fun r d => encode (inputAt xs 2 (job.2/x.shape.heads) (job.2%x.shape.heads) r d)) scale

/-- Writer selection is geometry only. It preserves the original native
domain even when output strides make distinct logical coordinates alias. -/
abbrev Writer (x : Layout) := Program × TileIndex [128,x.shape.dim]
def writerAddress (x : Layout) (w : Writer x) : Nat :=
  FlashAttn.address x.o (w.1.2/x.shape.heads) (w.1.2%x.shape.heads)
    (w.1.1*128+w.2.1.val) w.2.2.1.val

def writers (x : Layout) : List (Writer x) :=
  (programs (parameters x)).flatMap (fun job =>
    ((TileShape.allIndices [128,x.shape.dim]).filter (fun i => decide (job.1*128+i.1.val<x.shape.length))).map
      (fun i => (job,i)))

def lastWriter (x : Layout) (address : Nat) : Option (Writer x) :=
  ((writers x).filter (fun w => decide (writerAddress x w=address))).getLast?

end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.Logical
