import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Reference
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Grid
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly.Specification
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
noncomputable section
variable {α : Type} [Inhabited α]
/-- Independent state-free typed attention recurrence on original address reads.
All masks, casts, reductions and online updates are explicit. -/
abbrev Row (p : Parameters) := TileIndex [p.blockM]
abbrev QLane (p : Parameters) := TileIndex [p.blockM,p.head]
abbrev KLane (p : Parameters) := TileIndex [p.head,p.blockN]
abbrev VLane (p : Parameters) := TileIndex [p.blockN,p.head]
abbrev ScoreLane (p : Parameters) := TileIndex [p.blockM,p.blockN]
def row (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : Fin p.blockM) := x*p.blockM+i.val
def headBase (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) := b*p.qBatchStride+h*p.qHeadStride
def kvBase (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) := b*p.kBatchStride+h*p.kHeadStride
def valueBase (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) := b*p.vBatchStride+h*p.vHeadStride
def outputBase (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) := b*p.outBatchStride+h*p.outHeadStride
def biasBase (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) := b*p.biasBatchStride+h*p.biasHeadStride
def qOffset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : QLane p) := headBase p read x b h+row p read x b h i.1*p.qRowStride+i.2.1.val*p.qColStride
def kOffset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : KLane p) := kvBase p read x b h+i.1.val*p.kDimStride+(c*p.blockN+i.2.1.val)*p.kSeqStride
def vOffset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : VLane p) := valueBase p read x b h+(c*p.blockN+i.1.val)*p.vSeqStride+i.2.1.val*p.vDimStride
def b1Offset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : ScoreLane p) := biasBase p read x b h+row p read x b h i.1*p.biasRowStride+(i.2.1.val%p.biasLast+p.biasLast)
def b0Offset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : Fin p.blockM) := biasBase p read x b h+row p read x b h i*p.biasRowStride+c
def outputOffset (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : QLane p) :=
  outputBase p read x b h+row p read x b h i.1*p.outRowStride+i.2.1.val*p.outColStride
def qValue (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : QLane p) : α :=
  if row p read x b h i.1<p.length then read p.q (qOffset p read x b h i) else P.literal none .fp16 0
def kValue (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : KLane p) : α :=
  if c*p.blockN+i.2.1.val<p.length then read p.k (kOffset p read x b h c i) else P.literal none .fp16 0
def vValue (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : VLane p) : α :=
  if c*p.blockN+i.1.val<p.length then read p.v (vOffset p read x b h c i) else P.literal none .fp16 0
def b1Value (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : ScoreLane p) : α :=
  if row p read x b h i.1<p.length then read p.bias (b1Offset p read x b h i) else P.cast none .real .fp16 (P.literal none .real 0)
def b0Value (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : Fin p.blockM) : α :=
  if row p read x b h i<p.length then read p.bias (b0Offset p read x b h c i) else P.cast none .real .fp16 (P.literal none .real 0)
def scaleValue (P : Primitives α) (p : Parameters) : α :=
  P.binary none .real .mul (P.literal none .real p.scale) (P.literal none .real 1.44269504)
def qScaled (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : QLane p) : α :=
  P.cast none .real .fp16 (P.binary (some .fp32) .real .mul
    (P.cast (some .fp32) .fp16 .real (qValue P p read x b h i)) (scaleValue P p))
def halfZero (P : Primitives α) := P.cast none .real .fp16 (P.literal none .real 0)
def qkDot (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) : ScoreLane p → α :=
  P.dot none .fp16 .fp16 .fp16 {outDType:=some .fp16} (batch:=[])
    (m:=p.blockM) (k:=p.head) (n:=p.blockN)
    (qScaled P p read x b h) (kValue P p read x b h c)
def rawScore (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : ScoreLane p) : α :=
  P.binary none .fp16 .add (P.binary none .fp16 .add (halfZero P) (qkDot P p read x b h c i))
    (P.binary none .fp16 .add (b0Value P p read x b h c i.1) (b1Value P p read x b h i))
def score (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : ScoreLane p) : α :=
  if c*p.blockN+i.2.1.val<p.length then rawScore P p read x b h c i else P.cast (some .fp32) .real .fp16 P.negInf
def score32 (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat) (i : ScoreLane p) : α :=
  P.cast (some .fp32) .fp16 .real (score P p read x b h c i)
structure Accumulator (α : Type) (p : Parameters) where
  maximum : Row p → α
  denominator : Row p → α
  numerator : QLane p → α
def newMax (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat)
    (st : Accumulator α p) (i : Row p) : α :=
  P.binary (some .fp32) .real .max (st.maximum i)
    (P.max (some .fp32) ⟨1,by simp⟩ Bool.false (score32 P p read x b h c) i)
def alphaValue (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat)
    (st : Accumulator α p) (i : Row p) : α :=
  P.unary (some .fp32) .exp2 (P.binary (some .fp32) .real .sub (st.maximum i) (newMax P p read x b h c st i))
def weights (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat)
    (st : Accumulator α p) (i : ScoreLane p) : α :=
  P.unary (some .fp32) .exp2 (P.binary (some .fp32) .real .sub
    (score32 P p read x b h c i) (newMax P p read x b h c st (i.1,PUnit.unit)))
def weightedDot (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat)
    (st : Accumulator α p) : QLane p → α :=
  P.dot (some .fp32) .fp16 .fp16 .fp32 {} (batch:=[])
    (m:=p.blockM) (k:=p.blockN) (n:=p.head)
    (fun i => P.cast (some .fp32) .real .fp16 (weights P p read x b h c st i)) (vValue P p read x b h c)
def advance (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (c : Nat)
    (st : Accumulator α p) : Accumulator α p :=
  { maximum := newMax P p read x b h c st
    denominator := fun i => P.binary (some .fp32) .real .add
      (P.binary (some .fp32) .real .mul (st.denominator i) (alphaValue P p read x b h c st i))
      (P.sum (some .fp32) ⟨1,by simp⟩ Bool.false (weights P p read x b h c st) i)
    numerator := fun i => P.binary (some .fp32) .real .add
      (P.binary (some .fp32) .real .mul (st.numerator i) (alphaValue P p read x b h c st (i.1,PUnit.unit)))
      (P.cast (some .fp32) .fp32 .real (weightedDot P p read x b h c st i)) }
def initialAccumulator (P : Primitives α) (p : Parameters) : Accumulator α p :=
  ⟨fun _ => P.binary (some .fp32) .real .add (P.literal (some .fp32) .real 0) P.negInf,
   fun _ => P.literal (some .fp32) .real 0, fun _ => P.literal (some .fp32) .real 0⟩
def rounds (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) : Nat → Accumulator α p
  | 0 => initialAccumulator P p
  | c+1 => advance P p read x b h c (rounds P p read x b h c)
def outputValue (P : Primitives α) (p : Parameters) (read : RegionName → Nat → α) (x b h : Nat) (i : QLane p) : α :=
  let st := rounds P p read x b h p.blocks
  P.cast none .real .fp16 (P.binary (some .fp32) .real .div
    (st.numerator i) (st.denominator (i.1,PUnit.unit)))

#specNonCircular outputValue avoiding [FP.Structural.exec,
  VeriTile.Bench.Repairs.AttentionKernelAligned.Whole.runPrograms,
  VeriTile.Bench.Repairs.AttentionKernelAligned.Whole.answer,
  VeriTile.Bench.Repairs.AttentionKernelAligned.Source.rounds,
  VeriTile.Bench.Repairs.AttentionKernelAligned.Source.advance]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.KernelOnly.Specification
