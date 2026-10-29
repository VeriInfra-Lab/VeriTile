import bench.tritonbench_g.attention_kernel_aligned.Proofs.FullFp16Source
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullFp16
open VeriTile.Triton FP.Structural
open Full
open FPValue (Primitives backend)
noncomputable section
variable {α : Type} [Inhabited α]
abbrev Row (p : Parameters) := TileIndex [p.blockM]
abbrev QLane (p : Parameters) := TileIndex [p.blockM,p.head]
abbrev KLane (p : Parameters) := TileIndex [p.head,p.blockN]
abbrev VLane (p : Parameters) := TileIndex [p.blockN,p.head]
abbrev ScoreLane (p : Parameters) := TileIndex [p.blockM,p.blockN]
def row (p : Parameters) (s : State α) (i : Fin p.blockM) := s.pids 0*p.blockM+i.val
def headBase (p : Parameters) (s : State α) := s.pids 1*p.qHeadStride
def kvBase (p : Parameters) (s : State α) := s.pids 1*p.kHeadStride
def biasBase (p : Parameters) (s : State α) := s.pids 1*p.biasHeadStride
def qOffset (p : Parameters) (s : State α) (i : QLane p) := headBase p s+row p s i.1*p.qRowStride+i.2.1.val*p.qColStride
def kOffset (p : Parameters) (s : State α) (c : Nat) (i : KLane p) := kvBase p s+i.1.val*p.kDimStride+(c*p.blockN+i.2.1.val)*p.kSeqStride
def vOffset (p : Parameters) (s : State α) (c : Nat) (i : VLane p) := kvBase p s+(c*p.blockN+i.1.val)*p.vSeqStride+i.2.1.val*p.vDimStride
def b1Offset (p : Parameters) (s : State α) (i : ScoreLane p) := biasBase p s+row p s i.1*p.biasRowStride+(i.2.1.val%p.biasLast+p.biasLast)
def b0Offset (p : Parameters) (s : State α) (c : Nat) (i : Fin p.blockM) := biasBase p s+row p s i*p.biasRowStride+c
def outputOffset (p : Parameters) (s : State α) (i : QLane p) :=
  headBase p s+row p s i.1*p.outRowStride+i.2.1.val*p.outColStride
def qValue (p : Parameters) (s : State α) (i : QLane p) : α := (s.mem p.q (qOffset p s i)).read .fp16
def kValue (p : Parameters) (s : State α) (c : Nat) (i : KLane p) : α := (s.mem p.k (kOffset p s c i)).read .fp16
def vValue (p : Parameters) (s : State α) (c : Nat) (i : VLane p) : α := (s.mem p.v (vOffset p s c i)).read .fp16
def b1Value (p : Parameters) (s : State α) (i : ScoreLane p) : α := (s.mem p.bias (b1Offset p s i)).read .fp16
def b0Value (p : Parameters) (s : State α) (c : Nat) (i : Fin p.blockM) : α := (s.mem p.bias (b0Offset p s c i)).read .fp16
def scaleValue (P : Primitives α) (p : Parameters) : α :=
  P.binary none .real .mul (P.literal none .real p.scale) (P.literal none .real 1.44269504)
def qScaled (P : Primitives α) (p : Parameters) (s : State α) (i : QLane p) : α :=
  P.cast none .real .fp16 (P.binary (some .fp32) .real .mul
    (P.cast (some .fp32) .fp16 .real (qValue p s i)) (scaleValue P p))
def halfZero (P : Primitives α) := P.cast none .real .fp16 (P.literal none .real 0)
def qkDot (P : Primitives α) (p : Parameters) (s : State α) (c : Nat) : ScoreLane p → α :=
  P.dot none .fp16 .fp16 .fp16 {outDType:=some .fp16} (batch:=[])
    (m:=p.blockM) (k:=p.head) (n:=p.blockN)
    (qScaled P p s) (kValue (α:=α) p s c)
def score (P : Primitives α) (p : Parameters) (s : State α) (c : Nat) (i : ScoreLane p) : α :=
  P.binary none .fp16 .add (P.binary none .fp16 .add (halfZero P) (qkDot P p s c i))
    (P.binary none .fp16 .add (b0Value p s c i.1) (b1Value p s i))
def score32 (P : Primitives α) (p : Parameters) (s : State α) (c : Nat) (i : ScoreLane p) : α :=
  P.cast (some .fp32) .fp16 .real (score P p s c i)
structure Accumulator (α : Type) (p : Parameters) where
  maximum : Row p → α
  denominator : Row p → α
  numerator : QLane p → α
def newMax (P : Primitives α) (p : Parameters) (s : State α) (c : Nat)
    (st : Accumulator α p) (i : Row p) : α :=
  P.binary (some .fp32) .real .max (st.maximum i)
    (P.max (some .fp32) ⟨1,by simp⟩ Bool.false (score32 P p s c) i)
def alphaValue (P : Primitives α) (p : Parameters) (s : State α) (c : Nat)
    (st : Accumulator α p) (i : Row p) : α :=
  P.unary (some .fp32) .exp2 (P.binary (some .fp32) .real .sub (st.maximum i) (newMax P p s c st i))
def weights (P : Primitives α) (p : Parameters) (s : State α) (c : Nat)
    (st : Accumulator α p) (i : ScoreLane p) : α :=
  P.unary (some .fp32) .exp2 (P.binary (some .fp32) .real .sub
    (score32 P p s c i) (newMax P p s c st (i.1,PUnit.unit)))
def weightedDot (P : Primitives α) (p : Parameters) (s : State α) (c : Nat)
    (st : Accumulator α p) : QLane p → α :=
  P.dot (some .fp32) .fp16 .fp16 .fp32 {} (batch:=[])
    (m:=p.blockM) (k:=p.blockN) (n:=p.head)
    (fun i => P.cast (some .fp32) .real .fp16 (weights P p s c st i)) (vValue (α:=α) p s c)
def advance (P : Primitives α) (p : Parameters) (s : State α) (c : Nat)
    (st : Accumulator α p) : Accumulator α p :=
  { maximum := newMax P p s c st
    denominator := fun i => P.binary (some .fp32) .real .add
      (P.binary (some .fp32) .real .mul (st.denominator i) (alphaValue P p s c st i))
      (P.sum (some .fp32) ⟨1,by simp⟩ Bool.false (weights P p s c st) i)
    numerator := fun i => P.binary (some .fp32) .real .add
      (P.binary (some .fp32) .real .mul (st.numerator i) (alphaValue P p s c st (i.1,PUnit.unit)))
      (P.cast (some .fp32) .fp32 .real (weightedDot P p s c st i)) }
def initialAccumulator (P : Primitives α) (p : Parameters) : Accumulator α p :=
  ⟨fun _ => P.binary (some .fp32) .real .add (P.literal (some .fp32) .real 0) P.negInf,
   fun _ => P.literal (some .fp32) .real 0, fun _ => P.literal (some .fp32) .real 0⟩
def rounds (P : Primitives α) (p : Parameters) (s : State α) : Nat → Accumulator α p
  | 0 => initialAccumulator P p
  | c+1 => advance P p s c (rounds P p s c)
def outputValue (P : Primitives α) (p : Parameters) (s : State α) (i : QLane p) : α :=
  let st := rounds P p s p.blocks
  P.cast none .real .fp16 (P.binary (some .fp32) .real .div
    (st.numerator i) (st.denominator (i.1,PUnit.unit)))
def written (P : Primitives α) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    t.write p.output (outputOffset p s i) (.mk .fp16 (outputValue P p s i))) s

#specNonCircular written avoiding [source, FP.Structural.exec]
end
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FullFp16
