import bench.tritonbench_g.triton_attention.Proofs.FPBackend
namespace VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
structure ForwardParameters where
  dtype : FloatDType
  (q k v denominator maximum output : RegionName)
  scale : ℝ
  (headStride context rows blockM blockN head : Nat)
abbrev FRow (p : ForwardParameters) := TileIndex [p.blockM]
abbrev FQuery (p : ForwardParameters) := TileIndex [p.blockM,p.head]
abbrev FKey (p : ForwardParameters) := TileIndex [p.blockN,p.head]
abbrev FScore (p : ForwardParameters) := TileIndex [p.blockM,p.blockN]
def forwardSource (p : ForwardParameters) := forwardTyped p.dtype .fp32
  p.q p.k p.v p.denominator p.maximum p.output p.scale
  0 p.headStride p.head 1 0 p.headStride p.head 1 0 p.headStride p.head 1
  0 p.headStride p.head 1 0 0 p.context p.rows p.blockM p.head p.blockN
def forwardPrologue (p : ForwardParameters) := (forwardSource p).surfaceBody.take 13
def forwardBody (p : ForwardParameters) : List ComputeStmt :=
  match ((forwardSource p).surfaceBody.drop 13).head? with
  | some (.forRangeDyn _ _ _ _ b) => b | _ => []
def forwardEpilogue (p : ForwardParameters) := (forwardSource p).surfaceBody.drop 14

theorem forwardDecomposition (p : ForwardParameters) :
    (forwardSource p).surfaceBody = forwardPrologue p ++
      [.forRangeDyn "start_n" (.constNat 0)
        (.mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat p.blockM))
        (.constNat p.blockN) (forwardBody p)] ++ forwardEpilogue p := rfl

def forwardHeadRow (p : ForwardParameters) (s : State α) := s.pids 1*(p.headStride/p.head)
def forwardQueryRow (p : ForwardParameters) (s : State α) (i : Fin p.blockM) := s.pids 0*p.blockM+i.val
def forwardRowOffset (p : ForwardParameters) (s : State α) (i : FRow p) :=
  s.pids 1*p.context+forwardQueryRow p s i.1
def forwardOffset (p : ForwardParameters) (s : State α) (i : FQuery p) :=
  (forwardHeadRow p s+forwardQueryRow p s i.1)*p.head+i.2.1.val
def forwardKVOffset (p : ForwardParameters) (s : State α) (c : Nat) (i : FKey p) :=
  (forwardHeadRow p s+c*p.blockN+i.1.val)*p.head+i.2.1.val
def forwardStop (p : ForwardParameters) (s : State α) := (s.pids 0+1)*p.blockM
def forwardBlocks (p : ForwardParameters) (s : State α) := forwardStop p s/p.blockN

def forwardQPointer (p : ForwardParameters) (s : State α) : BlockPtr :=
  ⟨p.q,0,[p.rows,p.head],[p.blockM,p.head],[p.head,1],[forwardHeadRow p s+s.pids 0*p.blockM,0]⟩
def forwardKVPointer (p : ForwardParameters) (s : State α) (r : RegionName) (c : Nat) : BlockPtr :=
  ⟨r,0,[p.rows,p.head],[p.blockN,p.head],[p.head,1],[forwardHeadRow p s+c*p.blockN,0]⟩
def forwardOutPointer (p : ForwardParameters) (s : State α) : BlockPtr := {forwardQPointer p s with region:=p.output}
def forwardQ (p : ForwardParameters) (s : State α) : Values α p.dtype.toTileDType [p.blockM,p.head] :=
  fun i => (s.mem p.q (forwardOffset p s i)).read p.dtype.toTileDType
/-- Boundary-check without a specified padding value retains the typed undef oracle. -/
def forwardKV (p : ForwardParameters) (s : State α) (r : RegionName) (c : Nat) : Values α p.dtype.toTileDType [p.blockN,p.head] :=
  fun i => if (forwardKVPointer p s r c).inBounds (TileShape.indexToList _ i) [0,1]
    then (s.mem r (forwardKVOffset p s c i)).read p.dtype.toTileDType
    else s.undef p.dtype.toTileDType r (forwardKVOffset p s c i)
def forwardZero (P : Primitives α) := P.literal (some .fp32) .real 0
structure ForwardAccumulator (α : Type) (p : ForwardParameters) where
  maximum : FRow p → α
  denominator : FRow p → α
  numerator : FQuery p → α

def forwardDot (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat) : FScore p → α :=
  P.dot (some .fp32) p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.head) (n:=p.blockN)
    (forwardQ p s) (fun i => forwardKV p s p.k c (i.2.1,i.1,PUnit.unit))
def forwardScore (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat) (i : FScore p) : α :=
  if forwardQueryRow p s i.1 ≥ c*p.blockN+i.2.1.val then
    P.binary (some .fp32) .real .mul
      (P.binary (some .fp32) .real .add (forwardZero P)
        (P.cast (some .fp32) .fp32 .real (forwardDot P p s c i)))
      (P.literal (some .fp32) .real p.scale)
  else P.negInf

def forwardMax (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FRow p) :=
  P.binary (some .fp32) .real .max
    (P.max (some .fp32) ⟨1,by simp⟩ Bool.false (forwardScore P p s c) i) (st.maximum i)
def forwardOldDenominator (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FRow p) :=
  P.binary (some .fp32) .real .mul (st.denominator i)
    (P.unary (some .fp32) .exp (P.binary (some .fp32) .real .sub (st.maximum i) (forwardMax P p s c st i)))
def forwardWeights (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FScore p) :=
  P.unary (some .fp32) .exp (P.binary (some .fp32) .real .sub
    (forwardScore P p s c i) (forwardMax P p s c st (i.1,PUnit.unit)))
def forwardDenominator (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FRow p) :=
  P.binary (some .fp32) .real .add
    (P.sum (some .fp32) ⟨1,by simp⟩ Bool.false (forwardWeights P p s c st) i)
    (forwardOldDenominator P p s c st i)
def forwardReciprocal (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FRow p) :=
  P.binary (some .fp32) .real .div (P.literal (some .fp32) .real 1) (forwardDenominator P p s c st i)
def forwardNormalized (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FScore p) :=
  P.binary (some .fp32) .real .mul (forwardWeights P p s c st i) (forwardReciprocal P p s c st (i.1,PUnit.unit))
def forwardHalfWeights (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FScore p) :=
  P.cast none .real .fp16 (forwardNormalized P p s c st i)
def forwardRescaled (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) (i : FQuery p) :=
  P.binary (some .fp32) .real .mul (st.numerator i)
    (P.binary (some .fp32) .real .mul (forwardOldDenominator P p s c st (i.1,PUnit.unit))
      (forwardReciprocal P p s c st (i.1,PUnit.unit)))
def forwardValueDot (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) : FQuery p → α :=
  P.dot (some .fp32) .fp16 p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.blockN) (n:=p.head)
    (forwardHalfWeights P p s c st) (forwardKV p s p.v c)
def forwardAdvance (P : Primitives α) (p : ForwardParameters) (s : State α) (c : Nat)
    (st : ForwardAccumulator α p) : ForwardAccumulator α p :=
  ⟨forwardMax P p s c st,forwardDenominator P p s c st,
    fun i => P.binary (some .fp32) .real .add (forwardRescaled P p s c st i)
      (P.cast (some .fp32) .fp32 .real (forwardValueDot P p s c st i))⟩
def forwardInitial (P : Primitives α) (p : ForwardParameters) : ForwardAccumulator α p :=
  ⟨fun _ => P.binary (some .fp32) .real .add (forwardZero P) P.negInf,
    fun _ => forwardZero P,fun _ => forwardZero P⟩
def forwardRounds (P : Primitives α) (p : ForwardParameters) (s : State α) : Nat → ForwardAccumulator α p
  | 0 => forwardInitial P p
  | c+1 => forwardAdvance P p s c (forwardRounds P p s c)
def forwardOutput (P : Primitives α) (p : ForwardParameters) (s : State α) (i : FQuery p) : Cell α :=
  .mk p.dtype.toTileDType (ofFloat p.dtype (P.cast none .fp16 p.dtype
    (P.cast none .real .fp16 ((forwardRounds P p s (forwardBlocks p s)).numerator i))))
def forwardLCell (P : Primitives α) (p : ForwardParameters) (s : State α) (i : FRow p) : Cell α :=
  .mk .fp32 (P.cast none .real .fp32 ((forwardRounds P p s (forwardBlocks p s)).denominator i))
def forwardMCell (P : Primitives α) (p : ForwardParameters) (s : State α) (i : FRow p) : Cell α :=
  .mk .fp32 (P.cast none .real .fp32 ((forwardRounds P p s (forwardBlocks p s)).maximum i))
def forwardWrite (P : Primitives α) (p : ForwardParameters) (initial current : State α) : State α :=
  let t := (TileShape.allIndices [p.blockM]).foldl (fun t i =>
    t.write p.denominator (forwardRowOffset p initial i) (forwardLCell P p initial i)) current
  let t := (TileShape.allIndices [p.blockM]).foldl (fun t i =>
    t.write p.maximum (forwardRowOffset p initial i) (forwardMCell P p initial i)) t
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun t i =>
    if (forwardOutPointer p initial).inBounds (TileShape.indexToList _ i) [0,1] then
      t.write p.output (forwardOffset p initial i) (forwardOutput P p initial i) else t) t

#specNonCircular forwardWrite avoiding [forwardSource,FP.Structural.exec,run,step]
#axiomsClean forwardDecomposition
end
end VeriTile.Bench.TritonBenchG.TritonAttention.FPValue
