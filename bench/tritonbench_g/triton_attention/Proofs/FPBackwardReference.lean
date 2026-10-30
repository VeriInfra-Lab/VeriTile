import bench.tritonbench_g.triton_attention.Proofs.FPBackend
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]
structure BackwardParameters where
  dtype : FloatDType
  (q k v gradient dq dk dv maximum delta : RegionName)
  scale : ℝ
  (batchStride headStride heads rows blockM head blocks : Nat)
abbrev BLane (p : BackwardParameters) := TileIndex [p.blockM,p.head]
abbrev BScore (p : BackwardParameters) := TileIndex [p.blockM,p.blockM]
abbrev BRow (p : BackwardParameters) := TileIndex [p.blockM]
abbrev Memory (α : Type) := RegionName → Nat → Cell α

def backwardSource (p : BackwardParameters) := backwardTyped p.dtype .fp32
  p.q p.k p.v "unused_Out" p.gradient p.dq p.dk p.dv "unused_L" p.maximum p.delta p.scale
  p.batchStride p.headStride p.head 1 0 0 p.head 1 0 0 p.head 1
  0 p.heads (p.blockM*p.blocks) p.rows p.blocks p.blockM p.head p.blockM

def backwardPrologue (p : BackwardParameters) := (backwardSource p).surfaceBody.take 13
def backwardOuterBody (p : BackwardParameters) : List ComputeStmt :=
  match ((backwardSource p).surfaceBody.drop 13).head? with
  | some (.forRange _ _ _ _ b) => b
  | _ => []
def backwardInnerBody (p : BackwardParameters) : List ComputeStmt :=
  match ((backwardOuterBody p).drop 12).head? with
  | some (.forRangeDyn _ _ _ _ b) => b
  | _ => []

theorem backwardDecomposition (p : BackwardParameters) :
    (backwardSource p).surfaceBody=backwardPrologue p ++
      [.forRange "start_n" 0 p.blocks 1 (backwardOuterBody p)] := rfl

theorem backwardInnerDecomposition (p : BackwardParameters) :
    backwardOuterBody p=(backwardOuterBody p).take 12 ++
      [.forRangeDyn "start_m" (.ref .nat [] "lo")
        (.mul .nat .nil (.constNat p.blocks) (.constNat p.blockM))
        (.constNat p.blockM) (backwardInnerBody p)] ++ (backwardOuterBody p).drop 13 := rfl

def backwardHeadRow (p : BackwardParameters) (s : State α) :=
  (s.pids 0/p.heads)*(p.batchStride/p.head)+(s.pids 0%p.heads)*(p.headStride/p.head)
def backwardOffset (p : BackwardParameters) (s : State α) (c : Nat) (i : BLane p) :=
  (backwardHeadRow p s+c*p.blockM+i.1.val)*p.head+i.2.1.val
def backwardRowOffset (p : BackwardParameters) (s : State α) (c : Nat) (i : BRow p) :=
  s.pids 0*(p.blockM*p.blocks)+c*p.blockM+i.1.val
def backwardPointer (p : BackwardParameters) (s : State α) (r : RegionName) (c : Nat) : BlockPtr :=
  ⟨r,0,[p.rows,p.head],[p.blockM,p.head],[p.head,1],[backwardHeadRow p s+c*p.blockM,0]⟩
def backwardRead (p : BackwardParameters) (s : State α) (mem : Memory α) (r : RegionName) (c : Nat) :
    Values α p.dtype.toTileDType [p.blockM,p.head] := fun i =>
  if (backwardPointer p s r c).inBounds (TileShape.indexToList _ i) [0,1] then
    (mem r (backwardOffset p s c i)).read p.dtype.toTileDType
  else s.undef p.dtype.toTileDType r (backwardOffset p s c i)
structure BackwardInner (α : Type) (p : BackwardParameters) where
  mem : Memory α
  dk : BLane p → α
  dv : BLane p → α

def backwardQK (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key : Values α p.dtype.toTileDType [p.blockM,p.head]) (queryBlock : Nat) : BScore p → α :=
  P.dot none p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.head) (n:=p.blockM)
    (backwardRead p s mem p.q queryBlock) (fun i => key (i.2.1,i.1,PUnit.unit))
def backwardMaskedQK (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) (i : BScore p) : α :=
  if queryBlock*p.blockM+i.1.val ≥ keyBlock*p.blockM+i.2.1.val then
    P.cast (some .fp32) .fp32 .real (backwardQK P p s mem key queryBlock i) else P.negInf
def backwardWeights (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) (i : BScore p) : α :=
  P.unary none .exp (P.binary (some .fp32) .real .sub
    (P.binary (some .fp32) .real .mul (backwardMaskedQK P p s mem key keyBlock queryBlock i)
      (P.literal (some .fp32) .real p.scale))
    (P.cast (some .fp32) .fp32 .real ((mem p.maximum (backwardRowOffset p s queryBlock (i.1,PUnit.unit))).read .fp32)))
def backwardValueDot (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) : BLane p → α :=
  P.dot (some .fp32) p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.blockM) (n:=p.head)
    (fun i => ofFloat p.dtype (P.cast (some .fp32) .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (backwardWeights P p s mem key keyBlock queryBlock (i.2.1,i.1,PUnit.unit)))))
    (backwardRead p s mem p.gradient queryBlock)
def backwardDPInitial (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (queryBlock : Nat) (i : BScore p) : α :=
  P.binary (some .fp32) .real .sub (P.literal (some .fp32) .real 0)
    (P.cast (some .fp32) .fp32 .real ((mem p.delta (backwardRowOffset p s queryBlock (i.1,PUnit.unit))).read .fp32))
def backwardDPDot (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (value : Values α p.dtype.toTileDType [p.blockM,p.head]) (queryBlock : Nat) : BScore p → α :=
  P.dot (some .fp32) p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.head) (n:=p.blockM)
    (backwardRead p s mem p.gradient queryBlock) (fun i => value (i.2.1,i.1,PUnit.unit))
def backwardDP (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (value : Values α p.dtype.toTileDType [p.blockM,p.head]) (queryBlock : Nat) (i : BScore p) : α :=
  P.binary (some .fp32) .real .add (backwardDPInitial P p s mem queryBlock i)
    (P.cast (some .fp32) .fp32 .real (backwardDPDot P p s mem value queryBlock i))
def backwardDS (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) (i : BScore p) : α :=
  P.binary (some .fp32) .real .mul
    (P.binary (some .fp32) .real .mul (backwardWeights P p s mem key keyBlock queryBlock i)
      (backwardDP P p s mem value queryBlock i)) (P.literal (some .fp32) .real p.scale)
def backwardKeyDot (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) : BLane p → α :=
  P.dot (some .fp32) p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.blockM) (n:=p.head)
    (fun i => ofFloat p.dtype (P.cast (some .fp32) .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (backwardDS P p s mem key value keyBlock queryBlock (i.2.1,i.1,PUnit.unit)))))
    (backwardRead p s mem p.q queryBlock)
def backwardQueryDot (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) : BLane p → α :=
  P.dot (some .fp32) p.dtype p.dtype .fp32 {} (batch:=[]) (m:=p.blockM) (k:=p.blockM) (n:=p.head)
    (fun i => ofFloat p.dtype (P.cast (some .fp32) .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (backwardDS P p s mem key value keyBlock queryBlock i)))) key

def backwardQueryValue (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) (i : BLane p) : α :=
  P.binary (some .fp32) .real .add
    (P.cast (some .fp32) .fp32 .real ((mem p.dq (backwardOffset p s queryBlock i)).read .fp32))
    (P.cast (some .fp32) .fp32 .real (backwardQueryDot P p s mem key value keyBlock queryBlock i))
def backwardQueryCell (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat) (i : BLane p) : Cell α :=
  .mk .fp32 (P.cast none .real .fp32 (backwardQueryValue P p s mem key value keyBlock queryBlock i))
def writeMemory (mem : Memory α) (r : RegionName) (o : Nat) (v : Cell α) : Memory α :=
  fun r' o' => if r'=r ∧ o'=o then v else mem r' o'
def backwardInnerAdvance (P : Primitives α) (p : BackwardParameters) (s : State α)
    (key value : Values α p.dtype.toTileDType [p.blockM,p.head]) (keyBlock queryBlock : Nat)
    (st : BackwardInner α p) : BackwardInner α p :=
  { mem := (TileShape.allIndices [p.blockM,p.head]).foldl (fun mem i =>
      writeMemory mem p.dq (backwardOffset p s queryBlock i)
        (backwardQueryCell P p s st.mem key value keyBlock queryBlock i)) st.mem
    dk := fun i => P.binary (some .fp32) .real .add (st.dk i)
      (P.cast (some .fp32) .fp32 .real (backwardKeyDot P p s st.mem key value keyBlock queryBlock i))
    dv := fun i => P.binary (some .fp32) .real .add (st.dv i)
      (P.cast (some .fp32) .fp32 .real (backwardValueDot P p s st.mem key keyBlock queryBlock i)) }
def backwardInnerInitial (P : Primitives α) (p : BackwardParameters) (mem : Memory α) : BackwardInner α p :=
  ⟨mem,fun _ => P.literal (some .fp32) .real 0,fun _ => P.literal (some .fp32) .real 0⟩
def backwardInnerRounds (P : Primitives α) (p : BackwardParameters) (s : State α)
    (mem : Memory α) (keyBlock : Nat) : Nat → BackwardInner α p
  | 0 => backwardInnerInitial P p mem
  | n+1 => backwardInnerAdvance P p s (backwardRead p s mem p.k keyBlock)
      (backwardRead p s mem p.v keyBlock) keyBlock (keyBlock+n) (backwardInnerRounds P p s mem keyBlock n)
def backwardOuterAdvance (P : Primitives α) (p : BackwardParameters) (s : State α)
    (keyBlock : Nat) (mem : Memory α) : Memory α :=
  let st := backwardInnerRounds P p s mem keyBlock (p.blocks-keyBlock)
  let m := (TileShape.allIndices [p.blockM,p.head]).foldl (fun m i =>
    if (backwardPointer p s p.dv keyBlock).inBounds (TileShape.indexToList _ i) [0,1] then
      writeMemory m p.dv (backwardOffset p s keyBlock i) (.mk p.dtype.toTileDType
        (ofFloat p.dtype (P.cast none .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (st.dv i))))) else m) st.mem
  (TileShape.allIndices [p.blockM,p.head]).foldl (fun m i =>
    if (backwardPointer p s p.dk keyBlock).inBounds (TileShape.indexToList _ i) [0,1] then
      writeMemory m p.dk (backwardOffset p s keyBlock i) (.mk p.dtype.toTileDType
        (ofFloat p.dtype (P.cast none .fp16 p.dtype (P.cast (some .fp32) .real .fp16 (st.dk i))))) else m) m

def backwardRounds (P : Primitives α) (p : BackwardParameters) (s : State α) : Nat → Memory α
  | 0 => s.mem
  | c+1 => backwardOuterAdvance P p s c (backwardRounds P p s c)
def backwardFinalMemory (P : Primitives α) (p : BackwardParameters) (s : State α) := backwardRounds P p s p.blocks

#specNonCircular backwardFinalMemory avoiding [backwardSource,FP.Structural.exec,run,step]
#axiomsClean backwardDecomposition
#axiomsClean backwardInnerDecomposition
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Source
