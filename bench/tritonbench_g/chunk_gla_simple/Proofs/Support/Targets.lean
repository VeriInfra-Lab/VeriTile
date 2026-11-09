import bench.tritonbench_g.chunk_gla_simple.IO

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicMath
open VeriTile Triton Host PublicIO
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math
noncomputable section

/-- Gated history contraction plus the causal query/key/value contraction,
scaled once. All arrays are logical inputs; no execution state occurs here. -/
def formula (c : Shape) (bt : 0<c.BT) (scale : ℝ) (xs : RealInputs c)
    (_ : Fin 1) (i : Fin (c.heads*c.T*c.V)) : ℝ :=
  Host.formula c bt (arrays c scale xs) scale i.divNat.divNat i.divNat.modNat i.modNat

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicMath

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicNative
open VeriTile Triton FP.Structural Host PublicIO
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open NativeGraph (Expr)
noncomputable section

structure Arrays where
  query : Nat → Expr
  key : Nat → Expr
  value : Nat → Expr
  history : Nat → Expr
  gate : Nat → Expr

def zero : Expr := .literal (some .fp32) .real 0

def arrays (c : Shape) (xs : ChannelValues Expr (nativeInputs c)) : Arrays :=
  ⟨PublicMath.extend zero (fun j => xs (0 : Fin 5) (j,())),
   PublicMath.extend zero (fun j => xs (1 : Fin 5) (j,())),
   PublicMath.extend zero (fun j => xs (2 : Fin 5) (j,())),
   PublicMath.extend zero (fun j => xs (3 : Fin 5) (j,())),
   PublicMath.extend zero (fun j => xs (4 : Fin 5) (j,()))⟩

def row (c : Shape) (chunk : Nat) (i : Fin c.BT) := chunk*c.BT+i.val
def column (c : Shape) (value : Nat) (i : Fin c.BV) := value*c.BV+i.val
def key (c : Shape) (block : Nat) (i : Fin c.BK) := block*c.BK+i.val

def query (c : Shape) (a : Arrays) (head chunk block : Nat) (i : FPValue.QIndex c) : Expr :=
  .load (if row c chunk i.1<c.T ∧ key c block i.2.1<c.K then
    a.query (head*(c.T*c.K)+row c chunk i.1*c.K+key c block i.2.1) else zero)

def keyValue (c : Shape) (a : Arrays) (head chunk block : Nat) (i : FPValue.KIndex c) : Expr :=
  .load (if key c block i.1<c.K ∧ row c chunk i.2.1<c.T then
    a.key (head*(c.T*c.K)+key c block i.1+row c chunk i.2.1*c.K) else zero)

def history (c : Shape) (a : Arrays) (head chunk value block : Nat) (i : FPValue.HIndex c) : Expr :=
  .load (if key c block i.1<c.K ∧ column c value i.2.1<c.V then
    a.history (head*(c.NT*c.K*c.V)+chunk*c.K*c.V+key c block i.1*c.V+column c value i.2.1)
    else zero)

def value (c : Shape) (a : Arrays) (head chunk value : Nat) (i : FPValue.OIndex c) : Expr :=
  .load (if row c chunk i.1<c.T ∧ column c value i.2.1<c.V then
    a.value (head*(c.T*c.V)+row c chunk i.1*c.V+column c value i.2.1) else zero)

def gate (c : Shape) (a : Arrays) (head chunk : Nat) (i : Fin c.BT) : Expr :=
  .load (if row c chunk i<c.T then a.gate (head*c.T+row c chunk i) else zero)

def update (c : Shape) (a : Arrays) (head chunk value block : Nat)
    (acc : FPValue.Acc Expr c) : FPValue.Acc Expr c :=
  (fun i => .dotAcc (some .fp32) NativeGraph.options [] c.BT c.BK c.BV
    (query c a head chunk block) (history c a head chunk value block) (acc.1 i) i,
   fun i => .dotAcc (some .fp32) NativeGraph.options [] c.BT c.BK c.BT
    (query c a head chunk block) (keyValue c a head chunk block) (acc.2 i) i)

/-- Exact ordered source expression: padded FP32 loads, both carried dot
accumulators, gate exponentials, causal mask, third dot, scale and final cast. -/
def expression (c : Shape) (a : Arrays) (head chunk vblock : Nat)
    (i : FPValue.OIndex c) : Expr :=
  let acc := foldRange (update c a head chunk vblock) 0 ((c.K+c.BK-1)/c.BK) 1
    (fun _ => zero, fun _ => zero)
  let history := Expr.binary (some .fp32) .real .mul (acc.1 i)
    (.unary (some .fp32) .exp (gate c a head chunk i.1))
  let score := fun j : FPValue.SIndex c =>
    if j.2.1.val ≤ j.1.val then Expr.binary (some .fp32) .real .mul (acc.2 j)
      (.unary (some .fp32) .exp (.binary (some .fp32) .real .sub
        (gate c a head chunk j.1) (gate c a head chunk j.2.1)))
    else zero
  .cast (some .fp32) .real .fp32
    (.binary (some .fp32) .real .mul
      (.dotAcc (some .fp32) NativeGraph.options [] c.BT c.BT c.BV
        (fun j => .cast (some .fp32) .real .real (score j))
        (value c a head chunk vblock) history i)
      (.literal (some .fp32) .real ScalarGraph.scaleTag))

def expressionFormula (c : Shape) (bt : 0<c.BT) (xs : ChannelValues Expr (nativeInputs c)) :
    ChannelValues Expr [⟨.fp32,[c.heads,c.T,c.V],"o"⟩] :=
  Fin.cases (fun j => expression c (arrays c xs) j.1.val (j.2.1.val/c.BT)
    (j.2.2.1.val/c.BV) (lane c bt j.2.1 j.2.2.1)) (fun o => o.elim0)

/-- The original native observation is retained: all accumulator dots lower
to ordered scalar FMA chains, exp lowers to exp2/mul, and scale is the runtime
FP32 word. This is not a native-equals-Real assertion. -/
def formula (c : Shape) (bt : 0<c.BT) (scale : BitVec 32)
    (xs : ChannelValues Expr (nativeInputs c)) :
    ChannelValues VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.Graph
      [⟨.fp32,[c.heads,c.T,c.V],"o"⟩] :=
  Fin.cases (fun j => ScalarGraph.lower scale
    (expression c (arrays c xs) j.1.val (j.2.1.val/c.BT) (j.2.2.1.val/c.BV)
      (lane c bt j.2.1 j.2.2.1))) (fun o => o.elim0)

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PublicNative
