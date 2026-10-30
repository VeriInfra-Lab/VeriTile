import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Terms

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Program
open VeriTile.Triton
open AttentionFwdTriton1.Whole001
noncomputable section
abbrev preciseDot : DotOptions := {allowTF32 := some Bool.false}
def times (name : RegName) (n : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] name) (.constNat n)
def ref (m n : Nat) (name : RegName) : Op .real [m,n] := .ref .real [m,n] name
def cast {s : TileShape} (value : Op .real s) : Op .real s := .castFloat .real .real value
def comp {s : TileShape} (value : Op .real s) : ComputeExpr .real s :=
  .compute (.alg .fp32 value)
def put {s : TileShape} (name : RegName) (value : Op .real s) : ComputeStmt :=
  .assign .real s name (comp value)
def pointer (name region : String) (parent block strides : List Nat)
    (base : Op .nat []) (offsets : List (Op .nat [])) : ComputeStmt :=
  .assign .blockPtr block name (.alg (.makeBlockPtrDynOffsets region base parent block strides offsets))
def pointers (c : Shape) : List ComputeStmt :=
  [pointer "p_q" "q" [c.T,c.D] [32,c.D] [c.D,1] (times "i_bh" c.headSize) [times "i" 32,.constNat 0],
   pointer "p_k" "k" [c.D,c.T] [c.D,32] [1,c.D] (times "i_bh" c.headSize) [.constNat 0,times "i" 32],
   pointer "p_v" "v" [c.T,c.D] [32,c.D] [c.D,1] (times "i_bh" c.headSize) [times "i" 32,.constNat 0],
   pointer "p_h" "h" [c.chunks*c.D,c.D] [c.D,c.D] [c.D,1]
     (times "i_bh" c.historyHead) [times "i" c.D,.constNat 0],
   pointer "p_o" "o" [c.T,c.D] [32,c.D] [c.D,1] (times "i_bh" c.headSize) [times "i" 32,.constNat 0]]
def store (m n : Nat) (pointer value : String) (axes : List Nat) : ComputeStmt :=
  .store .real [m,n] (.blockPtr (.ref .blockPtr [m,n] pointer) axes)
    (comp (cast (ref m n value))) .none
def save (c : Shape) (enabled : Bool) : ComputeStmt :=
  .ifThen (.constBool enabled) [store c.D c.D "p_h" "b_h" []]
def load (m n : Nat) (pointer value : String) (axes : List Nat) : ComputeStmt :=
  .assign .real [m,n] value
    (.compute (.load .fp32 (.blockPtr (.ref .blockPtr [m,n] pointer) axes) .none .zero))
def dot {m k n : Nat} (a : Op .real [m,k]) (b : Op .real [k,n]) : Op .real [m,n] :=
  .dot (batch := []) a b preciseDot
def add {m n : Nat} (a b : Op .real [m,n]) : Op .real [m,n] :=
  .add .real (.consSame (.consSame .nil)) a b
def loads (c : Shape) (scale : ℝ) : List ComputeStmt :=
  [load 32 c.D "p_q" "b_q" [0],
   put "b_q" (cast (.mul .real .scalarR (ref 32 c.D "b_q") (.const scale))),
   load c.D 32 "p_k" "b_k" [1],
   load 32 c.D "p_v" "b_v" [0],
   put "b_s" (dot (ref 32 c.D "b_q") (ref c.D 32 "b_k")),
   put "b_o" (dot (cast (ref 32 32 "b_s")) (ref 32 c.D "b_v"))]
def carryDot (c : Shape) : Op .real [c.D,c.D] :=
  dot (ref c.D 32 "b_k") (ref 32 c.D "b_v")
def later (c : Shape) : List ComputeStmt :=
  [put "b_o" (add (ref 32 c.D "b_o")
    (dot (ref 32 c.D "b_q") (cast (ref c.D c.D "b_h")))),
   put "b_h" (add (ref c.D c.D "b_h") (carryDot c))]
def update (c : Shape) (ifcond : Bool) : ComputeStmt :=
  .ifThenElse (.constBool ifcond)
    [.ifThenElse (.eq .nat .nil (.ref .nat [] "i") (.constNat 0))
      [put "b_h" (carryDot c)] (later c)] (later c)
def body (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : List ComputeStmt :=
  pointers c ++ [save c storeFlag] ++ loads c scale ++
    [update c ifcond,store 32 c.D "p_o" "b_o" [0]]
def code (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : List ComputeStmt :=
  [.assign .nat [] "i_bh" (.alg (.programId 0)),
   .assign .real [c.D,c.D] "b_h" (.compute (.full [c.D,c.D] (.alg .fp32 (.const 0)))),
   .forRangeDyn "i" (.constNat 0) (.constNat c.chunks) (.constNat 1)
     (body c scale storeFlag ifcond)]
def kernel (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) : ComputeKernel :=
  source "q" "k" "v" "h" "o" c.headSize c.D 1 c.historyHead c.D c.T scale
    32 c.D c.chunks storeFlag ifcond

/-- Exact quotation-to-statement connection, with every original branch,
load, cast, dot option, explicit add and store present. No semantic premise. -/
theorem quoted (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    kernel c scale storeFlag ifcond = .mk ["q","k","v","h","o"] []
      (code c scale storeFlag ifcond) := rfl

#axiomsClean quoted
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Program
