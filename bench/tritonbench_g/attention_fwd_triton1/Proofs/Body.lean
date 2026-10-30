import bench.tritonbench_g.attention_fwd_triton1.Proofs.Program

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TermBody
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
abbrev Matrix (α : Type) (m n : Nat) := TileIndex [m,n] → Term α
abbrev num (x : ℝ) : Term α := .literal (some .fp32) .real x
abbrev cast (t : Term α) : Term α := .cast (some .fp32) .real .real t
abbrev mul (a b : Term α) : Term α := .binary (some .fp32) .real .mul a b
abbrev add (a b : Term α) : Term α := .binary (some .fp32) .real .add a b
def dot {m k n : Nat} (a : Matrix α m k) (b : Matrix α k n) : Matrix α m n :=
  fun i => .dot (some .fp32) Program.preciseDot k
    (fun d => a (i.1,d,PUnit.unit)) (fun d => b (d,i.2.1,PUnit.unit))
def mainAddress (c : Shape) (head chunk row col : Nat) : Nat :=
  head*c.headSize+(chunk*32+row)*c.D+col
def historyAddress (c : Shape) (head chunk row col : Nat) : Nat :=
  head*c.historyHead+(chunk*c.D+row)*c.D+col
def query (c : Shape) (q : Nat → Term α) (scale : ℝ) (head chunk : Nat) : Matrix α 32 c.D :=
  fun i => cast (mul (.load (if chunk*32+i.1.val<c.T then
    q (mainAddress c head chunk i.1.val i.2.1.val) else num 0)) (num scale))
def key (c : Shape) (k : Nat → Term α) (head chunk : Nat) : Matrix α c.D 32 :=
  fun i => .load (if chunk*32+i.2.1.val<c.T then
    k (mainAddress c head chunk i.2.1.val i.1.val) else num 0)
def value (c : Shape) (v : Nat → Term α) (head chunk : Nat) : Matrix α 32 c.D :=
  fun i => .load (if chunk*32+i.1.val<c.T then
    v (mainAddress c head chunk i.1.val i.2.1.val) else num 0)
def score (c : Shape) (q k : Nat → Term α) (scale : ℝ) (head chunk : Nat) :=
  dot (query c q scale head chunk) (key c k head chunk)
def within (c : Shape) (q k v : Nat → Term α) (scale : ℝ) (head chunk : Nat) :=
  dot (fun i => cast (score c q k scale head chunk i)) (value c v head chunk)
def output (c : Shape) (q k v : Nat → Term α) (scale : ℝ) (head chunk : Nat)
    (carry : Matrix α c.D c.D) (ifcond : Bool) : Matrix α 32 c.D :=
  let w := within c q k v scale head chunk
  if ifcond && decide (chunk=0) then w
  else fun i => add (w i) (dot (query c q scale head chunk) (fun j => cast (carry j)) i)
def nextCarry (c : Shape) (k v : Nat → Term α) (head chunk : Nat)
    (carry : Matrix α c.D c.D) (ifcond : Bool) : Matrix α c.D c.D :=
  let d := dot (key c k head chunk) (value c v head chunk)
  if ifcond && decide (chunk=0) then d else fun i => add (carry i) (d i)

end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.TermBody
