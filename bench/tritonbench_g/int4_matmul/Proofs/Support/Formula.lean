import bench.tritonbench_g.int4_matmul.IO

namespace VeriTile.Bench.Repairs.Int4Matmul.Formula
open VeriTile.Triton FP.Structural
noncomputable section

def a (s : Shape) (o : Ops α) (v : ChannelValues α (IO.inputs s)) (r k : Nat) : α :=
  if h : r*s.k+k<s.m*s.k then v 0 (⟨r*s.k+k,h⟩,()) else o.zero
def q (s : Shape) (v : ChannelValues α (IO.inputs s)) (k c : Nat) : Int :=
  nibble (BitVec.ofInt 32 (if h : k/8*s.n+c<(s.k/8)*s.n then v 1 (⟨k/8*s.n+c,h⟩,()) else 0)) k
def z (s : Shape) (v : ChannelValues α (IO.inputs s)) (k c : Nat) : Int :=
  nibble (BitVec.ofInt 32 (if h : k/s.group*(s.n/8)+c/8<(s.k/s.group)*(s.n/8) then
    v 3 (⟨k/s.group*(s.n/8)+c/8,h⟩,()) else 0)) c
def scale (s : Shape) (o : Ops α) (v : ChannelValues α (IO.inputs s)) (k c : Nat) : α :=
  if h : k/s.group*s.n+c<(s.k/s.group)*s.n then v 2 (⟨k/s.group*s.n+c,h⟩,()) else o.zero

def dequantize (s : Shape) (o : Ops α) (v : ChannelValues α (IO.inputs s)) (k c : Nat) : α :=
  o.mul (o.fromInt (q s v k c-z s v k c)) (scale s o v k c)

/-- Ordinary dequantized matrix product over the independent finite operands. -/
def mathematical (s : Shape) (v : ChannelValues ℝ (IO.inputs s)) (r c : Nat) : ℝ :=
  sumReal (fun k => a s realOps v r k * ((q s v k c-z s v k c : Int):ℝ) * scale s realOps v k c) s.k

/-- The existing fixed-K8 native instruction graph with its live carry and all
padding. This is independent of source execution and access/safety judgments. -/
def partialSum (s : Shape) (t : Tile) (o : Ops α) (v : ChannelValues α (IO.inputs s))
    (row col split : Nat) : α :=
  (List.range (ceil s.k (t.bk*t.splits))).foldl (fun acc block =>
    (List.range (t.bk/8)).foldl (fun seed chunk =>
      o.dot8
        (fun i => let k := (block*t.splits+split)*t.bk+chunk*8+i
          if k<s.k then a s o v row k else o.zero)
        (fun i => let k := (block*t.splits+split)*t.bk+chunk*8+i
          if k<s.k then dequantize s o v k col else o.mul (o.fromInt 0) o.zero)
        seed) acc) o.zero

def observed (s : Shape) (t : Tile) (o : Ops α) (v : ChannelValues α (IO.inputs s))
    (r c : Nat) (schedule : List IO.Program) : α :=
  schedule.foldl (fun seed p => if t.splits=1 then partialSum s t o v r c p.2
    else o.atomicAdd seed (partialSum s t o v r c p.2)) o.zero

def realOutput (s : Shape) (v : ChannelValues ℝ (IO.inputs s)) : ChannelValues ℝ (IO.outputs s) :=
  Fin.cases (fun j => mathematical s v j.1.val j.2.1.val) (fun i => Fin.elim0 i)
def nativeOutput (s : Shape) (t : Tile) (o : Ops α) (lanes : List Lane)
    (v : ChannelValues α (IO.inputs s)) : ChannelValues α (IO.outputs s) :=
  Fin.cases (fun j => observed s t o v j.1.val j.2.1.val
    (IO.order s t lanes j.1.val j.2.1.val)) (fun i => Fin.elim0 i)
end
end VeriTile.Bench.Repairs.Int4Matmul.Formula
