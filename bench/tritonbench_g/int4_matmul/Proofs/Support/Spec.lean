import bench.tritonbench_g.int4_matmul.Kernels
namespace VeriTile.Bench.Repairs.Int4Matmul
/-- Logical input tensor interpretation, from initial memory only. -/
def logicalA (s : Shape) (m : State α) (r k : Nat) := m.values .a (r*s.k+k)
def logicalQ (s : Shape) (m : State α) (k c : Nat) := nibble (m.packed .weight (k/8*s.n+c)) k
def logicalZ (s : Shape) (m : State α) (k c : Nat) := nibble (m.packed .zero (k/s.group*(s.n/8)+c/8)) c
def logicalScale (s : Shape) (m : State α) (k c : Nat) := m.values .scale (k/s.group*s.n+c)
def dequantized (s : Shape) (p : Ops α) (m : State α) (k c : Nat) :=
  p.mul (p.fromInt (logicalQ s m k c-logicalZ s m k c)) (logicalScale s m k c)
def sumReal (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => sumReal f n+f n
noncomputable def realOps : Ops ℝ :=
  ⟨0,(fun z => z), (·*·), (·+·),
    (fun a b seed => seed+sumReal (fun i => a ⟨i%8,Nat.mod_lt _ (by decide)⟩*b ⟨i%8,Nat.mod_lt _ (by decide)⟩) 8)⟩
/-- Independent ordinary matrix multiplication, with no tile/grid/output in its definition. -/
noncomputable def mathematical (s : Shape) (m : State ℝ) (r c : Nat) :=
  sumReal (fun k => logicalA s m r k * ((logicalQ s m k c-logicalZ s m k c : Int):ℝ) * logicalScale s m k c) s.k

/-- Finite native reference graph. The only reduction primitive has fixed eight-product
ISA arity. Padding is retained, including signed zeros and exceptional raw words. -/
def referencePartial (s : Shape) (t : Tile) (p : Ops α) (initial : State α) (w : Work) :=
  (List.range (ceil s.k (t.bk*t.splits))).foldl (fun acc block =>
    (List.range (t.bk/8)).foldl (fun seed chunk =>
      p.dot8
        (fun i => let k := (block*t.splits+w.split)*t.bk+chunk*8+i
          if k<s.k then logicalA s initial w.row k else p.zero)
        (fun i => let k := (block*t.splits+w.split)*t.bk+chunk*8+i
          if k<s.k then dequantized s p initial k w.col else p.mul (p.fromInt 0) p.zero)
        seed) acc) p.zero

def referenceOutput (s : Shape) (t : Tile) (p : Ops α) (initial : State α)
    (r c : Nat) : List Work → α → α
  | [],seed => seed
  | w::ws,seed => referenceOutput s t p initial r c ws
      (if w.row=r ∧ w.col=c then
        if t.splits=1 then referencePartial s t p initial w
        else p.atomicAdd seed (referencePartial s t p initial w)
       else seed)
end VeriTile.Bench.Repairs.Int4Matmul

