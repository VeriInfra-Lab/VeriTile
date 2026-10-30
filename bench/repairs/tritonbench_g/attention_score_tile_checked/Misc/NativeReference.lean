import bench.repairs.tritonbench_g.attention_score_tile_checked.RealIO

/-! Independent source-ordered native graph over logical Q/K/M arrays.
The fixed scalar, TF32 MMA and reduction instruction nodes retain the original
primitive model. This is an observation target, not an optimization claim. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeReference
open VeriTile Triton Reused Geometry Physical NativeInstructions NativeStaging
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

def query (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (row : Nat) (d : Fin c.D) : Graph :=
  if live : row<c.N then xs.q z h ⟨row,live⟩ d else NativeFormula.zero32
def key (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (col : Nat) (d : Fin c.D) : Graph :=
  if live : col<c.NK then xs.k z ⟨h.val/(c.H/c.HK),grouped_head c h⟩ ⟨col,live⟩ d
  else NativeFormula.zero32
def normalizer (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (row : Nat) : Graph :=
  if live : row<c.N then xs.m z h ⟨row,lt_of_lt_of_le live c.normalizerCapacity⟩
  else NativeFormula.zero32

def raw (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (scale : ℝ) (block row : Nat) (i : Fin c.B) (j : Fin c.B) : Graph :=
  .ex2ApproxFTZ (.fmaRN (.mulRN (.literal32 scale) (.literal32 1.4426950408889634))
    (dot c.D i.val j.val (query c xs z h (row+i.val))
      (key c xs z h (block*c.B+j.val)) NativeFormula.zero32)
    (.negateSign (normalizer c xs z h (row+i.val))))

def probability (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (block row : Nat) (i : Fin c.B) (j : Fin c.B) : Graph :=
  let dist : Int := (i.val : Int)-j.val+row-(block : Int)*c.B+offset
  let windowed := if window then
    if (if complement then size≤dist else 0≤dist ∧ dist<size) then raw c xs z h scale block row i j
    else NativeFormula.zero32
    else raw c xs z h scale block row i j
  if row+i.val<c.N then windowed else NativeFormula.zero32

def contribution (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (block row : Nat) (col : Fin c.B) : Graph :=
  (planFor c.B).value NativeInstructions.zero
    (fun r => if live : r<c.B then
      probability c xs z h offset size scale window complement block row ⟨r,live⟩ col
      else NativeInstructions.zero) Graph.move Graph.addRN

def update (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool)
    (block row : Nat) (acc : Fin c.B → Graph) (col : Fin c.B) : Graph :=
  .addRN (acc col) (contribution c xs z h offset size scale window complement block row col)

def column (c : Shape) (xs : LogicalInputs Graph c) (z : Fin c.Z) (h : Fin c.H)
    (offset size : Int) (scale : ℝ) (window complement : Bool) (j : Fin c.NK) : Graph :=
  foldRange (update c xs z h offset size scale window complement (j.val/c.B))
    0 c.N c.B (fun _ => NativeFormula.zero32) (FPReturned.ownerLane c j)

#specNonCircular column avoiding [kernel,nativeExecute,FP.Structural.State,FP.Structural.exec,FPGrid.run]
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.NativeReference
