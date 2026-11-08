import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFiniteLoads

/-! Closed mathematical attention sums for the repaired source. A signed32
start conversion is part of the data indexing, including wrapping metadata.
No execution, register state, or list fold occurs in the specification. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def lowerKey (c : Call) (tile : Nat) : Nat :=
  RealKeys.lower tile c.BM c.BN c.window c.complement c.offset c.size

def upperKey (c : Call) (tile : Nat) : Int :=
  if c.window≠0 ∧ c.complement=0 then
    min (((((tile+1)*c.BM : Nat) : Int)-1+c.offset+c.BN).tdiv c.BN*c.BN) c.NK else c.NK

def count (c : Call) (tile : Nat) : Nat := RealKeys.blockCount (lowerKey c tile) (upperKey c tile).toNat c.BN

def position (c : Call) (tile n : Nat) := lowerKey c tile+n*c.BN

def physical (c : Call) (tile key : Nat) :=
  if c.window≠0 ∧ c.complement=0 then
    (BitVec.ofInt 32 ((lowerKey c tile : Nat) : Int)).toInt.toNat+(key-lowerKey c tile)
  else key

def blockScore (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (tile n : Nat) :=
  score c (queryValues c tile xs) (keyValues c xs (physical c tile (position c tile n)))

def blockActive (c : Call) (tile n : Nat) (row : Fin c.BM) (key : Fin c.BN) : Bool :=
  (if c.window≠0 then
    let distance := (row.val : Int)-key.val+tile*c.BM-(position c tile n : Nat)+c.offset
    if c.complement≠0 then decide (distance≥c.size)
    else decide (distance≥0) && decide (distance<c.size)
  else Bool.true) &&
    (if c.params.evenN=0 then decide ((key.val : Int)+(position c tile n : Nat)<c.NK) else Bool.true)

def blockValue (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (tile n : Nat) :=
  valueValues c xs (physical c tile (position c tile n))

/-- Each row is the seed rescaled to the maximum of all kept scores,
plus the two weighted sums over every executed block. The zero-block
case returns the unmodified seed, including denominator one at INIT. -/
def closed (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (tile n : Nat) :=
  RealLoopFormula.closed (seed c xs) (blockScore c xs tile) (blockActive c tile) (blockValue c xs tile) n

def finalCarry (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (tile : Nat) := closed c xs tile (count c tile)

theorem lowerKey_eq (c : Call) (a : Reference.Inputs R) : lowerKey c a.tile=(Reference.lower c a).toNat :=
  (SourceSafety.lower_nat c a).symm

theorem upperKey_eq (c : Call) (a : Reference.Inputs R) : upperKey c a.tile=Reference.upper c a := rfl

theorem physical_eq (c : Call) (a : Reference.Inputs R) (cur : Nat) :
    physical c a.tile cur=SourceSafety.physicalKey c a cur := by
  unfold physical SourceSafety.physicalKey
  rw [lowerKey_eq,Int.toNat_of_nonneg (SourceSafety.lower_nonnegative c a)]

theorem blockActive_eq (c : Call) (a : Reference.Inputs R) (n : Nat) :
    blockActive c a.tile n=active c a (position c a.tile n) := rfl

theorem position_running (c : Call) (positive : 0<c.BN) (tile n : Nat) :
    (position c tile n : Int)<upperKey c tile ↔ n<count c tile := by
  have h:=RealKeys.iteration_lt_count (lowerKey c tile) (upperKey c tile).toNat c.BN n positive
  unfold position count
  omega

theorem position_succ (c : Call) (tile n : Nat) : position c tile (n+1)=position c tile n+c.BN := by
  simp [position,Nat.add_mul,Nat.add_assoc]

theorem position_aligned (c : Call) (positive : 0<c.BN) (a : Reference.Inputs R) (n : Nat) :
    c.BN∣position c a.tile n := by
  unfold position
  rw [lowerKey_eq]
  exact dvd_add (SourceSafety.lower_aligned c a positive) (dvd_mul_left _ _)

theorem position_started (c : Call) (a : Reference.Inputs R) (n : Nat) :
    (Reference.lower c a).toNat≤position c a.tile n := by
  unfold position
  rw [lowerKey_eq]
  omega

#specNonCircular finalCarry avoiding [Reference.result,Reference.iterate,FP.Structural.exec,FP.Structural.run,Kernels.source]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
