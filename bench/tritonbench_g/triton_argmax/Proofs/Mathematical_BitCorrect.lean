import bench.tritonbench_g.triton_argmax.Proofs.Mathematical_BitBackend
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonArgmax.BitCorrect
open VeriTile.Triton FP.Structural TypedSource MatrixReduction.MachineMax
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

def channel : Storage → FloatDType | .fp16 => .fp16 | .fp32 => .real

def stage2Input (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (i : Fin B) : Bits storage :=
  if i.val < N then toFloat (channel storage)
    ((s.mem X i.val).read (channel storage).toTileDType)
  else negativeInfinity storage

def stage2Selected (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (hB : 0 < B) : Fin B :=
  best storage (stage2Input storage s X N B) hB

def stage1Input (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (i : Fin B) : Bits storage :=
  if s.pids 0 * B + i.val < N then toFloat (channel storage)
    ((s.mem X (s.pids 0 * B + i.val)).read (channel storage).toTileDType)
  else negativeInfinity storage

def stage1Selected (storage : Storage) (s : State (Bits storage)) (X : RegionName)
    (N B : Nat) (hB : 0 < B) : Fin B :=
  best storage (stage1Input storage s X N B) hB

def stage1Memory (storage : Storage) (s : State (Bits storage)) (X Y : RegionName)
    (I : Region .int) (N B : Nat) (hB : 0 < B) : RegionName → Nat → Cell (Bits storage) :=
  let j := stage1Selected storage s X N B hB
  ((s.write Y (s.pids 0) (.mk (channel storage).toTileDType
    (ofFloat (channel storage) (stage1Input storage s X N B j)))).write
    (Region.cast I) (s.pids 0) (.mk .int (Int.ofNat (j.val + s.pids 0 * B)))).mem

/-- Independent machine argmax: largest comparison key, with the smallest lane
index on numerical ties, including a tie between positive and negative zero. -/
def IsArgMaximum {B : Nat} (storage : Storage) (xs : Fin B → Bits storage) (j : Fin B) : Prop :=
  (∀ i, rank storage (xs i) ≤ rank storage (xs j)) ∧
  (∀ i, rank storage (xs i) = rank storage (xs j) → j.val ≤ i.val)

end
end VeriTile.Bench.TritonBenchG.TritonArgmax.BitCorrect
