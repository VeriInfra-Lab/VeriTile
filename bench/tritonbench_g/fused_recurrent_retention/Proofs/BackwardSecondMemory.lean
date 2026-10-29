import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSecondState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Memory
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Addressing

/-! Exact tagged-cell effect of the two reverse-loop stores. Input preservation
and output uniqueness are separate from the numerical recurrence. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondMemory
open VeriTile.Triton Primitives MaskedMath BackwardSecondState
noncomputable section
set_option maxHeartbeats 900000
set_option linter.unusedSimpArgs false

def stores (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) : BlockState :=
  scatterV p l initial r t (scatterK p l initial r t u)

theorem full_frame (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) (region : RegionName) (off : Nat)
    (hk : ∀ k : Fin p.BK, activeKey p l k.val →
      (r.dk,keyGradientAddress p l t k.val) ≠ (region,off))
    (hv : ∀ j : Fin p.BV, activeValue p l j.val →
      (r.dv,outputAddress p l t j.val) ≠ (region,off)) :
    (stores p l initial r t u).mem region off = u.mem region off := by
  change (Memory.scatter p.BV r.dv (fun j => outputAddress p l t j.val)
    (MaskedMath.dv p l initial r t) (fun j => activeValue p l j.val)
    (Memory.scatter p.BK r.dk (fun k => keyGradientAddress p l t k.val)
      (MaskedMath.dk p l initial r t) (fun k => activeKey p l k.val) u)).mem region off = _
  rw [Memory.full_frame _ _ _ _ _ _ region off hv,
    Memory.full_frame _ _ _ _ _ _ region off hk]

theorem read_other (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) (region : RegionName) (off : Nat)
    (hk : region ≠ r.dk) (hv : region ≠ r.dv) :
    (stores p l initial r t u).readMem region off = u.readMem region off := by
  unfold stores scatterV scatterK
  rw [BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hv,
    BlockState.scatter_prop_masked_preserves_other_region _ _ _ _ _ hk]

theorem key_written (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) (hkv : r.dk ≠ r.dv) (k : Fin p.BK)
    (hk : activeKey p l k.val) :
    (stores p l initial r t u).mem r.dk (keyGradientAddress p l t k.val) =
      MemCell.real (MaskedMath.dk p l initial r t k) := by
  change (Memory.scatter p.BV r.dv (fun j => outputAddress p l t j.val)
    (MaskedMath.dv p l initial r t) (fun j => activeValue p l j.val)
    (scatterK p l initial r t u)).mem r.dk (keyGradientAddress p l t k.val) = _
  rw [Memory.full_frame _ _ _ _ _ _ r.dk (keyGradientAddress p l t k.val)
    (by intro j hj he; exact hkv (congrArg Prod.fst he).symm)]
  exact Memory.written_cell p.BK r.dk (fun i => keyGradientAddress p l t i.val)
    (MaskedMath.dk p l initial r t) (fun i => activeKey p l i.val) u k hk
    (fun i hi he => (Addressing.key_gradient_unique p l t t i k hi hk he).2)

theorem value_written (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (u : BlockState) (j : Fin p.BV) (hj : activeValue p l j.val) :
    (stores p l initial r t u).mem r.dv (outputAddress p l t j.val) =
      MemCell.real (MaskedMath.dv p l initial r t j) := by
  exact Memory.written_cell p.BV r.dv (fun i => outputAddress p l t i.val)
    (MaskedMath.dv p l initial r t) (fun i => activeValue p l i.val)
    (scatterK p l initial r t u) j hj
    (fun i hi he => (Addressing.output_unique p l t t i j hi hj he).2)

theorem key_future (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t s : Nat) (u : BlockState) (hkv : r.dk ≠ r.dv) (ht : t < s)
    (k : Fin p.BK) (hk : activeKey p l k.val) :
    (stores p l initial r t u).mem r.dk (keyGradientAddress p l s k.val) =
      u.mem r.dk (keyGradientAddress p l s k.val) := by
  apply full_frame
  · intro i hi he
    have hts := (Addressing.key_gradient_unique p l t s i k hi hk (congrArg Prod.snd he)).1
    omega
  · intro j hj he
    exact hkv (congrArg Prod.fst he).symm

theorem value_future (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t s : Nat) (u : BlockState) (hkv : r.dk ≠ r.dv) (ht : t < s)
    (j : Fin p.BV) (hj : activeValue p l j.val) :
    (stores p l initial r t u).mem r.dv (outputAddress p l s j.val) =
      u.mem r.dv (outputAddress p l s j.val) := by
  apply full_frame
  · intro k hk he
    exact hkv (congrArg Prod.fst he)
  · intro i hi he
    have hts := (Addressing.output_unique p l t s i j hi hj (congrArg Prod.snd he)).1
    omega

#axiomsClean full_frame
#axiomsClean read_other
#axiomsClean key_written
#axiomsClean value_written
#axiomsClean key_future
#axiomsClean value_future
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardSecondMemory
