import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TwoLoops
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealKeys

/-! The two host-stage key intervals partition precisely the causal prefix,
including the final partial key block. This is independent of execution. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.KeyPartition
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
set_option linter.unusedSimpArgs false

def keep (N m : Nat) (diagonal : Bool) (key : Nat) : Bool :=
  if diagonal then decide (key ≤ m ∧ key < N) else decide (key < N)

def keys (lo count N m : Nat) (diagonal : Bool) : List Nat :=
  (List.range count).flatMap (fun i => (List.range' (lo+i*64) 64).filter (keep N m diagonal))

theorem concatenated (lo count N m : Nat) (diagonal : Bool) :
    keys lo count N m diagonal = (List.range' lo (count*64)).filter (keep N m diagonal) := by
  rw [keys,← List.filter_flatMap,RealKeys.concatenate_blocks]

theorem prefix_count (startM : Nat) : Loop.count 0 (startM*128) 64 = startM*2 := by
  unfold Loop.count VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFormula.tripCount
  omega

theorem prefix_keys (startM N row : Nat) (hr : startM*128+row < N) :
    keys 0 (Loop.count 0 (startM*128) 64) N (startM*128+row) Bool.false =
      List.range' 0 (startM*128) := by
  rw [concatenated,prefix_count,show startM*2*64=startM*128 by omega]
  apply List.filter_eq_self.mpr
  intro key hk
  simp only [List.mem_range'_1] at hk
  simp only [keep,Bool.false_eq_true,↓reduceIte,decide_eq_true_eq]
  omega

theorem diagonal (startM N row : Nat) (hrow : row < 128) (hr : startM*128+row < N) :
    keys (startM*128) (Loop.count (startM*128) (min ((startM+1)*128) N) 64)
      N (startM*128+row) Bool.true = List.range' (startM*128) (row+1) := by
  rw [concatenated]
  apply List.Pairwise.eq_of_mem_iff ((List.pairwise_lt_range').filter _) List.pairwise_lt_range'
  intro key
  simp only [List.mem_filter,List.mem_range'_1,keep,Bool.true_eq,↓reduceIte,decide_eq_true_eq]
  have hcover : row+1 ≤ Loop.count (startM*128) (min ((startM+1)*128) N) 64*64 := by
    unfold Loop.count VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFormula.tripCount
    omega
  omega

theorem partition (startM N row : Nat) (hrow : row < 128) (hr : startM*128+row < N) :
    keys 0 (Loop.count 0 (startM*128) 64) N (startM*128+row) Bool.false ++
      keys (startM*128) (Loop.count (startM*128) (min ((startM+1)*128) N) 64)
        N (startM*128+row) Bool.true = List.range (startM*128+row+1) := by
  rw [prefix_keys startM N row hr,diagonal startM N row hrow hr]
  simpa [Nat.add_assoc,List.range_eq_range'] using
    (List.range'_append_1 (s := 0) (m := startM*128) (n := row+1))

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.KeyPartition
