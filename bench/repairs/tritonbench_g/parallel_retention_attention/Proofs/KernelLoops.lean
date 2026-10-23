import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelBinding

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (foldRange descendingCount descendingStart)
noncomputable section

theorem foldN_shift (f : Nat → α → α) (n : Nat) (z : α) :
    foldN f (n+1) z=foldN (fun j=>f (j+1)) n (f 0 z) := by
  induction n with
  | zero => rfl
  | succ n ih => simpa only [foldN] using congrArg (f (n+1)) ih

theorem source_range_count (f : Nat → α → α) (n stride base : Nat)
    (positive : 0<stride) (z : α) :
    foldRange f base (base+n*stride) stride z=
      foldN (fun j=>f (base+j*stride)) n z := by
  induction n generalizing base z with
  | zero =>
    rw [foldRange]
    simp [Nat.ne_of_gt positive,foldN]
  | succ n ih =>
    have active : base<base+(n+1)*stride := by nlinarith
    have stop : base+(n+1)*stride=base+stride+n*stride := by ring
    rw [foldRange]
    simp only [Nat.ne_of_gt positive,↓reduceDIte,active,↓reduceIte]
    rw [stop,ih,foldN_shift]
    congr 1
    · funext j value
      congr 1
      ring
    · simp

theorem foldN_project (f : Nat → α → α) (g : Nat → β → β) (project : α → β)
    (step : ∀ j x,project (f j x)=g j (project x)) (n : Nat) (z : α) :
    project (foldN f n z)=foldN g n (project z) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,step,ih]

theorem source_range_project (f : Nat → α → α) (g : Nat → β → β) (project : α → β)
    (n stride base : Nat) (positive : 0<stride)
    (step : ∀ j x,project (f (base+j*stride) x)=g j (project x)) (z : α) :
    project (foldRange f base (base+n*stride) stride z)=foldN g n (project z) := by
  rw [source_range_count f n stride base positive z]
  exact foldN_project _ _ project step n z

theorem backward_count (s : Shape) (tile : Tile) :
    descendingCount (backwardConfig s tile)=blocks s.T 32-(tile.chunk+1)*2 := by
  unfold descendingCount backwardConfig config blocks
  dsimp
  omega

theorem backward_start (s : Shape) (tile : Tile) (j : Nat) :
    descendingStart (backwardConfig s tile) j=(blocks s.T 32-1-j)*32 := by
  unfold descendingStart backwardConfig config blocks
  dsimp
  omega

#axiomsClean source_range_count
#axiomsClean source_range_project
#axiomsClean backward_count
#axiomsClean backward_start
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
