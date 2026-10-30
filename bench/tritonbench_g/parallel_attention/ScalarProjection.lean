import bench.tritonbench_g.parallel_attention.ScalarModel
import bench.tritonbench_g.parallel_attention.FPRange

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

theorem scalar_dot_projection (a : Scalars α) (scale : ℝ) (site : List Nat)
    {m k n : Nat} (x : TileIndex [m,k] → α) (y : TileIndex [k,n] → α)
    (r : Fin m) (v : Fin n) (u w : Nat → α)
    (hx : ∀ j : Fin k, x ⟨r,j,()⟩ = u j.val)
    (hy : ∀ j : Fin k, y ⟨j,v,()⟩ = w j.val) :
    scalarDot a [] x y ⟨r,v,()⟩ = dot (a.ops scale) sourcePlans site k u w := by
  unfold scalarDot dot sourcePlans
  rw [dotPlan_scale]
  apply linear_dot_congr
  · intro j hj
    simp only [dif_pos hj]
    exact hx ⟨j,hj⟩
  · intro j hj
    simp only [dif_pos hj]
    exact hy ⟨j,hj⟩

theorem scalar_sum_projection (a : Scalars α) (scale : ℝ) (site : List Nat)
    {m n : Nat} (x : TileIndex [m,n] → α) (r : Fin m) (u : Nat → α)
    (hx : ∀ j : Fin n, x ⟨r,j,()⟩=u j.val) :
    scalarSum a ⟨1,by simp⟩ Bool.false x ⟨r,()⟩ = total (a.ops scale) sourcePlans site n u := by
  change (serialSum n).eval (a.ops 0)
    (fun j => if h : j<n then x ⟨r,⟨j,h⟩,()⟩ else a.literal 0) =
      (serialSum n).eval (a.ops scale) u
  rw [sumPlan_scale]
  apply serial_sum_congr
  intro j hj
  simpa [hj] using hx ⟨j,hj⟩

/-- Projecting the pure array recurrence is independent of execution. -/
theorem foldRange_projection {β γ : Type} (f : Nat → β → β) (g : Nat → γ → γ)
    (observe : β → γ) (cur stop stride : Nat) (initial : β)
    (hstep : ∀ n a, observe (f n a)=g n (observe a)) :
    observe (foldRange f cur stop stride initial) = foldRange g cur stop stride (observe initial) := by
  by_cases hz : stride=0
  · simp [foldRange,hz]
  · by_cases hc : cur<stop
    · conv_lhs => rw [foldRange]
      conv_rhs => rw [foldRange]
      simp only [if_neg hz,if_pos hc,dif_neg hz,dif_pos hc]
      rw [foldRange_projection f g observe (cur+stride) stop stride (f cur initial) hstep,hstep]
    · simp [foldRange,hz,hc]
termination_by stop-cur
decreasing_by omega

theorem foldN_shift {β : Type} (g : Nat → β → β) (n : Nat) (z : β) :
    foldN (fun j => g (j+1)) n (g 0 z)=foldN g (n+1) z := by
  induction n with
  | zero => rfl
  | succ n ih => simpa only [foldN] using congrArg (g (n+1)) ih

/-- Dynamic source ranges and the scalar graph's finite loop share exactly
the same increasing indices. No numerical fold law is used. -/
theorem foldRange_count {β : Type} (f : Nat → β → β) (start stride n : Nat)
    (positive : 0<stride) (initial : β) :
    foldRange f start (start+n*stride) stride initial =
      foldN (fun j => f (start+j*stride)) n initial := by
  induction n generalizing start initial with
  | zero => simp [foldRange,foldN]
  | succ n ih =>
    rw [foldRange]
    have hs : stride≠0 := by omega
    have hlt : start<start+(n+1)*stride := by nlinarith
    simp only [if_neg hs,if_pos hlt,dif_neg hs,dif_pos hlt]
    have hend : start+(n+1)*stride=(start+stride)+n*stride := by ring
    rw [hend,ih]
    simpa [Nat.add_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
      foldN_shift (fun j => f (start+j*stride)) n initial

def inputReaders {α : Type} [Inhabited α] (shape : Shape) (s : State α) : TensorInputs α where
  q := fun bh t e => (s.mem "q" (bh*(shape.T*shape.K)+t*shape.K+e)).read .real
  k := fun bh t e => (s.mem "k" (bh*(shape.T*shape.K)+t*shape.K+e)).read .real
  v := fun bh t e => (s.mem "v" (bh*(shape.T*shape.V)+t*shape.V+e)).read .real
  do_ := fun bh t e => (s.mem "do" (bh*(shape.T*shape.V)+t*shape.V+e)).read .real
  dz := fun bh t => (s.mem "dz" (bh*shape.T+t)).read .real

def launchTile {α : Type} (shape : Shape) (s : State α) : Tile where
  bh := s.pids 2
  chunk := s.pids 1
  key := s.pids 0 / ((shape.V+shape.BV-1)/shape.BV)
  value := s.pids 0 % ((shape.V+shape.BV-1)/shape.BV)

def forwardConfig (shape : Shape) (scale : ℝ) : ForwardConfig where
  q := "q"; k := "k"; v := "v"; o := "o"; z := "z"
  s_qk_h := shape.T*shape.K; s_qk_t := shape.K; s_qk_d := 1
  s_vo_h := shape.T*shape.V; s_vo_t := shape.V; s_vo_d := 1
  scale := scale
  B := shape.B; H := shape.H; T := shape.T; K := shape.K; V := shape.V
  BTL := 128; BTS := 32; BK := shape.BK; BV := shape.BV

def backwardConfig (shape : Shape) (scale : ℝ) (tile : Tile) : BackwardConfig where
  toForwardConfig := { forwardConfig shape scale with BTL := 64 }
  do_ := "do"; dz := "dz"; dk := "dk"; dv := "dv"
  i_bh := tile.bh; i_c := tile.chunk; i_k := tile.key; i_v := tile.value; i_h := tile.bh%shape.H

def dqConfig (shape : Shape) (scale : ℝ) (tile : Tile) : QueryConfig where
  toBackwardConfig := backwardConfig shape scale tile
  dq := "dq"

#axiomsClean scalar_dot_projection
#axiomsClean scalar_sum_projection
#axiomsClean foldRange_projection
#axiomsClean foldRange_count
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
