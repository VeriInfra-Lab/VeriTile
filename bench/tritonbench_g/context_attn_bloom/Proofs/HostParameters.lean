import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_PrologueDefinition
import bench.tritonbench_g.context_attn_bloom.Proofs.Mathematical_Bounds
import bench.tritonbench_g.context_attn_fwd.Proofs.Mathematical_HostParameters
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host (View3 HostArithmetic)
noncomputable section

structure Arguments where
  q : View3
  k : View3
  v : View3
  o : View3
  requestRows : Nat
  requestColumns : Nat
  requestRowStride : Nat
  requestTokenStride : Nat
  batch : Nat
  maxInput : Nat
  tesla : Bool
  arithmetic : HostArithmetic

def block (a : Arguments) : Nat := if a.tesla then 64 else 128

def paddedDim (a : Arguments) : Nat := max 16 (2 ^ Nat.clog 2 a.q.dims)

def layout (a : Arguments) : Layout where
  BM := block a
  BN := block a
  D := paddedDim a
  headDim := a.q.dims
  kvGroup := a.q.heads / a.k.heads
  qRow := a.q.rowStride
  qHead := a.q.headStride
  qDim := a.q.dimStride
  kRow := a.k.rowStride
  kHead := a.k.headStride
  kDim := a.k.dimStride
  vRow := a.v.rowStride
  vHead := a.v.headStride
  vDim := a.v.dimStride
  outRow := a.o.rowStride
  outHead := a.o.headStride
  outDim := a.o.dimStride
  requestRow := a.requestRowStride
  requestToken := a.requestTokenStride

def scale (a : Arguments) : ℝ :=
  a.arithmetic.divide 1 (a.arithmetic.powerHalf a.q.dims)

def numWarps (a : Arguments) : Nat := if a.k.dims ≤ 64 then 4 else 8

def grid (a : Arguments) : Grid :=
  ⟨[a.batch, a.q.heads, Bounds.ceilBlocks a.maxInput (block a)]⟩

def launchOf (a : Arguments) (idx : GridIndex (grid a)) : Launch :=
  ⟨(idx ⟨0, by simp [grid, Grid.rank]⟩).val,
    (idx ⟨1, by simp [grid, Grid.rank]⟩).val,
    (idx ⟨2, by simp [grid, Grid.rank]⟩).val⟩

@[simp] theorem launch_grid (a : Arguments) (s : BlockState) (idx : GridIndex (grid a)) :
    launch (s.withGridIndex idx) = launchOf a idx := by
  simp [launch, launchOf, BlockState.withGridIndex, BlockState.withPids,
    BlockState.withNumPids, GridIndex.toPids, grid, Grid.rank]

@[simp] theorem inputs_grid (r : Regions) (s : BlockState) {g : Grid} (idx : GridIndex g) :
    inputs (s.withGridIndex idx) r = inputs s r := rfl

theorem block_positive (a : Arguments) : 0 < block a := by
  simp [block]; split <;> omega

theorem padded_covers (a : Arguments) : a.q.dims ≤ paddedDim a := by
  exact le_trans (Nat.le_pow_clog (by decide) _) (Nat.le_max_right _ _)

theorem launch_ext (a : Arguments) {idx jdx : GridIndex (grid a)}
    (hb : (launchOf a idx).batch = (launchOf a jdx).batch)
    (hh : (launchOf a idx).head = (launchOf a jdx).head)
    (hm : (launchOf a idx).block = (launchOf a jdx).block) : idx = jdx := by
  funext ax
  apply Fin.ext
  rcases ax with ⟨ax, ha⟩
  have ha' : ax < 3 := by simpa [grid, Grid.rank] using ha
  interval_cases ax
  · exact hb
  · exact hh
  · exact hm

/-- Default source storage is FP32, retaining all probability/output casts. -/
def source (a : Arguments) (r : Regions) (swapped : Bool) : ComputeKernel :=
  Program.kernel (layout a) r .fp32 swapped (scale a)

/-- A logical lazy-negative V view preserves the complete raw source ABI. -/
def negateValueView (a : Arguments) : Arguments :=
  { a with v := { a.v with negative := !a.v.negative } }

theorem source_ignores_value_sign (a : Arguments) (r : Regions) (swapped : Bool) :
    source (negateValueView a) r swapped = source a r swapped := rfl

theorem grid_ignores_value_sign (a : Arguments) : grid (negateValueView a) = grid a := rfl

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Host
