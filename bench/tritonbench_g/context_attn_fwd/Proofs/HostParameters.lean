import bench.tritonbench_g.context_attn_fwd.Proofs.Program
import bench.tritonbench_g.context_attn_fwd.Proofs.Bounds
import VeriTile.Triton.Launch.Grid

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1000000

/-- A caller tensor view. Strides are the actual nonnegative element strides;
its region denotes the passed data pointer, including its storage offset. -/
structure View3 where
  rows : Nat
  heads : Nat
  dims : Nat
  rowStride : Nat
  headStride : Nat
  dimStride : Nat
  /-- Torch lazy-negative flag; the original raw-pointer kernel ignores it. -/
  negative : Bool := Bool.false

structure View4 where
  batches : Nat
  heads : Nat
  rows : Nat
  dims : Nat
  batchStride : Nat
  headStride : Nat
  rowStride : Nat
  dimStride : Nat
  /-- Torch lazy-negative flag; the original raw-pointer kernel ignores it. -/
  negative : Bool := Bool.false

/-- Host scalar operations are a numerical boundary. They represent Python's
binary64 power/division/multiplication, before Triton's scalar conversion. -/
structure HostArithmetic where
  powerHalf : ℝ → ℝ
  divide : ℝ → ℝ → ℝ
  multiply : ℝ → ℝ → ℝ

def View3.address (v : View3) (row : Int) (head dim : Nat) : Int :=
  row*v.rowStride + head*v.headStride + dim*v.dimStride

def View3.Injective (v : View3) : Prop :=
  ∀ x y : Int, ∀ h k d e : Nat,
    0 ≤ x → x < v.rows → 0 ≤ y → y < v.rows →
    h < v.heads → k < v.heads → d < v.dims → e < v.dims →
    (v.address x h d).toNat = (v.address y k e).toNat → x = y ∧ h = k ∧ d = e

structure Arguments where
  q : View3
  k : View4
  v : View4
  o : View3
  batch : Nat
  maxInput : Nat
  tesla : Bool
  arithmetic : HostArithmetic

def block (a : Arguments) : Nat := if a.tesla then 64 else 128

def layout (a : Arguments) : Layout where
  BM := block a
  BN := block a
  D := a.k.dims
  heads := a.q.heads
  kvGroup := a.q.heads / a.k.heads
  qRow := a.q.rowStride
  qHead := a.q.headStride
  qDim := a.q.dimStride
  kBatch := a.k.batchStride
  kHead := a.k.headStride
  kRow := a.k.rowStride
  kDim := a.k.dimStride
  vBatch := a.v.batchStride
  vHead := a.v.headStride
  vRow := a.v.rowStride
  vDim := a.v.dimStride
  outRow := a.o.rowStride
  outHead := a.o.headStride
  outDim := a.o.dimStride

def scale (a : Arguments) : ℝ :=
  a.arithmetic.multiply
    (a.arithmetic.divide 1 (a.arithmetic.powerHalf a.q.dims)) 1.4426950408889634

def numWarps (a : Arguments) : Nat := if a.k.dims ≤ 64 then 4 else 8

def grid (a : Arguments) : Grid :=
  ⟨[Bounds.ceilBlocks a.maxInput (block a), a.batch*a.q.heads, 1]⟩

def launchOf (a : Arguments) (idx : GridIndex (grid a)) : Launch :=
  ⟨(idx ⟨0, by simp [grid, Grid.rank]⟩).val,
    (idx ⟨1, by simp [grid, Grid.rank]⟩).val⟩

@[simp] theorem launch_grid (a : Arguments) (s : BlockState) (idx : GridIndex (grid a)) :
    launch (s.withGridIndex idx) = launchOf a idx := by
  simp [launch, launchOf, BlockState.withGridIndex, BlockState.withPids,
    BlockState.withNumPids, GridIndex.toPids, grid, Grid.rank]

@[simp] theorem inputs_grid (r : Regions) (s : BlockState) {g : Grid} (idx : GridIndex g) :
    inputs (s.withGridIndex idx) r = inputs s r := rfl

theorem block_positive (a : Arguments) : 0 < block a := by
  simp [block]; split <;> omega

theorem launch_ext (a : Arguments) {idx jdx : GridIndex (grid a)}
    (hb : (launchOf a idx).block = (launchOf a jdx).block)
    (hh : (launchOf a idx).batchHead = (launchOf a jdx).batchHead) : idx = jdx := by
  funext ax
  apply Fin.ext
  rcases ax with ⟨ax, ha⟩
  have ha' : ax < 3 := by simpa [grid, Grid.rank] using ha
  interval_cases ax
  · exact hb
  · exact hh
  · have hi := (idx ⟨2, by simp [grid, Grid.rank]⟩).isLt
    have hj := (jdx ⟨2, by simp [grid, Grid.rank]⟩).isLt
    change (idx ⟨2, ha⟩).val < 1 at hi
    change (jdx ⟨2, ha⟩).val < 1 at hj
    omega

/-- Selected production storage is FP32. -/
def source (a : Arguments) (r : Regions) (swapped : Bool) : ComputeKernel :=
  kernel (layout a) r .fp32 swapped (scale a)

/-- A logical lazy-negative V view preserves the complete raw source ABI. -/
def negateValueView (a : Arguments) : Arguments :=
  { a with v := { a.v with negative := !a.v.negative } }

theorem source_ignores_value_sign (a : Arguments) (r : Regions) (swapped : Bool) :
    source (negateValueView a) r swapped = source a r swapped := rfl

theorem grid_ignores_value_sign (a : Arguments) : grid (negateValueView a) = grid a := rfl

end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host
