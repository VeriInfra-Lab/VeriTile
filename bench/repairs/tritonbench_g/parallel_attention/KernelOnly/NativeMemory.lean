import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.NativeStage
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ForwardObserved

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

/-- Change the floating payload channel while preserving every raw cell tag
and every integer, pointer and block-pointer payload. -/
def mapValue (f : α → β) : (d : TileDType) → Value α d → Value β d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => f
  | .nat | .int | .bool | .ptr | .blockPtr => id

def mapCell (f : α → β) : Cell α → Cell β
  | .mk d value => .mk d (mapValue f d value)

def entry (s : State Word) : State Expr where
  mem := fun R off => mapCell (Expr.input R off) (s.mem R off)
  regs := fun d sh name => (s.regs d sh name).map (fun value i => mapValue (Expr.input "register" 0) d (value i))
  pids := s.pids
  numPids := s.numPids
  undef := fun d R off => mapValue (Expr.input R off) d (s.undef d R off)

def finish (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor) (tile : Tile)
    (s : State Expr) : State Word where
  mem := fun R off => mapCell (evaluate a phase d tile) (s.mem R off)
  regs := fun dt sh name => (s.regs dt sh name).map (fun value i => mapValue (evaluate a phase d tile) dt (value i))
  pids := s.pids
  numPids := s.numPids
  undef := fun dt R off => mapValue (evaluate a phase d tile) dt (s.undef dt R off)

theorem entry_finish_cell (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor)
    (tile : Tile) (s : State Word) (R : RegionName) (off : Nat) :
    (finish a phase d tile (entry s)).mem R off=s.mem R off := by
  change mapCell _ (mapCell _ (s.mem R off))=s.mem R off
  cases h : s.mem R off with
  | mk dt value => cases dt <;> rfl

theorem finish_frame (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor)
    (tile : Tile) (s : State Word) (after : State Expr) (R : RegionName) (off : Nat)
    (frame : after.mem R off=(entry s).mem R off) :
    (finish a phase d tile after).mem R off=s.mem R off := by
  change mapCell _ (after.mem R off)=_
  rw [frame]
  exact entry_finish_cell a phase d tile s R off

theorem query_stores_algebra {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) (values : TileIndex [64,shape.BK] → α) :
    queryOutputMemory M (dqConfig shape scale tile) s values =
      rectangleStore (queryRectangle shape tile) s (fun i => M.cast (some .fp32) .real .real (values i)) := by
  simp only [queryOutputMemory,rectangleStore,queryRectangle,dqConfig,backwardConfig,forwardConfig,
    Rectangle.active,Rectangle.address,Nat.mul_one]

theorem backward_stores_algebra {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (scale : ℝ)
    (tile : Tile) (s : State α) (values : BackwardAccumulator α 64 shape.BK shape.BV) :
    backwardOutputMemory M (backwardConfig shape scale tile) s values =
      rectangleStore (valueRectangle shape tile)
        (rectangleStore (keyRectangle shape tile) s (fun i => M.cast (some .fp32) .real .real (values.key i)))
        (fun i => M.cast (some .fp32) .real .real (values.value i)) := by
  simp only [backwardOutputMemory,rectangleStore,keyRectangle,valueRectangle,queryRectangle,
    outputRectangle,backwardConfig,forwardConfig,Rectangle.active,Rectangle.address,Nat.mul_one]

theorem forward_stores_algebra {α : Type} [Inhabited α] (M : Algebra α) (shape : Shape) (scale : ℝ)
    (s : State α) (values : ForwardAccumulator α 128 shape.BV) :
    outputMemory M (forwardConfig shape scale) s values =
      normalizerStore shape (launchTile shape s)
        (rectangleStore (outputRectangle shape (launchTile shape s) 128) s
          (fun i => M.cast (some .fp32) .real .real (values.output i)))
        (fun r => M.cast (some .fp32) .real .real (values.normalizer ⟨r,()⟩)) := by
  simp only [outputMemory,normalizerStore,rectangleStore,outputRectangle,launchTile,forwardConfig,
    queryChunk,valueChunk,Rectangle.active,Rectangle.address,normalizerAddress,Nat.mul_one,Nat.add_assoc]

#axiomsClean entry_finish_cell
#axiomsClean finish_frame
#axiomsClean query_stores_algebra
#axiomsClean backward_stores_algebra
#axiomsClean forward_stores_algebra
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
