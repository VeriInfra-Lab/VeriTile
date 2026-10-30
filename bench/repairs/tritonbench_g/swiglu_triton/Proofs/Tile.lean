import bench.repairs.tritonbench_g.swiglu_triton.Kernels
import bench.tritonbench_g.swiglu_backward.Proofs.WholeTile

namespace VeriTile.Bench.Repairs.SwigluTriton.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000

/-- Source-ordered FP32 graph. The gate's explicit cast differs syntactically
from B/DC's FP32 load ports. No cast identity, FMA, reassociation or sigmoid
law is assumed by the structural interpreter. -/
def scalar (M : Algebra α) (ch : Channel) (a b dc : α) : α :=
  let a := M.cast (some .fp32) .real .real a
  let b := M.fp32Load b
  let dc := M.fp32Load dc
  let s := M.unary (some .fp32) .sigmoid a
  let one := M.literal (some .fp32) .real 1
  let mul := M.binary (some .fp32) .real .mul
  let silu := mul a s
  match ch with
  | .c => mul silu b
  | .da => mul (mul dc (M.binary (some .fp32) .real .add
      (mul silu (M.binary (some .fp32) .real .sub one s)) s)) b
  | .db => mul dc silu

def rowState {α : Type} [Inhabited α] (row : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then row else 0 }

/-- Full successful execution of all 10/15 statements of the original typed
source. This is constructive execution, not equality of two failed runs. -/
def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) :
    { t : State α // run M (source backward block cols).surfaceBody (rowState row s) = some t } := by
  cases backward
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 900000) [source, forwardKernel, backwardKernel,
      typedFwd, typedBwd, typedBwd_fp32_fp32, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, natLt, rowState, ofFloat, toFloat, TileShape.axisDim,
      Region.cast, State.setReg]
    rfl

def raw {α : Type} [Inhabited α] (M : Algebra α) (block cols row : Nat)
    (s : State α) (region : RegionName) (i : TileIndex [block]) : α :=
  if i.1.val < cols then (s.mem region (row*block+i.1.val)).read .real
  else M.literal (some .fp32) .real 0

def scatter {α : Type} [Inhabited α] (M : Algebra α) (block cols row : Nat)
    (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [block]).foldl (fun mem i =>
    if i.1.val < cols then
      writeCell mem (outputRegion ch) (row*block+i.1.val)
        (.mk .real (scalar M ch (raw M block cols row s "A" i)
          (raw M block cols row s "B" i) (raw M block cols row s "DC" i)))
    else mem) mem

def memory {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) : RegionName → Nat → Cell α :=
  if backward then scatter M block cols row s .db (scatter M block cols row s .da s.mem)
  else scatter M block cols row s .c s.mem

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) :
    (sourceRun M backward block cols row s).val.mem = memory M backward block cols row s := by
  cases backward <;> simp only [sourceRun, store, scatter_mem]
  all_goals rfl

theorem scatter_readback {α : Type} [Inhabited α] (M : Algebra α)
    (block cols row : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α)
    (target : TileIndex [block]) (active : target.1.val < cols) :
    scatter M block cols row s ch mem (outputRegion ch) (row*block+target.1.val) =
      .mk .real (scalar M ch
        ((s.mem "A" (row*block+target.1.val)).read .real)
        ((s.mem "B" (row*block+target.1.val)).read .real)
        ((s.mem "DC" (row*block+target.1.val)).read .real)) := by
  have unique : ∀ i : TileIndex [block], i.1.val < cols →
      row*block+i.1.val = row*block+target.1.val → i=target := by
    intro i _ h; apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ := List.append_of_mem (TileShape.mem_allIndices [block] target)
  unfold scatter
  rw [hall, List.foldl_append, List.foldl_cons]
  simp only [active, if_true]
  suffices h : _ = Cell.mk .real (scalar M ch (raw M block cols row s "A" target)
      (raw M block cols row s "B" target) (raw M block cols row s "DC" target)) by
    simpa only [raw, active, if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell, raw, active]

theorem scatter_frame {α : Type} [Inhabited α] (M : Algebra α)
    (block cols row : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α)
    (region : RegionName) (address : Nat)
    (outside : region ≠ outputRegion ch ∨ ∀ i : TileIndex [block],
      i.1.val < cols → address ≠ row*block+i.1.val) :
    scatter M block cols row s ch mem region address = mem region address :=
  fold_frame _ _ _ _ _ _ _ _ outside

def Outside (backward : Bool) (block cols row : Nat) (region : RegionName) (address : Nat) : Prop :=
  ∀ ch, enabled backward ch → region ≠ outputRegion ch ∨
    ∀ i : TileIndex [block], i.1.val < cols → address ≠ row*block+i.1.val

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) (region : RegionName) (address : Nat)
    (outside : Outside backward block cols row region address) :
    (sourceRun M backward block cols row s).val.mem region address = s.mem region address := by
  rw [sourceRun_mem]
  cases backward <;> simp only [memory, if_false, if_true]
  · exact scatter_frame _ _ _ _ _ _ _ _ _ (outside .c rfl)
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ (outside .db (by decide))]
    exact scatter_frame _ _ _ _ _ _ _ _ _ (outside .da (by decide))

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) (ch : Channel) (on : enabled backward ch)
    (target : TileIndex [block]) (active : target.1.val < cols) :
    (sourceRun M backward block cols row s).val.mem (outputRegion ch) (row*block+target.1.val) =
      .mk .real (scalar M ch
        ((s.mem "A" (row*block+target.1.val)).read .real)
        ((s.mem "B" (row*block+target.1.val)).read .real)
        ((s.mem "DC" (row*block+target.1.val)).read .real)) := by
  rw [sourceRun_mem]
  cases backward <;> cases ch <;> simp only [memory, if_false, if_true]
  · exact scatter_readback _ _ _ _ _ _ _ _ active
  · simp [enabled] at on
  · simp [enabled] at on
  · simp [enabled] at on
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
    exact scatter_readback _ _ _ _ _ _ _ _ active
  · exact scatter_readback _ _ _ _ _ _ _ _ active

#axiomsClean sourceRun
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Tile
