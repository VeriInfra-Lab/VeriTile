import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.Scalar
import VeriTile.Triton.Float.ExecutionProfile
import bench.tritonbench_g.swiglu_backward.Proofs.WholeTile

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000


def rowState {α : Type} [Inhabited α] (row : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then row else 0 }

private theorem index_toNat (row block lane : Nat) :
    ((row : Int)*(block : Int)+(lane : Int)).toNat=row*block+lane := by
  rw [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

private theorem index_lt (row block lane count : Nat) :
    ((row : Int)*(block : Int)+(lane : Int)<(count : Int)) ↔ row*block+lane<count := by
  rw [← Int.natCast_mul, ← Int.natCast_add]
  omega

/-- Full successful execution of every statement of the repaired typed source. This is constructive execution, not equality of two failed runs. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols : Nat) (s : State α) :
    { t : State α // run M (source backward block cols).surfaceBody s = some t } := by
  cases backward
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 900000) [source, forwardKernel, backwardKernel,
      ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, store, Option.bind, ComputeDType.eraseDType,
      numeric, numericLt, bop, natLt,  rowState, ofFloat, toFloat, TileShape.axisDim,
      Region.cast, State.setReg, index_toNat, index_lt]
    rfl

def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols row : Nat) (s : State α) :
    { t : State α // run M (source backward block cols).surfaceBody (rowState row s) = some t } :=
  execution M backward block cols (rowState row s)

def raw {α : Type} [Inhabited α] (M : Algebra α) (block cols row : Nat)
    (s : State α) (region : RegionName) (i : TileIndex [block]) : α :=
  if row*block+i.1.val < cols then (s.mem region (row*block+i.1.val)).read .real
  else M.literal (some .fp32) .real 0

def scatter {α : Type} [Inhabited α] (M : Algebra α) (block cols row : Nat)
    (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [block]).foldl (fun mem i =>
    if row*block+i.1.val < cols then
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
  cases backward <;> simp only [sourceRun, execution, store, scatter_mem]
  all_goals rfl

theorem scatter_readback {α : Type} [Inhabited α] (M : Algebra α)
    (block cols row : Nat) (s : State α) (ch : Channel) (mem : RegionName → Nat → Cell α)
    (target : TileIndex [block]) (active : row*block+target.1.val < cols) :
    scatter M block cols row s ch mem (outputRegion ch) (row*block+target.1.val) =
      .mk .real (scalar M ch
        ((s.mem "A" (row*block+target.1.val)).read .real)
        ((s.mem "B" (row*block+target.1.val)).read .real)
        ((s.mem "DC" (row*block+target.1.val)).read .real)) := by
  have unique : ∀ i : TileIndex [block], row*block+i.1.val < cols →
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
      row*block+i.1.val < cols → address ≠ row*block+i.1.val) :
    scatter M block cols row s ch mem region address = mem region address :=
  fold_frame _ _ _ _ _ _ _ _ outside

def Outside (backward : Bool) (block cols row : Nat) (region : RegionName) (address : Nat) : Prop :=
  ∀ ch, enabled backward ch → region ≠ outputRegion ch ∨
    ∀ i : TileIndex [block], row*block+i.1.val < cols → address ≠ row*block+i.1.val

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
    (target : TileIndex [block]) (active : row*block+target.1.val < cols) :
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
#axiomsClean execution
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Tile
