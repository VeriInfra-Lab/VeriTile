import bench.repairs.tritonbench_g.kldiv_triton.Kernels
import bench.tritonbench_g.swiglu_backward.Proofs.WholeTile

namespace VeriTile.Bench.Repairs.KldivTriton.Tile
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Floating equality is a supported local primitive, not equality of outputs.
All other numerical operations remain the algebra's explicit local operations. -/
structure Model (α : Type) where
  alg : Algebra α
  eq : α → α → Bool
  eq_ok : alg.compareEq (some .fp32) .real = some eq

def zero {α : Type} (m : Model α) : α := m.alg.literal (some .fp32) .real 0.0

def rawLoss {α : Type} (m : Model α) (y t : α) : α :=
  m.alg.binary (some .fp32) .real .mul t
    (m.alg.binary (some .fp32) .real .sub (m.alg.unary (some .fp32) .log t) y)

def loss {α : Type} (m : Model α) (flag : Bool) (y t : α) : α :=
  if flag then
    m.alg.binary (some .fp32) .real .mul (m.alg.unary (some .fp32) .exp t)
      (m.alg.binary (some .fp32) .real .sub t y)
  else if m.eq t (zero m) then zero m else rawLoss m y t

def gradient {α : Type} (m : Model α) (flag : Bool) (t : α) : α :=
  if flag then m.alg.unary (some .fp32) (.neg .real) (m.alg.unary (some .fp32) .exp t)
  else m.alg.binary (some .fp32) .real .mul t (m.alg.literal (some .fp32) .real (-1))

def loaded {α : Type} (m : Model α) (N col : Nat) (values : Nat → α) : α :=
  m.alg.fp32Load (if col < N then values col else zero m)

def laneLoss {α : Type} (m : Model α) (flag : Bool) (N col : Nat)
    (ys ts : Nat → α) : α :=
  if col < N then loss m flag (loaded m N col ys) (loaded m N col ts) else zero m

def chunkSum {α : Type} (m : Model α) (flag : Bool) (N B chunk : Nat)
    (ys ts : Nat → α) : α :=
  m.alg.reduceSum (some .fp32) (shape := [B]) 0 Bool.false
    (fun j => laneLoss m flag N (chunk * B + j.1.val) ys ts) PUnit.unit

/-- Inactive input coordinates do not affect a masked chunk reduction. -/
theorem chunkSum_congr {α : Type} (m : Model α) (flag : Bool) (N B k : Nat)
    (ys ts us vs : Nat → α) (hy : ∀ i, i < N → ys i = us i)
    (ht : ∀ i, i < N → ts i = vs i) :
    chunkSum m flag N B k ys ts = chunkSum m flag N B k us vs := by
  unfold chunkSum
  congr 2
  funext j
  by_cases active : k * B + j.1.val < N
  · simp only [laneLoss, loaded, active, if_true, hy _ active, ht _ active]
  · simp only [laneLoss, active, if_false]

def input {α : Type} [Inhabited α] (s : State α) (region : RegionName) (base : Nat) : Nat → α :=
  fun col => (s.mem region (base + col)).read .real

def forwardBody (sy st so N B : Nat) (flag : Bool) (reduction : Nat) : List ComputeStmt :=
  match ((forwardSource sy st so N B flag reduction).surfaceBody[5]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def forwardInitial {α : Type} (s : State α) (sy st so B : Nat) : State α :=
  ((((s.setReg "pid" .int [] (fun _ => (s.pids 0 : Int))).setReg
    "y_ptr" .ptr [] (fun _ => ("y_ptr", s.pids 0 * sy))).setReg
    "gt_ptr" .ptr [] (fun _ => ("gt_ptr", s.pids 0 * st))).setReg
    "loss_ptr" .ptr [] (fun _ => ("loss_ptr", s.pids 0 * so))).setReg
    "base_offsets" .nat [B] (fun j => j.1.val)

def f1 {α : Type} (s : State α) (B current : Nat) : State α :=
  s.setReg "offsets" .int [B] (fun j => ((current + j.1.val : Nat) : Int))

def f2 {α : Type} (s : State α) (N B current : Nat) : State α :=
  (f1 s B current).setReg "mask" .bool [B] (fun j => decide (current + j.1.val < N))

def f3 {α : Type} [Inhabited α] (m : Model α) (s : State α) (yBase N B current : Nat) : State α :=
  (f2 s N B current).setReg "y" .real [B]
    (fun j => loaded m N (current + j.1.val) (input s "y_ptr" yBase))

def f4 {α : Type} [Inhabited α] (m : Model α) (s : State α) (yBase tBase N B current : Nat) : State α :=
  (f3 m s yBase N B current).setReg "y_true" .real [B]
    (fun j => loaded m N (current + j.1.val) (input s "gt_ptr" tBase))

def f5 {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase N B current : Nat) (flag : Bool) : State α :=
  let ys := input s "y_ptr" yBase
  let ts := input s "gt_ptr" tBase
  let before := f4 m s yBase tBase N B current
  let after := if flag then before else before.setReg "loss" .real [B]
    (fun j => rawLoss m (loaded m N (current + j.1.val) ys) (loaded m N (current + j.1.val) ts))
  after.setReg "loss" .real [B]
    (fun j => loss m flag (loaded m N (current + j.1.val) ys) (loaded m N (current + j.1.val) ts))

def f6 {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase N B current : Nat) (flag : Bool) : State α :=
  (f5 m s yBase tBase N B current flag).setReg "loss" .real [B]
    (fun j => laneLoss m flag N (current + j.1.val) (input s "y_ptr" yBase) (input s "gt_ptr" tBase))

def scatter {α : Type} (s : State α) (region : RegionName) (base N B current : Nat)
    (values : TileIndex [B] → α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t j => if current + j.1.val < N then
    t.write region (base + (current + j.1.val)) (.mk .real (values j)) else t) s

def forwardResult {α : Type} [Inhabited α] (m : Model α) (s : State α)
    (yBase tBase oBase N B current : Nat) (flag : Bool) (reduction : Nat) : State α :=
  let middle := f6 m s yBase tBase N B current flag
  let values := fun j : TileIndex [B] =>
    laneLoss m flag N (current + j.1.val) (input s "y_ptr" yBase) (input s "gt_ptr" tBase)
  if reduction = 0 then scatter middle "loss_ptr" oBase N B current values
  else
    let sums := m.alg.reduceSum (some .fp32) (shape := [B]) 0 Bool.false values
    let sum := sums PUnit.unit
    ((middle.setReg "loss" .real [] sums).write "loss_ptr" oBase (.mk .real sum)).setReg
      "loss_ptr" .ptr [] (fun _ => ("loss_ptr", oBase + 1))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Tile
