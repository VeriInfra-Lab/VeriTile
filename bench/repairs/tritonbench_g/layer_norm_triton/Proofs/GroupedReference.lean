import bench.repairs.tritonbench_g.layer_norm_triton.IO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedResult

/-! Independent ordered arithmetic and address-level overwrite recurrence for
one grouped-backward source program. Logical arrays are the only inputs; no
Structural state, executor, source body or memory updater occurs below. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.GroupedReference
open VeriTile.Triton FP.Structural SourceModel Interface SourceForwardGraph
noncomputable section
variable {α : Type} [Inhabited α]

def loaded (m : Model α) (N : Nat) (x : Fin B → α) (j : Fin B) : α :=
  cast m (if j.val<N then x j else lit m 0)
def xhat (m : Model α) (N : Nat) (x : Fin B → α) (mean inverse : α) (j : Fin B) : α :=
  if j.val<N then bin m .mul (bin m .sub (loaded m N x j) (m.scalar.fp32Load mean))
    (m.scalar.fp32Load inverse) else lit m 0

def wdy (m : Model α) (N : Nat) (dy w : Fin B → α) (j : Fin B) : α :=
  if j.val<N then bin m .mul (cast m (w j)) (loaded m N dy j) else lit m 0

def dx (m : Model α) (N B : Nat) (x dy w : Fin B → α) (mean inverse : α) (j : Fin B) : α :=
  let normalized := xhat m N x mean inverse
  let weighted := wdy m N dy w
  let first := bin m .div (sum m B (fun k => bin m .mul (normalized k) (weighted k))) (lit m N)
  let second := bin m .div (sum m B weighted) (lit m N)
  cast m (bin m .mul (bin m .sub (weighted j)
    (bin m .add (bin m .mul (normalized j) first) second)) (m.scalar.fp32Load inverse))

def partials (m : Model α) (N : Nat) (x dy : Fin B → α) (mean inverse : α)
    (same : Bool) (a : (Fin B → α)×(Fin B → α)) : (Fin B → α)×(Fin B → α) :=
  (fun j => cast m (let p := cast m (bin m .mul (loaded m N dy j) (xhat m N x mean inverse j));
    if same then p else bin m .add p (a.1 j)),
   fun j => cast m (let p := cast m (loaded m N dy j);
    if same then p else bin m .add p (a.2 j)))

abbrev Carries (α : Type) (B : Nat) := (Fin B → α)×(Fin B → α)

/-- A query address is overwritten exactly when an active source lane writes
it. Folding in source lane order handles overlapping rows without a new stride
restriction; zero group step keeps the prior value and zero partial carries. -/
def fold (m : Model α) (stride M N G B group cur query : Nat)
    (x dy : Nat → Fin B → α) (w : Fin B → α) (mean inverse : Nat → α)
    (prior : α) (a : Carries α B) : α × Carries α B :=
  if G=0 then (prior,a) else if cur<M then
    let output := (List.finRange B).foldl (fun v j =>
      if j.val<N ∧ query=cur*stride+j.val then dx m N B (x cur) (dy cur) w (mean cur) (inverse cur) j else v) prior
    fold m stride M N G B group (cur+G) query x dy w mean inverse output
      (partials m N (x cur) (dy cur) (mean cur) (inverse cur) (decide (cur=group)) a)
  else (prior,a)
termination_by M-cur
decreasing_by omega

def expected (m : Model α) (stride M N G B group : Nat)
    (observe : RegionName → Nat → α → α) (vals : Grouped.Inputs α M N B) : Grouped.Outputs α M N B :=
  let zero := (fun _ : Fin B => lit m 0,fun _ : Fin B => lit m 0)
  let result := fun query prior => fold m stride M N G B group group query
    (Grouped.x vals) (Grouped.dy vals) (Grouped.w vals) (Grouped.mean vals) (Grouped.inverse vals) prior zero
  Grouped.result
    (fun row j => observe "DX" (row.val*stride+j.val)
      (result (row.val*stride+j.val) (Grouped.priorDX vals row j)).1)
    (fun j => observe "DW" (group*N+j.val) ((result 0 default).2.1 j))
    (fun j => observe "DB" (group*N+j.val) ((result 0 default).2.2 j))
end
end VeriTile.Bench.Repairs.LayerNormTriton.GroupedReference
