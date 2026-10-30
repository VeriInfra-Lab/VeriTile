import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostConfig
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeFP
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.AutotunePlan
import VeriTile.Triton.Float.ProfiledSoundness

/-! Complete FP host composition. Each lane executes the actual repaired typed
source from the common pre-launch memory. Its output cells are merged by the
source's decoded ownership, then the next legal batch sees that merged memory.
The domain checks only the original source's local reached sites. Shared tensor
materialization, slicing, casts, copy-back and empty branches are ordinary
deterministic host operations and require no numerical identity. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HostFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

def gridState {α : Type} (s : State α) (size lane : Nat) : State α :=
  { s with
    pids := fun ax => if ax=0 then lane else 0
    numPids := fun ax => if ax=0 then size else 1 }

abbrev Ownership (size : Nat) := Fin size → RegionName → Nat → Prop

/-- Ordinary simultaneous disjoint-output merge. No partial launch is committed
if a lane fails. Equality below also preserves failure, rather than assuming
successful execution. Ownership uniqueness is proved separately in Real. -/
def merge {α : Type} (s : State α) (owns : Ownership size)
    (results : Fin size → Option (State α)) : Option (State α) := by
  classical
  exact if h : ∀ i, (results i).isSome then
    some {s with mem := fun r off =>
      if ho : ∃ i, owns i r off then
        ((results (Classical.choose ho)).get (h (Classical.choose ho))).mem r off
      else (s.mem r off)}
  else none

def launch {α : Type} [Inhabited α] (M : Algebra α) (program : Nat → Program)
    (owns : (p : LaunchPlan.Item) → Ownership p.size) (p : LaunchPlan.Item)
    (s : State α) : Option (State α) :=
  merge s (owns p) (fun i => FP.Structural.exec (engine M) (program p.base).kernel
    (gridState s p.size i.val))

def batches {α : Type} [Inhabited α] (M : Algebra α) (program : Nat → Program)
    (owns : (p : LaunchPlan.Item) → Ownership p.size) : List LaunchPlan.Item → State α → Option (State α)
  | [],s => some s
  | p::ps,s => (launch M program owns p s).bind (batches M program owns ps)

/-- Local finite-operand domains only, at each original lane actually reached
after previous batches. This contains no output equation or success premise. -/
def Domains {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (program : Nat → Program) (owns : (p : LaunchPlan.Item) → Ownership p.size) :
    List LaunchPlan.Item → State α → Prop
  | [],_ => True
  | p::ps,s =>
    (∀ i : Fin p.size, ∀ site ∈ (program p.base).domain,
      site.Holds M D (gridState s p.size i.val)) ∧
    (∀ t, launch M program owns p s=some t → Domains M D program owns ps t)

theorem launch_equiv {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (R : Spec.Assumptions FP.GuardedFragment) (hM : FP.Guarded.Models R M D)
    (lhs rhs : Nat → Program) (he : ∀base, Equivalent R (lhs base) (rhs base))
    (owns : (p : LaunchPlan.Item) → Ownership p.size) (p : LaunchPlan.Item) (s : State α)
    (hd : ∀ i : Fin p.size, ∀ site ∈ (lhs p.base).domain,
      site.Holds M D (gridState s p.size i.val)) :
    launch M lhs owns p s=launch M rhs owns p s := by
  unfold launch
  congr 1
  funext i
  exact he p.base α M D hM _ (hd i)

theorem batches_equiv {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (R : Spec.Assumptions FP.GuardedFragment) (hM : FP.Guarded.Models R M D)
    (lhs rhs : Nat → Program) (he : ∀base, Equivalent R (lhs base) (rhs base))
    (owns : (p : LaunchPlan.Item) → Ownership p.size) (ps : List LaunchPlan.Item)
    (s : State α) (hd : Domains M D lhs owns ps s) :
    batches M lhs owns ps s=batches M rhs owns ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨hlocal,hnext⟩ := hd
    simp only [batches]
    rw [←launch_equiv M D R hM lhs rhs he owns p s hlocal]
    cases hr : launch M lhs owns p s with
    | none => rfl
    | some t => exact ih t (hnext t hr)

def Inner.program (c : Inner) (swapped : Bool) (base : Nat) : Program :=
  (if swapped then NativeFP.recurrentSumLeft else NativeFP.recurrent)
    c.dot c.gate c.history c.seed c.final c.k c.v c.g c.h c.h0 c.ht
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal

def Inner.owns (c : Inner) (p : LaunchPlan.Item) : Ownership p.size := fun i rg off =>
  let n := p.base+i.val
  let head := Grid.decodeZ (c.PV/c.BV) (c.PK/c.BK) n
  let key := Grid.decodeY (c.PV/c.BV) (c.PK/c.BK) n*c.BK
  let value := Grid.decodeX (c.PV/c.BV) n*c.BV
  ∃ e<c.BK, ∃ f<c.BV,
    (rg=c.h ∧ ∃ t<c.NT, off=head*c.NT*c.PK*c.PV+t*c.PK*c.PV+(key+e)*c.PV+value+f) ∨
    (c.storeFinal=Bool.true ∧ rg=c.ht ∧ off=head*c.PK*c.PV+(key+e)*c.PV+value+f)

theorem inner_batches {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models R.assumptions M D)
    (c : Inner) (s : State α)
    (hd : Domains M D (c.program Bool.false) c.owns c.plans s) :
    batches M (c.program Bool.false) c.owns c.plans s =
      batches M (c.program Bool.true) c.owns c.plans s := by
  apply batches_equiv M D R.assumptions hM _ _ _ _ _ s hd
  intro base
  exact FP.ProfiledRewrite.sound (NativeFP.recurrent_equiv
    c.dot c.gate c.history c.seed c.final c.k c.v c.g c.h c.h0 c.ht
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal R)

def Cumulative.program (c : Cumulative) (swapped : Bool) (base : Nat) : Program :=
  (if swapped then NativeFP.cumulativeIndexFirst else NativeFP.cumulative)
    c.dtype c.input c.output (c.PT*c.PS) c.PS 1 c.PT c.PS c.BT c.BS c.heads base
def Cumulative.owns (c : Cumulative) (p : LaunchPlan.Item) : Ownership p.size := fun i rg off =>
  let n := p.base+i.val
  let head := Grid.decodeZ (c.PS/c.BS) (c.PT/c.BT) n
  let time := Grid.decodeY (c.PS/c.BS) (c.PT/c.BT) n*c.BT
  let feature := Grid.decodeX (c.PS/c.BS) n*c.BS
  n<c.total ∧ rg=c.output ∧ ∃ t<c.BT, ∃ f<c.BS,
    off=head*c.PT*c.PS+(time+t)*c.PS+feature+f

theorem cumulative_domains {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (c : Cumulative) (ps : List LaunchPlan.Item) (s : State α) :
    Domains M D (c.program Bool.false) c.owns ps s := by
  induction ps generalizing s with
  | nil => trivial
  | cons p ps ih =>
    constructor
    · intro i site hs
      simp [Cumulative.program,NativeFP.cumulative] at hs
    · exact fun t _ => ih t

theorem cumulative_batches {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models [] M D) (c : Cumulative) (s : State α) :
    batches M (c.program Bool.false) c.owns c.plans s =
      batches M (c.program Bool.true) c.owns c.plans s := by
  apply batches_equiv M D [] hM _ _ _ _ _ s (cumulative_domains M D c c.plans s)
  intro base
  exact FP.ProfiledRewrite.sound (NativeFP.cumulative_equiv
    c.dtype c.input c.output (c.PT*c.PS) c.PS 1 c.PT c.PS c.BT c.BS c.heads base)

/-- Shared host adapters. `prepare` is private role-by-role logical
materialization, `finish` is history/gate slicing and optional caller-view
copy-back, and `empty` retains the source's actual zero-plus-seed and casts.
All three are the SAME operation on both sides; no equality is an assumption. -/
structure Adapters (Input Output α : Type) where
  prepare : Input → Option (State α)
  finish : Input → State α → Option Output
  empty : Input → Option Output

def host {Input Output α : Type} (a : Adapters Input Output α) (empty : Bool)
    (runGrid : State α → Option (State α)) (x : Input) : Option Output :=
  if empty then a.empty x else (a.prepare x).bind (fun s => (runGrid s).bind (a.finish x))

theorem inner_host {Input Output α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (c : Inner) (a : Adapters Input Output α) (x : Input)
    (hd : c.empty=Bool.false → ∀ s, a.prepare x=some s →
      Domains M D (c.program Bool.false) c.owns c.plans s) :
    host a c.empty (batches M (c.program Bool.false) c.owns c.plans) x =
      host a c.empty (batches M (c.program Bool.true) c.owns c.plans) x := by
  unfold host
  cases hempty : c.empty with
  | «true» => rfl
  | «false» =>
    simp only [Bool.false_eq_true,↓reduceIte]
    cases hs : a.prepare x with
    | none => rfl
    | some s =>
      simp only [Option.bind_some]
      rw [inner_batches R M D hM c s (hd hempty s hs)]

theorem cumulative_host {Input Output α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (hM : FP.Guarded.Models [] M D)
    (c : Cumulative) (a : Adapters Input Output α) (x : Input) :
    host a c.empty (batches M (c.program Bool.false) c.owns c.plans) x =
      host a c.empty (batches M (c.program Bool.true) c.owns c.plans) x := by
  unfold host
  cases c.empty with
  | «true» => rfl
  | «false» =>
    simp only [Bool.false_eq_true,↓reduceIte]
    cases a.prepare x with
    | none => rfl
    | some s =>
      simp only [Option.bind_some]
      rw [cumulative_batches M D hM c s]

#print_fp_assumptions NativeFP.cumulative_equiv
#print_fp_assumptions NativeFP.recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HostFP
