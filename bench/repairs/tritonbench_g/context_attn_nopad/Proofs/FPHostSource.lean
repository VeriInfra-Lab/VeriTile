import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeProgram
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPComplete
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Logical
import bench.tritonbench_g.context_attn_nopad.Proofs.FPHost

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPHostSource
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Host HostPlan
noncomputable section
set_option maxHeartbeats 2000000

/-- Exact integer metadata channel of the normalized FP state. -/
structure Metadata {α : Type} [Inhabited α] (c : Call) (r : Regions) (s : State α) : Prop where
  start : ∀ b, b < c.batches → (s.mem r.start b).read .int = (c.start b : Int)
  length : ∀ b, b < c.batches → (s.mem r.sequence b).read .int = (c.length b : Int)

def metadataMaximum {α : Type} [Inhabited α] (c : Call) (r : Regions) (s : State α) : Nat :=
  (Finset.univ : Finset (Fin c.batches)).sup
    (fun b => ((s.mem r.sequence b.val).read .int).toNat)

theorem metadataMaximum_eq {α : Type} [Inhabited α] {c : Call} {r : Regions} {s : State α}
    (hm : Metadata c r s) : metadataMaximum c r s = HostPlan.maximum c := by
  unfold metadataMaximum HostPlan.maximum
  congr 1
  funext b
  rw [hm.length b b.isLt, Int.toNat_natCast]

def effective {α : Type} [Inhabited α] (c : Call) (r : Regions) (s : State α) : Call :=
  NativeIndex.capCall c (metadataMaximum c r s)

/-- The integer projection is independent of FP payloads. `layoutState` only
supplies the exact metadata and the ordinary allocation/view bounds; no Real
execution or Real/FP output relation is a premise. -/
theorem finite_source_projection {α : Type} [Inhabited α]
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : Memory c r A layoutState) (ha : NativeProgram.ABI c A) (hi : Metadata c r s) :
    ∀ j : Job (effective c r s),
      NativeProgram.Projects (effective c r s) r A layoutState j := by
  have he : effective c r s = HostPlan.effective c r layoutState := by
    unfold effective HostPlan.effective
    rw [metadataMaximum_eq hi, HostPlan.metadataMaximum_eq hm]
  rw [he]
  exact NativeProgram.projects (HostPlan.effective_memory hm) ⟨ha.heads,ha.bytes⟩
    (NativeProgram.effective_blocks_bound hm)

theorem blocks_bound {α : Type} [Inhabited α] {c : Call} {r : Regions} {s : State α}
    (hs : Shape c) (hm : Metadata c r s) : (effective c r s).blocks ≤ 16777216 := by
  have hmax := HostPlan.maximum_lt_int32 hs
  rw [← metadataMaximum_eq hm] at hmax
  change (min c.maxInputLen (metadataMaximum c r s)+127)/128 ≤ _
  norm_num at hmax
  omega

def atProgram {α : Type} (c : Call) (j : Job c) (s : State α) : State α :=
  { s with pids := NativeProgram.nativePids c j, numPids := c.grid.toNumPids }

theorem atProgram_eq {α : Type} (c : Call) (ha : NativeProgram.ShapeABI c)
    (hb : c.blocks ≤ 16777216) (j : Job c) (s : State α) :
    atProgram c j s = FPHost.atProgram c (HostPlan.index c j) s := by
  unfold atProgram FPHost.atProgram
  rw [NativeProgram.nativePids_of_shape ha hb]

/-- Entire split-axis program stream, retaining the actual evolving state and
all typed arithmetic/casts/default precision. A primitive failure propagates. -/
def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Regions) :
    List (Job c) → State α → Option (State α)
  | [], s => some s
  | j :: rest, s =>
      (FP.Structural.exec (FP.ProfiledRewrite.engine M) (kernel c r) (atProgram c j s)).bind
        (runPrograms M c r rest)

theorem runPrograms_projects {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Regions)
    (ha : NativeProgram.ShapeABI c) (hb : c.blocks ≤ 16777216)
    (order : List (Job c)) (s : State α) :
    runPrograms M c r order s =
      FPHost.runJobs M c r Bool.false (order.map (HostPlan.index c)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons j rest ih =>
    simp only [runPrograms, List.map_cons, FPHost.runJobs, atProgram_eq c ha hb]
    change (FP.Structural.exec _ _ _).bind _ = (FP.Structural.exec _ _ _).bind _
    congr 1
    funext next
    exact ih next

/-- Construct successful execution of every program in the actual native
stream. Each step invokes the checked whole typed source success theorem. -/
theorem runPrograms_success {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Call) (r : Regions) (order : List (Job c)) (s : State α) :
    ∃ t, runPrograms o.algebra c r order s = some t ∧
      ∀ rg off, rg ≠ r.out → t.mem rg off = s.mem rg off := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,hm,hframe⟩ := FPSymbolic.complete o c.layout r c.scale
      (by change 0 < (128 : Nat); omega) (atProgram c j s)
    have hm' : FP.Structural.exec (FP.ProfiledRewrite.engine o.algebra)
        (kernel c r) (atProgram c j s) = some mid := hm
    obtain ⟨final,hf,hrest⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [runPrograms,hm',Option.bind_some,hf]
    · intro rg off hrg
      exact (hrest rg off hrg).trans (hframe rg off hrg)

/-- A caller view acts on opaque FP values; involution is the ordinary logical
negative-view contract, not an arithmetic approximation or kernel-result law. -/
structure OutputView (α : Type) (c : Call) where
  objectId : Nat
  allocationId : Nat
  address : Logical.Coord c → Nat
  injective : Function.Injective address
  action : α → α
  involution : Function.Involutive action

structure Result (α : Type) where
  objectId : Nat
  allocationId : Nat
  memory : Nat → α
  pythonReturn : Option Unit

def writeback {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (s : State α) : Nat → α := fun addr =>
  if h : ∃ i : Logical.Coord c, v.address i = addr then
    v.action ((s.mem r.out (Logical.coordOffset c (Classical.choose h))).read .real)
  else before addr

def finish {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (s : State α) : Result α :=
  ⟨v.objectId,v.allocationId,writeback r v before s,none⟩

theorem caller_readback {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (s : State α) (i : Logical.Coord c) :
    v.action ((finish r v before s).memory (v.address i)) =
      (s.mem r.out (Logical.coordOffset c i)).read .real := by
  dsimp only [finish,writeback]
  have he : ∃ j : Logical.Coord c, v.address j = v.address i := ⟨i,rfl⟩
  rw [dif_pos he]
  have hi : Classical.choose he = i := v.injective (Classical.choose_spec he)
  rw [hi,v.involution]

theorem caller_frame {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (s : State α) (addr : Nat)
    (h : ¬ ∃ i : Logical.Coord c, v.address i = addr) :
    (finish r v before s).memory addr = before addr := by
  simp only [finish,writeback,dif_neg h]

theorem caller_identity {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (s : State α) :
    (finish r v before s).objectId = v.objectId ∧
    (finish r v before s).allocationId = v.allocationId ∧
    (finish r v before s).pythonReturn = none := ⟨rfl,rfl,rfl⟩

/-- Actual complete host FP program: synchronize metadata, cap max length,
execute the complete native chunk stream, then write back to the caller view. -/
def runHost {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Regions)
    (s : State α) (v : OutputView α c) (before : Nat → α) : Option (Result α) :=
  let ec := effective c r s
  (runPrograms M ec r (HostPlan.jobs ec) s).map (finish r v before)

/-- Logical-PID execution of the unchanged original typed source, in the
candidate's actual complete program order, followed by identical caller IO.
This reference does not replace default TF32 dots or fuse source arithmetic. -/
def reference {α : Type} [Inhabited α] (M : Algebra α) (c : Call) (r : Regions)
    (s : State α) (v : OutputView α c) (before : Nat → α) : Option (Result α) :=
  let ec := effective c r s
  (FPHost.runJobs M ec r Bool.false ((HostPlan.jobs ec).map (HostPlan.index ec)) s).map
    (finish r v before)

theorem whole_host_preserved {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (r : Regions) (s : State α) (hs : Shape c) (ha : NativeProgram.ShapeABI c)
    (hm : Metadata c r s) (v : OutputView α c) (before : Nat → α) :
    runHost M c r s v before = reference M c r s v before := by
  dsimp only [runHost, reference]
  rw [runPrograms_projects M (effective c r s) r ⟨ha.batches,ha.heads⟩ (blocks_bound hs hm)]

/-- The default-profile source uses the `.real` AST storage channel; its
floating operations are opaque and `engine` sets actual FP32 compute precision.
Readback must use that exact stored channel, not an unrelated `.fp32` tag. -/
def Success {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (r : Regions) (s : State α) (v : OutputView α c) (before : Nat → α) : Prop :=
    ∃ raw result,
      runPrograms M (effective c r s) r (HostPlan.jobs (effective c r s)) s = some raw ∧
      runHost M c r s v before = some result ∧
      reference M c r s v before = some result ∧
      result = finish r v before raw ∧
      result.objectId = v.objectId ∧ result.allocationId = v.allocationId ∧
      result.pythonReturn = none ∧
      (∀ i, v.action (result.memory (v.address i)) =
        (raw.mem r.out (Logical.coordOffset c i)).read .real) ∧
      (∀ addr, (¬ ∃ i, v.address i = addr) → result.memory addr = before addr) ∧
      (∀ rg off, rg ≠ r.out → raw.mem rg off = s.mem rg off)

/-- Successful complete host run and its original caller readback, identity
and outside-view frame, constructed without an execution/output premise. -/
theorem whole_host_success {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Call) (r : Regions) (s : State α) (hs : Shape c) (ha : NativeProgram.ShapeABI c)
    (hm : Metadata c r s) (v : OutputView α c) (before : Nat → α) :
    Success o.algebra c r s v before := by
  obtain ⟨raw,hr,hframe⟩ := runPrograms_success o (effective c r s) r (HostPlan.jobs _) s
  have hh : runHost o.algebra c r s v before = some (finish r v before raw) := by
    dsimp only [runHost]
    rw [hr]
    rfl
  refine ⟨raw,finish r v before raw,hr,hh,?_,rfl,rfl,rfl,rfl,?_,?_,hframe⟩
  · rw [← whole_host_preserved o.algebra c r s hs ha hm v before]
    exact hh
  · exact caller_readback r v before raw
  · exact caller_frame r v before raw

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPHostSource
