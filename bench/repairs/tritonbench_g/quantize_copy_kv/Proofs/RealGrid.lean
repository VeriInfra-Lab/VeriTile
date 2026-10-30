import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Contract
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Real

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Mathematical
noncomputable section
open Classical
set_option maxHeartbeats 4000000

/-- Mathematical Tensor state. Input and scale cells are arbitrary Reals; code
cells retain the source's signed-byte representation. No finite decoder occurs. -/
structure State where
  input : Nat → Nat → Nat → ℝ
  codes : Nat → Byte
  scales : Nat → ℝ

def scatter {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Nat → α) : Nat → α :=
  events.foldl (fun state e => Function.update state (key e) (value e)) before

theorem scatter_frame {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Nat → α)
    (x : Nat) (miss : ∀ e∈events, x≠key e) : scatter events key value before x=before x := by
  induction events generalizing before with
  | nil => rfl
  | cons e tail ih =>
    change scatter tail key value (Function.update before (key e) (value e)) x=before x
    rw [ih _ (fun f hf => miss f (by simp [hf])),Function.update_of_ne (miss e (by simp))]

theorem scatter_stable {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Nat → α)
    (x : Nat) (v : α) (initial : before x=v) (agree : ∀ e∈events, key e=x → value e=v) :
    scatter events key value before x=v := by
  induction events generalizing before with
  | nil => exact initial
  | cons e tail ih =>
    apply ih
    · change Function.update before (key e) (value e) x=v
      by_cases eq : key e=x
      · rw [eq,Function.update_self]; exact agree e (by simp) eq
      · rw [Function.update_of_ne (Ne.symm eq)]; exact initial
    · intro f hf; exact agree f (by simp [hf])

theorem scatter_read {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (before : Nat → α)
    (e : ι) (member : e∈events) (agree : ∀ f∈events, key f=key e → value f=value e) :
    scatter events key value before (key e)=value e := by
  obtain ⟨a,b,he⟩ := List.append_of_mem member
  rw [he]
  unfold scatter
  rw [List.foldl_append,List.foldl_cons]
  apply scatter_stable
  · simp
  · intro f hf eq
    exact agree f (by rw [he]; simp [hf]) eq

def heads (p : Request) : List (Nat × Nat) :=
  (List.range p.tokens).flatMap fun s =>
    (List.range (headBlock p)).filterMap fun h => if h<p.heads then some (s,h) else none

def cells (p : Request) : List (Nat × Nat × Nat) :=
  (heads p).flatMap fun sh => (List.range p.columns).map fun c => (sh.1,sh.2,c)

theorem heads_member (p : Request) (s h : Nat) : (s,h)∈heads p ↔ s<p.tokens ∧ h<p.heads := by
  simp only [heads,List.mem_flatMap,List.mem_range,List.mem_filterMap]
  constructor
  · rintro ⟨t,ht,a,ha,he⟩
    split_ifs at he with active
    · cases he; exact ⟨ht,active⟩
  · rintro ⟨hs,hh⟩
    exact ⟨s,hs,h,lt_of_lt_of_le hh (headBlock_covers p),by simp [hh]⟩

theorem cells_member (p : Request) (s h c : Nat) : (s,h,c)∈cells p ↔ s<p.tokens ∧ h<p.heads ∧ c<p.columns := by
  simp only [cells,List.mem_flatMap,List.mem_map,List.mem_range]
  constructor
  · rintro ⟨⟨t,a⟩,hta,b,hb,he⟩
    cases he
    exact ⟨(heads_member p _ _).mp hta |>.1,(heads_member p _ _).mp hta |>.2,hb⟩
  · rintro ⟨hs,hh,hc⟩
    exact ⟨(s,h),(heads_member p s h).mpr ⟨hs,hh⟩,c,hc,rfl⟩

def codeKey (p : Request) (v : Layout) (e : Nat × Nat × Nat) : Nat :=
  outputAddress p v (p.destination e.1) e.2.1 e.2.2

def scaleKey (p : Request) (v : Layout) (e : Nat × Nat) : Nat :=
  scaleAddress p v (p.destination e.1) e.2 ⟨0,by decide⟩

/-- The source row registers are evaluated at every actual head-mask event.
Read-only mathematical input cells retain their initial values, as justified by
checked separation in the physical execution proof. -/
def run (p : Request) (plan : Plan p) (v : Layout) (before : State) : State :=
  ⟨before.input,
    scatter (cells p) (codeKey p v)
      (fun e => (sourceRegisters realOps plan.tree (before.input e.1 e.2.1)).1 e.2.2) before.codes,
    scatter (heads p) (scaleKey p v)
      (fun e => (sourceRegisters realOps plan.tree (before.input e.1 e.2)).2) before.scales⟩

/-- Independent logical Real target, with no execution or source-register reference. -/
def Quantized (p : Request) (v : Layout) (before after : State) : Prop :=
  after.input=before.input ∧
  (∀ s h c, s<p.tokens → h<p.heads → c<p.columns →
    after.codes (outputAddress p v (p.destination s) h c)=mathematicalCode p (before.input s h) c) ∧
  (∀ s h, s<p.tokens → h<p.heads →
    after.scales (scaleAddress p v (p.destination s) h ⟨0,by decide⟩)=mathematicalScale p (before.input s h)) ∧
  (∀ a, (∀ s h c, s<p.tokens → h<p.heads → c<p.columns → a≠outputAddress p v (p.destination s) h c) →
    after.codes a=before.codes a) ∧
  (∀ a, (∀ s h, s<p.tokens → h<p.heads → a≠scaleAddress p v (p.destination s) h ⟨0,by decide⟩) →
    after.scales a=before.scales a)

theorem correct (p : Request) (d : Domain p) (plan : Plan p) (v : Layout) (before : State) :
    Quantized p v before (run p plan v before) := by
  refine ⟨rfl,?_,?_,?_,?_⟩
  · intro s h c hs hh hc
    change scatter _ _ _ _ _=_
    have read := scatter_read (cells p) (codeKey p v)
      (fun e => (sourceRegisters realOps plan.tree (before.input e.1 e.2.1)).1 e.2.2)
      before.codes (s,h,c) ((cells_member p s h c).mpr ⟨hs,hh,hc⟩) (by
        rintro ⟨t,k,e⟩ member same
        obtain ⟨ht,hk,he⟩ := (cells_member p t k e).mp member
        obtain ⟨slots,heads,columns⟩ := output_injective p v _ _ _ _ _ _ hk hh he hc same
        have tokens := d.destination_unique t s ht hs slots
        subst t k e; rfl)
    exact read.trans ((sourceRegisters_real p d plan (before.input s h)).1 c hc)
  · intro s h hs hh
    change scatter _ _ _ _ _=_
    have read := scatter_read (heads p) (scaleKey p v)
      (fun e => (sourceRegisters realOps plan.tree (before.input e.1 e.2)).2)
      before.scales (s,h) ((heads_member p s h).mpr ⟨hs,hh⟩) (by
        rintro ⟨t,k⟩ member same
        obtain ⟨ht,hk⟩ := (heads_member p t k).mp member
        obtain ⟨slots,heads,_⟩ := scale_injective p v _ _ _ _ _ _ hk hh same
        have tokens := d.destination_unique t s ht hs slots
        subst t k; rfl)
    exact read.trans ((sourceRegisters_real p d plan (before.input s h)).2)
  · intro a outside
    apply scatter_frame
    rintro ⟨s,h,c⟩ member
    obtain ⟨hs,hh,hc⟩ := (cells_member p s h c).mp member
    exact outside s h c hs hh hc
  · intro a outside
    apply scatter_frame
    rintro ⟨s,h⟩ member
    obtain ⟨hs,hh⟩ := (heads_member p s h).mp member
    exact outside s h hs hh

#axiomsClean correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Mathematical
