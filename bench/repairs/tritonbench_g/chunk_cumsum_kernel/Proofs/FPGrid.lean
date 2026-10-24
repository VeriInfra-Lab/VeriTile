import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.FPIndexBridge
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Host

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.FPGrid
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Layout
noncomputable section

def atRow {α : Type} (rows pid : Nat) (s : State α) : State α :=
  {s with regs := fun _ _ _ => none
          pids := fun ax => if ax=0 then pid else 0
          numPids := fun ax => if ax=0 then rows else 1}

/-- Every launch executes complete source rows on the evolving memory. -/
def run {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T rows : Nat) (cfg : Config) :
    List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (FP.Structural.exec (engine ops.algebra)
      (repairedKernel .fp32 .fp32 input output T cfg.block) (atRow rows pid s)).bind
        (run ops input output T rows cfg rest)

def Frame {α : Type} (output : RegionName) (T : Nat) (order : List Nat) (s t : State α) : Prop :=
  ∀ rg off,(rg≠output ∨ ∀ pid∈order,∀ i,i<T → off≠pid*T+i) → t.mem rg off=s.mem rg off

theorem complete {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T rows : Nat)
    (cfg : Config) (order : List Nat) (s : State α) :
    ∃ t,run ops input output T rows cfg order s=some t ∧ Frame output T order s t := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,hm,hf⟩ := FPIndexBridge.complete ops input output T cfg.block (atRow rows pid s)
    obtain ⟨t,ht,tf⟩ := ih mid
    refine ⟨t,?_,?_⟩
    · simp only [run,hm,Option.bind_some]; exact ht
    · intro rg off hn
      have tail : rg≠output ∨ ∀ p∈rest,∀ i,i<T → off≠p*T+i := by
        rcases hn with h | h
        · exact Or.inl h
        · exact Or.inr (fun p hp i hi => h p (List.mem_cons_of_mem _ hp) i hi)
      rw [tf rg off tail]
      apply hf rg off
      rcases hn with h | h
      · exact Or.inl h
      · exact Or.inr (fun i hi => by simpa [atRow] using h pid List.mem_cons_self i hi)

def source {α : Type} [Inhabited α] (ops : Config → FPRow.Ops α)
    (c : Shape) (input output : RegionName) :
    Host.Source c (State α) (Cell α) where
  input := fun s i => s.mem input (offset c i)
  output := fun s i => s.mem output (offset c i)
  launch := fun cfg => run (ops cfg) input output c.T (c.B*c.H) cfg (List.range (c.B*c.H))

/-- Constructive full autotune/selected execution. The frame retains every
tagged cell outside the row windows, not only the input projection. -/
theorem choices_complete {α : Type} [Inhabited α] (ops : Config → FPRow.Ops α)
    (c : Shape) (input output : RegionName)
    (choices : List Config) (s : State α) :
    ∃ t,Host.sourceLaunches (source ops c input output) choices s=some t ∧
      Frame output c.T (List.range (c.B*c.H)) s t := by
  induction choices generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons cfg rest ih =>
    obtain ⟨mid,hm,hf⟩ := complete (ops cfg) input output c.T (c.B*c.H) cfg
      (List.range (c.B*c.H)) s
    obtain ⟨t,ht,tf⟩ := ih mid
    refine ⟨t,?_,fun rg off hn => (tf rg off hn).trans (hf rg off hn)⟩
    change (run (ops cfg) input output c.T (c.B*c.H) cfg _ s).bind _=some t
    rw [hm]
    exact ht

theorem launch_total {α : Type} [Inhabited α] (ops : Config → FPRow.Ops α)
    (c : Shape) (input output : RegionName)
    (cfg : Config) (s : State α) :
    ∃ t,(source ops c input output).launch cfg s=some t := by
  obtain ⟨t,ht,_⟩ := complete (ops cfg) input output c.T (c.B*c.H) cfg (List.range (c.B*c.H)) s
  exact ⟨t,ht⟩

/-- Reference uses the original mathematical addressing on the packed words;
it is not the defective native original wrapper on a noncontiguous tensor. -/
def referenceRun {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T rows : Nat) (cfg : Config) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (FP.Structural.exec (engine ops.algebra)
      (kernel .fp32 .fp32 input output T cfg.block) (atRow rows pid s)).bind
        (referenceRun ops input output T rows cfg rest)

def Observed {α : Type} (a b : State α) := a.mem=b.mem ∧ a.undef=b.undef

theorem atRow_eq {α : Type} (rows pid : Nat) (a b : State α) (h : Observed a b) :
    atRow rows pid a=atRow rows pid b := by
  cases a
  cases b
  simp_all [Observed,atRow]

theorem equivalent {α : Type} [Inhabited α] (ops : FPRow.Ops α)
    (input output : RegionName) (T rows : Nat) (cfg : Config) (order : List Nat)
    (a b : State α) (related : Observed a b) :
    ∃ x y,run ops input output T rows cfg order a=some x ∧
      referenceRun ops input output T rows cfg order b=some y ∧ Observed x y := by
  induction order generalizing a b with
  | nil => exact ⟨a,b,rfl,rfl,related⟩
  | cons pid rest ih =>
    obtain ⟨midA,midB,ha,hb,hr⟩ := FPIndexBridge.complete_related ops input output T cfg.block (atRow rows pid a)
    have hrow := atRow_eq rows pid a b related
    rw [hrow] at hb
    obtain ⟨x,y,hx,hy,hxy⟩ := ih midA midB ⟨hr.1,hr.2.1⟩
    exact ⟨x,y,by simpa only [run,ha,Option.bind_some] using hx,
      by simpa only [referenceRun,hb,Option.bind_some] using hy,hxy⟩

def referenceSource {α : Type} [Inhabited α] (ops : Config → FPRow.Ops α)
    (c : Shape) (input output : RegionName) : Host.Source c (State α) (Cell α) where
  input := fun s i => s.mem input (offset c i)
  output := fun s i => s.mem output (offset c i)
  launch := fun cfg => referenceRun (ops cfg) input output c.T (c.B*c.H) cfg (List.range (c.B*c.H))

theorem choices_equivalent {α : Type} [Inhabited α] (ops : Config → FPRow.Ops α)
    (c : Shape) (input output : RegionName) (choices : List Config)
    (a b : State α) (related : Observed a b) :
    ∃ x y,Host.sourceLaunches (source ops c input output) choices a=some x ∧
      Host.sourceLaunches (referenceSource ops c input output) choices b=some y ∧ Observed x y := by
  induction choices generalizing a b with
  | nil => exact ⟨a,b,rfl,rfl,related⟩
  | cons cfg rest ih =>
    obtain ⟨midA,midB,ha,hb,hr⟩ := equivalent (ops cfg) input output c.T (c.B*c.H) cfg
      (List.range (c.B*c.H)) a b related
    obtain ⟨x,y,hx,hy,hxy⟩ := ih midA midB hr
    exact ⟨x,y,by simpa only [Host.sourceLaunches,source,ha,Option.bind_some] using hx,
      by simpa only [Host.sourceLaunches,referenceSource,hb,Option.bind_some] using hy,hxy⟩

#axiomsClean equivalent
#axiomsClean choices_equivalent
#axiomsClean complete
#axiomsClean choices_complete
#axiomsClean launch_total
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.FPGrid
