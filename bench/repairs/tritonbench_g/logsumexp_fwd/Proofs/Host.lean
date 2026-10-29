import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Grid

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Host
open VeriTile.Triton FP.Structural VeriTile.Torch
noncomputable section
open Classical

def warpChoices : List Nat := [1,2,4,8,16,32]

/-- One original autotuning configuration and its compiler-selected engines. -/
structure Profile (A : Type) where
  warps : Nat
  choice : warps∈warpChoices
  engines : Nat → Engine A

/-- The retained Torch operations. Allocation freshness, contiguous view,
device ownership, Torch logsumexp and the optional FP64 cast are ordinary
library semantics. Native transcendental ports concern only the JIT body. -/
structure Library (A B : Type) where
  negInf : A
  logsumexp : (n : Nat) → (Fin n → A) → A
  cast64 : A → B

structure Returned (A B : Type) where
  shape : List Nat
  dtype : DType
  device : Device
  value : Nat → Sum A B

def castOutput (lib : Library A B) (a : Arguments) (x : A) : Sum A B :=
  if a.outputDType=.other "float64" then .inr (lib.cast64 x) else .inl x

def target (lib : Library A B) (a : Arguments) (engines : Nat → Engine A)
    (scale : A) (input : Nat → A) (row : Nat) : A :=
  if a.columns=0 then lib.negInf else
    lib.logsumexp a.blocks (fun b => Grid.compute engines a scale input (row*a.blocks+b.val))

def returned (lib : Library A B) (a : Arguments) (value : Nat → A) : Returned A B :=
  ⟨a.input.shape.dropLast,a.outputDType,a.input.device,fun row => castOutput lib a (value row)⟩

def trials [Inhabited A] (a : Arguments) (scale : A) : List (Profile A) → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
      let t ← Grid.run p.engines a scale (List.range (a.rows*a.blocks)) s
      trials a scale rest t

theorem trials_frame [Inhabited A] (a : Arguments) (scale : A) (probes : List (Profile A)) (s : State A) :
    ∃ t,trials a scale probes s=some t ∧
      (∀ (r : RegionName) (i : Nat),(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  induction probes generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,hm,_,hf⟩ := Grid.complete p.engines a scale s
    obtain ⟨t,ht,tf⟩ := ih mid
    exact ⟨t,by simp only [trials,hm]; exact ht,fun r i miss => (tf r i miss).trans (hf r i miss)⟩

/-- The actual checked wrapper: D=0 fills minus infinity; otherwise every
probe and the selected launch execute, then Torch reads the reached scratch.
N=0 has an empty grid and an empty returned tensor. Rank one returns shape []. -/
def run [Inhabited A] (lib : Library A B) (a : Arguments) (scale : A)
    (probes : List (Profile A)) (selected : Profile A) (s : State A) : Option (State A × Returned A B) := do
  match checkInputs a with
  | .error _ => none
  | .ok _ =>
    if a.columns=0 then some (s,returned lib a (fun _ => lib.negInf)) else do
      let mid ← trials a scale probes s
      let t ← Grid.run selected.engines a scale (List.range (a.rows*a.blocks)) mid
      let values := fun row => lib.logsumexp a.blocks
        (fun b => (t.mem "scratch" (row*a.blocks+b.val)).read .real)
      some (t,returned lib a values)

theorem complete [Inhabited A] (lib : Library A B) (a : Arguments) (h : InputContract a)
    (scale : A) (probes : List (Profile A)) (selected : Profile A) (s : State A) :
    ∃ t out,run lib a scale probes selected s=some (t,out) ∧
      out.shape=a.input.shape.dropLast ∧ out.dtype=a.outputDType ∧ out.device=a.input.device ∧
      (∀ row,row<a.rows → out.value row=castOutput lib a
        (target lib a selected.engines scale (fun i => (s.mem "input" i).read .real) row)) ∧
      (∀ (r : RegionName) (i : Nat),(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  have checked := (checkInputs_iff a).mpr h
  by_cases hd : a.columns=0
  · refine ⟨s,returned lib a (fun _ => lib.negInf),?_,rfl,rfl,rfl,?_,by intros; rfl⟩
    · simp [run,checked,hd]
    · intro row _
      simp [returned,target,hd]
  · obtain ⟨mid,hm,hf⟩ := trials_frame a scale probes s
    obtain ⟨t,ht,hv,tf⟩ := Grid.complete selected.engines a scale mid
    have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) := by
      funext i
      rw [hf "input" i (Or.inl (by decide))]
    refine ⟨t,returned lib a (fun row => lib.logsumexp a.blocks
      (fun b => (t.mem "scratch" (row*a.blocks+b.val)).read .real)),?_,rfl,rfl,rfl,?_,?_⟩
    · simp [run,checked,hd,hm,ht]
    · intro row hr
      change castOutput lib a _=castOutput lib a _
      apply congrArg (castOutput lib a)
      simp only [target,if_neg hd]
      apply congrArg (lib.logsumexp a.blocks)
      funext b
      rw [hv _ (output_offset_bound a row b.val hr b.isLt),input]
      rfl
    · exact fun r i miss => (tf r i miss).trans (hf r i miss)

theorem target_local (lib : Library A B) (a : Arguments) (h : InputContract a)
    (engines : Nat → Engine A) (scale : A) (f g : Nat → A) (row : Nat) (hr : row<a.rows)
    (same : ∀ i,i<a.input.numel → f i=g i) :
    target lib a engines scale f row=target lib a engines scale g row := by
  unfold target
  split
  · rfl
  · apply congrArg (lib.logsumexp a.blocks)
    funext b
    have coords := Grid.full_coverage a row b.val hr b.isLt
    simp only [Grid.compute,coords.2.1,coords.2.2]
    apply block_local
    intro i hi
    exact same _ (by simpa using input_offset_bound a h row 0 i hr (by simpa using hi))

#axiomsClean trials_frame
#axiomsClean complete
#axiomsClean target_local
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Host
