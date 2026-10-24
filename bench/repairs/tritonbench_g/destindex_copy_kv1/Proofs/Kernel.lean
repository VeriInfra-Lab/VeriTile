import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Native

namespace VeriTile.Bench.Repairs.DestindexCopyKv1
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def initial (source output : Nat → α) (dest : Nat → Nat) (other : State α) : State α :=
  {other with mem := fun region x =>
    if region=inputRegion then .mk .real (source x)
    else if region=metadataRegion.cast then .mk .int (dest x)
    else if region=outputRegion then .mk .real (output x)
    else other.mem region x}

theorem initial_bound (source output : Nat → α) (dest : Nat → Nat) (other : State α) :
    Grid.Bound (initial source output dest other) source dest := by
  constructor
  · intro x; simp [initial]
  · intro x; simp [initial,inputRegion,metadataRegion,Region.cast]

@[simp] theorem initial_output (source output : Nat → α) (dest : Nat → Nat) (other : State α) (x : Nat) :
    ((initial source output dest other).mem outputRegion x).read .real=output x := by
  simp [initial,inputRegion,metadataRegion,outputRegion,Region.cast]

def launch (rest : Algebra α) (c : Config) (source output : Nat → α) (dest : Nat → Nat)
    (other : State α) : Nat → α :=
  fun x => ((Grid.run rest c (List.range c.count) (initial source output dest other)).mem outputRegion x).read .real

theorem launch_correct (rest : Algebra α) (c : Config) (source output : Nat → α)
    (dest : Nat → Nat) (other : State α)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q) :
    RowCopy c dest source output (launch rest c source output dest other) := by
  simpa only [initial_output] using Grid.correct rest c (List.range c.count)
    (initial source output dest other) source dest (initial_bound source output dest other)
    unique (by simp)

/-- Physical observation of the original output region after typed source
execution. It uses the original data pointer and does not allocate or copy back. -/
def project (base size : Nat) (before : Nat → α) (out : Nat → α) (cell : Nat) : α :=
  if base≤cell ∧ (cell-base)%4=0 ∧ (cell-base)/4<size
  then out ((cell-base)/4) else before cell

theorem project_at (base size : Nat) (before out : Nat → α) (x : Nat) (hx : x<size) :
    project base size before out (addr base x)=out x := by
  simp [project,addr,Nat.add_sub_cancel_left,hx]

theorem project_frame (base size : Nat) (before out : Nat → α) (selected : Nat → Prop)
    (values : ∀ x, x<size → ¬selected x → out x=before (addr base x)) (cell : Nat)
    (miss : ¬∃ x, selected x ∧ cell=addr base x) :
    project base size before out cell=before cell := by
  unfold project
  split_ifs with h
  · have eq : cell=addr base ((cell-base)/4) := by
      have div := Nat.mod_add_div (cell-base) 4
      unfold addr
      omega
    rw [values _ h.2.2 (by intro hx; exact miss ⟨_,hx,eq⟩),←eq]
  · rfl

/-- Reification commutes with an actual active local word store. This is a
local memory law proved from the mapping, rather than a whole-kernel premise. -/
theorem project_store (base size : Nat) (before out : Nat → α) (x : Nat) (hx : x<size) (v : α) :
    project base size before (Function.update out x v)=
      Function.update (project base size before out) (addr base x) v := by
  funext cell
  by_cases eq : cell=addr base x
  · subst cell
    simp [project_at base size before _ x hx]
  · rw [Function.update_of_ne eq]
    unfold project
    split_ifs with h
    · rw [Function.update_of_ne]
      intro he
      have div := Nat.mod_add_div (cell-base) 4
      unfold addr at eq
      omega
    · rfl

def touched (c : Config) (dest : Nat → Nat) (x : Nat) : Prop :=
  ∃ p a b, p<c.count ∧ a<c.heads ∧ b<c.dims ∧ x=flat c (dest p) a b

def execute (rest : Algebra α) (m : Inputs) (w : Nat → Word) (before : Nat → α) (other : State α) : Nat → α :=
  project (base m .output) (m .output).numel before
    (launch rest (config m) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x)) (destination w) other)

theorem execute_correct (rest : Algebra α) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (before : Nat → α) (other : State α) :
    RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => execute rest m w before other (addr (base m .output) x)) ∧
    (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      execute rest m w before other cell=before cell) := by
  have core := launch_correct rest (config m) (fun x => before (addr (base m .source) x))
    (fun x => before (addr (base m .output) x)) (destination w) other h.unique
  have frame : ∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      execute rest m w before other cell=before cell := by
    intro cell miss
    exact project_frame _ _ _ _ _ (fun x _ hx => core.2 x hx) cell miss
  refine ⟨⟨?_,?_⟩,frame⟩
  · intro p a b hp ha hb
    dsimp only [execute]
    rw [project_at _ _ _ _ _ (output_bound m w h p a b hp ha hb)]
    exact core.1 p a b hp ha hb
  · intro x miss
    apply frame
    rintro ⟨y,hy,he⟩
    have eq : x=y := by unfold addr at he; omega
    exact miss (eq ▸ hy)

def observeByte (memory : Memory) (byte : Nat) : BitVec 8 :=
  (memory (byte/4*4)).extractLsb' (byte%4*8) 8

theorem byte_frame (m : Inputs) (w : Nat → Word) (h : InputContract m w) (before after : Memory)
    (frame : ∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      after cell=before cell) (byte : Nat)
    (outside : ∀ x, touched (config m) (destination w) x →
      byte<addr (base m .output) x ∨ addr (base m .output) x+4≤byte) :
    observeByte after byte=observeByte before byte := by
  unfold observeByte
  rw [frame]
  rintro ⟨x,hx,he⟩
  obtain ⟨p,a,b,hp,ha,hb,rfl⟩ := hx
  have bound := output_bound m w h p a b hp ha hb
  have align := h.alignment .output (by omega)
  have ext := outside _ ⟨p,a,b,hp,ha,hb,rfl⟩
  unfold addr at he ext
  omega

#axiomsClean execute_correct
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1
