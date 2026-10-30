import bench.tritonbench_g.parallel_attention.Proofs.Grid

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Tags distinguish writes of each tensor in the same actual kernel launch. -/
abbrev Event := OutputKind × Nat

def runTagged (addr : Event → Nat) (value : Memory α → Event → α) : List Event → Memory α → Memory α
  | [],m => m
  | e::es,m => runTagged addr value es (Function.update m (addr e) (value m e))

theorem tagged_frame (addr : Event → Nat) (value : Memory α → Event → α)
    (es : List Event) (m : Memory α) (i : Nat) (outside : ∀ e∈es,addr e ≠ i) :
    runTagged addr value es m i=m i := by
  induction es generalizing m with
  | nil => rfl
  | cons e es ih =>
    simp only [runTagged]
    rw [ih _ (by intro f hf;exact outside f (List.mem_cons_of_mem e hf))]
    exact Function.update_of_ne (Ne.symm (outside e (by simp))) _ _

theorem tagged_output (R : Set Nat) (addr : Event → Nat) (value : Memory α → Event → α)
    (locality : ∀ m m',Set.EqOn m m' R → ∀ e,value m e=value m' e)
    (es : List Event) (initial : Memory α) (fresh : ∀ e∈es,addr e∉R)
    (unique : ∀ e∈es,∀ f∈es,addr e=addr f → e=f)
    (target : Event) (member : target∈es) :
    runTagged addr value es initial (addr target)=value initial target := by
  have stable (m : Memory α) (same : Set.EqOn m initial R) :
      runTagged addr value es m (addr target)=value initial target := by
    induction es generalizing m with
    | nil => simp at member
    | cons e es ih =>
      simp only [runTagged]
      by_cases later : target∈es
      · apply ih (by intro f hf;exact fresh f (List.mem_cons_of_mem e hf))
          (by intro f hf g hg;exact unique f (List.mem_cons_of_mem e hf) g (List.mem_cons_of_mem e hg)) later
        intro i hi
        rw [Function.update_of_ne (by intro eq;exact fresh e (by simp) (eq ▸ hi))]
        exact same hi
      · have eq : target=e := by simpa [later] using member
        subst e
        rw [tagged_frame addr value es _ (addr target) (by
          intro f hf eq
          exact later (unique f (List.mem_cons_of_mem target hf) target (by simp) eq ▸ hf))]
        simpa using locality m initial same target
  exact stable initial (by intro i hi;rfl)

def events (s : Shape) (scratch : Bool) (kinds : List OutputKind) : List Event :=
  kinds.flatMap (fun kind => (List.range (if scratch then kind.count s else kind.cells s)).map (fun i => (kind,i)))

def tensorAddress (bases : OutputKind → Nat) (e : Event) : Nat := bases e.1+e.2

def combinedScratch (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Event) (m : Memory α) : Memory α :=
  runTagged (tensorAddress bases) (fun mem e => scratchCurrent a p s b e.1 mem e.2) schedule m

def allScratchCells (s : Shape) (bases : OutputKind → Nat) (kinds : List OutputKind) : Set Nat :=
  {address | ∃ kind∈kinds,address∈scratchCells (kind.shards s) (bases kind) (kind.cells s)}

def combinedHost (a : Ops α) (s : Shape) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (schedule : List Event) (m : Memory α) : Memory α :=
  runTagged (tensorAddress output)
    (fun mem e => hostValue a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) mem e.2) schedule m

def combinedRun (a : Ops α) (p : Plans) (s : Shape) (b : Bases) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (kernelSchedule hostSchedule : List Event) (m : Memory α) :=
  combinedHost a s trees scratch output hostSchedule (combinedScratch a p s b scratch kernelSchedule m)

theorem events_member (s : Shape) (scratch : Bool) (kinds : List OutputKind) (kind : OutputKind) (i : Nat) :
    (kind,i)∈events s scratch kinds ↔ kind∈kinds ∧ i < (if scratch then kind.count s else kind.cells s) := by
  simp [events]

theorem combined_scratch_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Event) (m : Memory α)
    (fresh : ∀ e∈schedule,tensorAddress bases e∉inputCells s b)
    (unique : ∀ e∈schedule,∀ f∈schedule,tensorAddress bases e=tensorAddress bases f → e=f)
    (target : Event) (member : target∈schedule) :
    combinedScratch a p s b bases schedule m (tensorAddress bases target)=
      scratchValue a p s target.1 (currentInputs a s b m) target.2 := by
  exact tagged_output (inputCells s b) (tensorAddress bases) _
    (by intro m m' same e;exact scratch_local a p s b e.1 m m' same e.2)
    schedule m fresh unique target member

/-- All scratch writes precede all host reductions, as in the actual wrapper.
The kernel schedule may interleave distinct tensor stores arbitrarily. -/
theorem combined_output (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (kernelCoverage : ks.Perm (events s true kinds))
    (hostCoverage : hs.Perm (events s false kinds))
    (kernelFresh : ∀ e∈ks,tensorAddress scratch e∉inputCells s b)
    (kernelUnique : ∀ e∈ks,∀ f∈ks,tensorAddress scratch e=tensorAddress scratch f → e=f)
    (hostFresh : ∀ e∈hs,tensorAddress output e∉allScratchCells s scratch kinds)
    (hostUnique : ∀ e∈hs,∀ f∈hs,tensorAddress output e=tensorAddress output f → e=f)
    (m : Memory α) (kind : OutputKind) (hkind : kind∈kinds) (i : Nat) (hi : i < kind.cells s) :
    combinedRun a p s b trees scratch output ks hs m (output kind+i)=
      wordReference a p s kind (trees kind) (currentInputs a s b m) i := by
  have hsMember : (kind,i)∈hs := hostCoverage.mem_iff.mpr ((events_member _ _ _ _ _).mpr ⟨hkind,hi⟩)
  have localHost : ∀ m m',Set.EqOn m m' (allScratchCells s scratch kinds) →
      ∀ e, e∈hs → hostValue a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) m e.2=
        hostValue a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) m' e.2 := by
    intro m m' same e he
    have hk := ((events_member _ _ _ _ _).mp (hostCoverage.mem_iff.mp he)).1
    apply host_local a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) m m'
    intro j hj
    exact same ⟨e.1,hk,hj⟩
  -- Give inactive tags zero, so the reader locality theorem has no hidden tag premise.
  let value := fun mem e => if e∈hs then hostValue a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) mem e.2 else a.zero
  have sameRun (es : List Event) (subset : ∀ e∈es,e∈hs) (mem : Memory α) :
      runTagged (tensorAddress output)
        (fun mem e => hostValue a (trees e.1) (e.1.shards s) (scratch e.1) (e.1.cells s) mem e.2) es mem=
      runTagged (tensorAddress output) value es mem := by
    induction es generalizing mem with
    | nil => rfl
    | cons e es ih =>
      simp only [runTagged,value,if_pos (subset e (by simp))]
      exact ih (by intro f hf;exact subset f (List.mem_cons_of_mem e hf)) _
  unfold combinedRun combinedHost
  rw [sameRun hs (by intro e he;exact he)]
  have result := tagged_output (allScratchCells s scratch kinds) (tensorAddress output) value
    (by intro m m' same e;dsimp [value];split_ifs with he;exact localHost m m' same e he;rfl)
    hs (combinedScratch a p s b scratch ks m) hostFresh hostUnique (kind,i) hsMember
  change _=wordReference a p s kind (trees kind) (currentInputs a s b m) i
  simp only [tensorAddress,Prod.fst,Prod.snd] at result
  rw [result]
  dsimp [value]
  rw [if_pos hsMember]
  unfold hostValue wordReference
  congr 2
  funext shard
  split_ifs with h
  · have hb : shard*kind.cells s+i < kind.count s := by
      change shard*kind.cells s+i < kind.shards s*kind.cells s
      nlinarith [h.1,h.2]
    have member : (kind,shard*kind.cells s+i)∈ks :=
      kernelCoverage.mem_iff.mpr ((events_member _ _ _ _ _).mpr ⟨hkind,hb⟩)
    simpa [tensorAddress,Nat.add_assoc] using
      combined_scratch_output a p s b scratch ks m kernelFresh kernelUnique _ member
  · rfl

theorem combined_frame (a : Ops α) (p : Plans) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (address : Nat)
    (outsideKernel : ∀ e∈ks,tensorAddress scratch e ≠ address)
    (outsideHost : ∀ e∈hs,tensorAddress output e ≠ address) :
    combinedRun a p s b trees scratch output ks hs m address=m address := by
  unfold combinedRun combinedHost combinedScratch
  rw [tagged_frame _ _ _ _ _ outsideHost,tagged_frame _ _ _ _ _ outsideKernel]

#axiomsClean tagged_output
#axiomsClean combined_scratch_output
#axiomsClean combined_output
#axiomsClean combined_frame
end
end VeriTile.Bench.Repairs.ParallelAttention
