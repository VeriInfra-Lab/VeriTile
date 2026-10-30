import bench.tritonbench_g.triton_argmax.Proofs.Formats
import bench.tritonbench_g.triton_argmax.Proofs.RealMemory

/-! Explicit Real scratch replay: zero initialized index padding, actual
ordered stage-one trees, intermediate stores, second-stage loads/tree/gather,
and final output store. Input loads use the immutable prepared input snapshot;
the full scratch-write frame is proved separately below. This is a source
projection helper, not yet the final same-request wrapper theorem. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.RealScratch
open Source Memory Reduction
noncomputable section

def write (heap : Address → α) (point : Address) (value : α) : Address → α :=
  fun other => if other=point then value else heap other

def stores (allocation : Allocation) (width : Nat) (values : Nat → α) (jobs : List Nat)
    (heap : Address → α) : Address → α :=
  jobs.foldl (fun current job => write current (address allocation width job) (values job)) heap

theorem address_injective (allocation : Allocation) (width : Nat) (positive : 0 < width)
    (i j : Nat) : address allocation width i=address allocation width j ↔ i=j := by
  constructor
  · intro equal
    have equal := congrArg Prod.snd equal
    simp only [address] at equal
    exact Nat.eq_of_mul_eq_mul_left positive (Nat.add_left_cancel equal)
  · intro equal
    rw [equal]

theorem stores_at (allocation : Allocation) (width : Nat) (positive : 0 < width)
    (values : Nat → α) (jobs : List Nat) (heap : Address → α) (i : Nat) :
    stores allocation width values jobs heap (address allocation width i)=
      if i∈jobs then values i else heap (address allocation width i) := by
  induction jobs generalizing heap with
  | nil => simp [stores]
  | cons head tail ih =>
    rw [stores,List.foldl_cons]
    change stores allocation width values tail (write heap (address allocation width head) (values head)) _=_
    rw [ih]
    by_cases member : i∈tail
    · simp [member]
    · by_cases same : i=head
      · subst i
        simp [member,write]
      · simp [member,same,write,(address_injective allocation width positive i head).not.mpr same]

theorem stores_frame (allocation : Allocation) (width : Nat) (values : Nat → α) (jobs : List Nat)
    (heap : Address → α) (point : Address)
    (outside : ∀ i,i∈jobs → point≠address allocation width i) :
    stores allocation width values jobs heap point=heap point := by
  induction jobs generalizing heap with
  | nil => rfl
  | cons head tail ih =>
    rw [stores,List.foldl_cons]
    change stores allocation width values tail (write heap (address allocation width head) (values head)) point=_
    rw [ih _ (fun i hi => outside i (by simp [hi]))]
    simp only [write,if_neg (outside head (by simp))]

structure State where
  values : Address → Math.Value
  indices : Address → Nat

/-- The projection explicitly erases the precision adapter as a macro. For
integer/bool this means the complete NaN0/clamp/truncate/store/load adapter;
it does not assert that a literal finite clamp is identity on arbitrary Reals.
The exact native adapter graph is retained by Formats and its scalar binding. -/
inductive FloatingKind where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq,Repr

def FloatingKind.format : FloatingKind → MidFormat
  | .fp32 => ⟨"torch.float32",4,by decide⟩
  | .fp16 => ⟨"torch.float16",2,by decide⟩
  | .bf16 => ⟨"torch.bfloat16",2,by decide⟩
  | .fp64 => ⟨"torch.float64",8,by decide⟩

/-- Generic direct-cast proof descriptor. The executable checked adapter
selects exactly the four FloatingKind values; the descriptor itself remains
generic so the scratch and tree lemmas do not depend on a storage width. -/
structure DirectFormat where
  format : MidFormat
  nonInteger : ∀ kind : Formats.IntegerKind,format.name≠kind.dtypeName

def FloatingKind.direct (kind : FloatingKind) : DirectFormat where
  format:=kind.format
  nonInteger:=by intro integer; cases kind <;> cases integer <;> decide

inductive Adapter where
  | directCast : DirectFormat → Adapter
  | integerSafe : Formats.IntegerKind → Adapter

def Adapter.format : Adapter → MidFormat
  | .directCast descriptor => descriptor.format
  | .integerSafe kind => Formats.format kind

theorem direct_not_integer (direct : DirectFormat) (integer : Formats.IntegerKind) :
    direct.format≠Formats.format integer := by
  intro same
  exact direct.nonInteger integer (congrArg MidFormat.name same)

def Adapter.erase (_ : Adapter) (value : Math.Value) : Math.Value := value

def firstValue (adapter : Adapter) (c : GlobalConfig) (logical : Nat → ℝ) (pid : Nat) : Math.Value :=
  adapter.erase (Math.padded c.size (pid*c.block) c.block logical
    (Math.blockSelected c.size (pid*c.block) c.block logical (c.firstTree pid)))

theorem first_value (adapter : Adapter) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (logical : Nat → ℝ) (pid : Nat) (active : pid < c.blocks) :
    firstValue adapter c logical pid=(logical (Math.firstIndex c logical pid) : Math.Value) := by
  have valid := Math.block_valid c.size (pid*c.block) c.block logical (c.firstTree pid)
    (coverage.firstComplete pid active) (coverage.eachActive pid active) coverage.blockPositive
  simp only [firstValue,Adapter.erase,Math.padded,if_pos valid,Math.firstIndex]

def zeroIndices (c : GlobalConfig) (p : GlobalMemory.Buffers) (state : State) : State :=
  {state with indices:=stores p.midIndex 8 (fun _ => 0) (List.range c.midBlock) state.indices}

def firstGrid (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) : State :=
  ⟨stores p.midValue adapter.format.bytes (firstValue adapter c logical) (List.range c.blocks) state.values,
   stores p.midIndex 8 (Math.firstIndex c logical) (List.range c.blocks) state.indices⟩

def firstJob (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) (pid : Nat) : State :=
  ⟨write state.values (address p.midValue adapter.format.bytes pid) (firstValue adapter c logical pid),
   write state.indices (address p.midIndex 8 pid) (Math.firstIndex c logical pid)⟩

theorem separate_stores (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (jobs : List Nat) (state : State) :
    jobs.foldl (firstJob adapter c p logical) state=
      ⟨stores p.midValue adapter.format.bytes (firstValue adapter c logical) jobs state.values,
       stores p.midIndex 8 (Math.firstIndex c logical) jobs state.indices⟩ := by
  induction jobs generalizing state with
  | nil => rfl
  | cons head tail ih =>
    simp only [List.foldl_cons,ih,stores,firstJob]

theorem firstGrid_loop (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) :
    (List.range c.blocks).foldl (firstJob adapter c p logical) state=firstGrid adapter c p logical state :=
  separate_stores adapter c p logical _ state

def secondLoaded (c : GlobalConfig) (p : GlobalMemory.Buffers) (adapter : Adapter)
    (state : State) (lane : Fin c.midBlock) : Math.Value :=
  if lane.val < c.blocks then state.values (address p.midValue adapter.format.bytes lane.val) else ⊥

def second (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers) (state : State) : State :=
  let selected := c.secondTree.pick orderedGt orderedEq (secondLoaded c p adapter state)
  let index := state.indices (address p.midIndex 8 selected.val)
  {state with indices:=write state.indices (address p.output 8 0) index}

def run (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) : State :=
  second adapter c p (firstGrid adapter c p logical (zeroIndices c p state))

theorem first_values (adapter : Adapter) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (p : GlobalMemory.Buffers) (logical : Nat → ℝ) (state : State) :
    secondLoaded c p adapter (firstGrid adapter c p logical state)=Math.midValues c logical := by
  funext lane
  by_cases active : lane.val < c.blocks
  · simp only [secondLoaded,firstGrid,stores_at _ _ adapter.format.positive,
      List.mem_range,if_pos active,first_value adapter c coverage logical lane.val active,Math.midValues]
  · simp [secondLoaded,Math.midValues,active]

theorem initialized_padding (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) (i : Nat) (padding : c.blocks ≤ i) (bound : i < c.midBlock) :
    (firstGrid adapter c p logical (zeroIndices c p state)).indices (address p.midIndex 8 i)=0 := by
  simp [firstGrid,zeroIndices,stores_at _ _ (by omega : 0 < 8),List.mem_range,
    show ¬i < c.blocks by omega,bound]

theorem gathered_index (adapter : Adapter) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (p : GlobalMemory.Buffers) (logical : Nat → ℝ) (state : State) :
    (run adapter c p logical state).indices (address p.output 8 0)=Math.globalIndex c logical := by
  have valid := Math.mid_valid c coverage logical
  unfold run second
  rw [first_values adapter c coverage]
  simp only [write,ite_true]
  change stores p.midIndex 8 (Math.firstIndex c logical) (List.range c.blocks) _ _=_
  rw [stores_at _ _ (by omega)]
  simp only [List.mem_range]
  change (if (Math.globalSelected c logical).val < c.blocks then _ else _)=_
  rw [if_pos valid]
  rfl

/-- Every address outside actual scratch/output stores preserves both shadow
channels, including all caller padding. The local prepare frame composes with
this source replay frame; no whole frame is a premise. -/
theorem frame (adapter : Adapter) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (logical : Nat → ℝ) (state : State) (point : Address)
    (midValue : ∀ i,i < c.blocks → point≠address p.midValue adapter.format.bytes i)
    (midIndex : ∀ i,i < c.midBlock → point≠address p.midIndex 8 i)
    (fits : c.blocks ≤ c.midBlock) (output : point≠address p.output 8 0) :
    (run adapter c p logical state).values point=state.values point ∧
      (run adapter c p logical state).indices point=state.indices point := by
  constructor
  · exact stores_frame _ _ _ _ _ _ (fun i hi => midValue i (List.mem_range.mp hi))
  · unfold run second
    simp only [write,if_neg output]
    change stores p.midIndex 8 (Math.firstIndex c logical) (List.range c.blocks)
      (stores p.midIndex 8 (fun _ => 0) (List.range c.midBlock) state.indices) point=_
    rw [stores_frame _ _ _ _ _ _ (fun i hi => midIndex i (lt_of_lt_of_le (List.mem_range.mp hi) fits)),
      stores_frame _ _ _ _ _ _ (fun i hi => midIndex i (List.mem_range.mp hi))]

#axiomsClean stores_at
#axiomsClean direct_not_integer
#axiomsClean stores_frame
#axiomsClean first_value
#axiomsClean firstGrid_loop
#axiomsClean first_values
#axiomsClean initialized_padding
#axiomsClean gathered_index
#axiomsClean frame
end
end VeriTile.Bench.Repairs.TritonArgmax.RealScratch
