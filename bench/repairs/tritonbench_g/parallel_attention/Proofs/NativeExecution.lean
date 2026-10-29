import bench.repairs.tritonbench_g.parallel_attention.Proofs.NativeGraph
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Combined

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

def nativeScratchValue (a : Ops α) (c : CompilerLayout) (s : Shape) (kind : OutputKind)
    (x : TensorInputs α) (i : Nat) : α :=
  let D := kind.width s
  let e := i%D
  let t := i/D%s.T
  let bh := i/D/s.T%(s.B*s.H)
  let shard := i/(s.B*s.H*s.T*D)
  match kind with
  | .forward => (nativeForwardLane a c s x ⟨bh,t/128,0,e/s.BV⟩ (t%128) (e%s.BV)).1
  | .normalizer => (nativeForwardLane a c s x ⟨bh,t/128,0,0⟩ (t%128) 0).2
  | .query => nativeQueryLane a c s x ⟨bh,t/64,0,shard⟩ (t%64) e
  | .key => (nativeKeyValueLane a c s x ⟨bh,t/64,0,shard⟩ (t%64) e 0).1
  | .value => (nativeKeyValueLane a c s x ⟨bh,t/64,0,e/s.BV⟩ (t%64) 0 (e%s.BV)).2

def nativeWordReference (a : Ops α) (c : CompilerLayout) (s : Shape) (kind : OutputKind)
    (tree : SumPlan) (x : TensorInputs α) (i : Nat) :=
  a.cast (tree.eval a (fun shard => if shard<kind.shards s ∧ i<kind.cells s then
    nativeScratchValue a c s kind x (shard*kind.cells s+i) else a.zero))

def compiledScratch (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Event) (m : Memory α) : Memory α :=
  runTagged (tensorAddress bases)
    (fun mem e => nativeScratchValue a c s e.1 (currentInputs a s b mem) e.2) schedule m

def compiledRun (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (ks hs : List Event) (m : Memory α) :=
  combinedHost a s trees scratch output hs (compiledScratch a c s b scratch ks m)

theorem native_scratch_local (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases) (kind : OutputKind)
    (m m' : Memory α) (same : Set.EqOn m m' (inputCells s b)) (i : Nat) :
    nativeScratchValue a c s kind (currentInputs a s b m) i=
      nativeScratchValue a c s kind (currentInputs a s b m') i := by
  rw [currentInputs_congr a s b m m' same]

theorem compiled_scratch_output (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (bases : OutputKind → Nat) (schedule : List Event) (m : Memory α)
    (fresh : ∀ e∈schedule,tensorAddress bases e∉inputCells s b)
    (unique : ∀ e∈schedule,∀ f∈schedule,tensorAddress bases e=tensorAddress bases f → e=f)
    (target : Event) (member : target∈schedule) :
    compiledScratch a c s b bases schedule m (tensorAddress bases target)=
      nativeScratchValue a c s target.1 (currentInputs a s b m) target.2 := by
  exact tagged_output (inputCells s b) (tensorAddress bases) _
    (by intro m m' same e;exact native_scratch_local a c s b e.1 m m' same e.2)
    schedule m fresh unique target member

/-- All scratch writes precede all host reductions, as in the actual wrapper.
The kernel schedule may interleave distinct tensor stores arbitrarily. -/
theorem compiled_output (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat) (kinds : List OutputKind)
    (ks hs : List Event) (kernelCoverage : ks.Perm (events s true kinds))
    (hostCoverage : hs.Perm (events s false kinds))
    (kernelFresh : ∀ e∈ks,tensorAddress scratch e∉inputCells s b)
    (kernelUnique : ∀ e∈ks,∀ f∈ks,tensorAddress scratch e=tensorAddress scratch f → e=f)
    (hostFresh : ∀ e∈hs,tensorAddress output e∉allScratchCells s scratch kinds)
    (hostUnique : ∀ e∈hs,∀ f∈hs,tensorAddress output e=tensorAddress output f → e=f)
    (m : Memory α) (kind : OutputKind) (hkind : kind∈kinds) (i : Nat) (hi : i < kind.cells s) :
    compiledRun a c s b trees scratch output ks hs m (output kind+i)=
      nativeWordReference a c s kind (trees kind) (currentInputs a s b m) i := by
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
  unfold compiledRun combinedHost
  rw [sameRun hs (by intro e he;exact he)]
  have result := tagged_output (allScratchCells s scratch kinds) (tensorAddress output) value
    (by intro m m' same e;dsimp [value];split_ifs with he;exact localHost m m' same e he;rfl)
    hs (compiledScratch a c s b scratch ks m) hostFresh hostUnique (kind,i) hsMember
  change _=nativeWordReference a c s kind (trees kind) (currentInputs a s b m) i
  simp only [tensorAddress,Prod.fst,Prod.snd] at result
  rw [result]
  dsimp [value]
  rw [if_pos hsMember]
  unfold hostValue nativeWordReference
  congr 2
  funext shard
  split_ifs with h
  · have hb : shard*kind.cells s+i < kind.count s := by
      change shard*kind.cells s+i < kind.shards s*kind.cells s
      nlinarith [h.1,h.2]
    have member : (kind,shard*kind.cells s+i)∈ks :=
      kernelCoverage.mem_iff.mpr ((events_member _ _ _ _ _).mpr ⟨hkind,hb⟩)
    simpa [tensorAddress,Nat.add_assoc] using
      compiled_scratch_output a c s b scratch ks m kernelFresh kernelUnique _ member
  · rfl

theorem compiled_frame (a : Ops α) (c : CompilerLayout) (s : Shape) (b : Bases)
    (trees : OutputKind → SumPlan) (scratch output : OutputKind → Nat)
    (ks hs : List Event) (m : Memory α) (address : Nat)
    (outsideKernel : ∀ e∈ks,tensorAddress scratch e ≠ address)
    (outsideHost : ∀ e∈hs,tensorAddress output e ≠ address) :
    compiledRun a c s b trees scratch output ks hs m address=m address := by
  unfold compiledRun combinedHost compiledScratch
  rw [tagged_frame _ _ _ _ _ outsideHost,tagged_frame _ _ _ _ _ outsideKernel]



end
end VeriTile.Bench.Repairs.ParallelAttention
