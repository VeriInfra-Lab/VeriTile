import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPEpilogue
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPViews
import bench.tritonbench_g.context_attn_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.ContextAttnLlama.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def atIndex {g : Grid} (s : State α) (idx : GridIndex g) : State α :=
  {s with pids := idx.toPids, numPids := g.toNumPids}

def runGrid (c : Host.Call) (M : Algebra α) :
    List (GridIndex c.grid) → State α → Option (State α)
  | [], s => some s
  | idx :: rest, s => do
    let t ← exec (M.withDefaultPrecision .fp32)
      (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)) (atIndex s idx)
    runGrid c M rest t

/-- Every program in the actual finite grid executes. No whole-program success
premise, arithmetic exactness or output equation is assumed. -/
theorem finite_grid_success (c : Host.Call) (M : Algebra α)
    (hM : DotSupport c.layout M) (order : List (GridIndex c.grid)) (s : State α) :
    ∃ t, runGrid c M order s = some t := by
  have hBN : 0 < c.layout.BN := by rw [c.blockN]; exact Host.bm_positive c
  induction order generalizing s with
  | nil => exact ⟨s,rfl⟩
  | cons idx rest ih =>
    obtain ⟨v,t,ht,_⟩ := program_success c.layout c.regions (Host.scale c.layout.D)
      M hM hBN (atIndex s idx)
    obtain ⟨u,hu⟩ := ih t
    exact ⟨u,by simp only [runGrid,ht,Option.bind_some]; exact hu⟩

def allPrograms (c : Host.Call) : List (GridIndex c.grid) := Finset.univ.toList

theorem allPrograms_complete (c : Host.Call) (idx : GridIndex c.grid) :
    idx ∈ allPrograms c := by simp [allPrograms]

def Metadata (c : Host.Call) (s : State α) : Prop :=
  (∀ b, b < c.batches → (s.mem c.regions.start b).read .int = (c.start b : Int)) ∧
  (∀ b, b < c.batches → (s.mem c.regions.sequence b).read .int = ((c.length b+c.prompt b : Nat) : Int)) ∧
  (∀ b, b < c.batches → (s.mem c.regions.prompt b).read .int = (c.prompt b : Int))

def writes (c : Host.Call) (order : List (GridIndex c.grid)) (rg : RegionName) (o : Nat) : Prop :=
  ∃ idx ∈ order, ∃ i, c.active idx i ∧ rg = c.regions.out ∧ o = c.offset idx i


theorem active_matches (c : Host.Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    active c.layout c.regions (atIndex s idx) i ↔ c.active idx i := by
  have hb := Host.batch_bound c idx
  change idx.toPids 1 / c.layout.heads < c.batches at hb
  change ((c.layout.BM * idx.toPids 0 + i.1.val : Nat) : Int) <
    (s.mem c.regions.sequence (idx.toPids 1 / c.layout.heads)).read .int -
    (s.mem c.regions.prompt (idx.toPids 1 / c.layout.heads)).read .int ↔ _
  rw [hm.2.1 _ hb,hm.2.2 _ hb]
  simp only [Host.Call.active,Formula.queryRow,Formula.batch,Host.Call.launchAt]
  rw [Nat.mul_comm c.layout.BM (idx.toPids 0)]
  omega

theorem offset_matches (c : Host.Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    offset c.layout c.regions (atIndex s idx) i = c.offset idx i := by
  have hb := Host.batch_bound c idx
  change idx.toPids 1 / c.layout.heads < c.batches at hb
  unfold offset start row atIndex
  simp only
  rw [hm.1 _ hb]
  simp only [Host.Call.offset,Host.Call.output,Formula.queryRow,Formula.batch,
    Formula.head,Host.Call.launchAt]
  norm_cast
  simp [Nat.mul_comm]
  rfl

theorem program_call (c : Host.Call) (M : Algebra α) (hM : DotSupport c.layout M)
    (s : State α) (hm : Metadata c s) (idx : GridIndex c.grid) :
    ∃ t, exec (M.withDefaultPrecision .fp32)
      (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)) (atIndex s idx) = some t ∧
      ∀ rg o, (¬ ∃ i, c.active idx i ∧ rg = c.regions.out ∧ o = c.offset idx i) →
        t.mem rg o = s.mem rg o := by
  have hBN : 0 < c.layout.BN := by rw [c.blockN]; exact Host.bm_positive c
  obtain ⟨v,t,ht,hmt⟩ := program_success c.layout c.regions (Host.scale c.layout.D)
    M hM hBN (atIndex s idx)
  refine ⟨t,ht,?_⟩
  intro rg o hmiss
  rw [hmt]
  apply scatter_frame
  intro i hi hr ho
  apply hmiss
  exact ⟨i,(active_matches c s hm idx i).mp hi,hr,ho.trans (offset_matches c s hm idx i)⟩

/-- The metadata is immutable during the private launch. The caller's output
may alias any original input: only the fresh scratch region must be separate. -/
def Separate (c : Host.Call) : Prop :=
  ∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,
    c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.pages] → c.regions.out ≠ rg

theorem grid_success_frame (c : Host.Call) (M : Algebra α) (hM : DotSupport c.layout M)
    (sep : Separate c) (order : List (GridIndex c.grid))
    (s : State α) (hm : Metadata c s) :
    ∃ t, runGrid c M order s = some t ∧ Metadata c t ∧
      ∀ rg o, ¬ writes c order rg o → t.mem rg o = s.mem rg o := by
  induction order generalizing s with
  | nil =>
    exact ⟨s,rfl,hm,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨t,ht,hframe⟩ := program_call c M hM s hm idx
    have hregion : ∀ rg, c.regions.out ≠ rg → ∀ o, t.mem rg o = s.mem rg o := by
      intro rg hr o
      apply hframe
      rintro ⟨i,hi,he,ho⟩
      exact hr he.symm
    have hmeta : Metadata c t := by
      refine ⟨?_,?_,?_⟩
      · intro b hb
        rw [hregion c.regions.start (sep _ (by simp)) b]
        exact hm.1 b hb
      · intro b hb
        rw [hregion c.regions.sequence (sep _ (by simp)) b]
        exact hm.2.1 b hb
      · intro b hb
        rw [hregion c.regions.prompt (sep _ (by simp)) b]
        exact hm.2.2 b hb
    obtain ⟨u,hu,hmu,hfu⟩ := ih t hmeta
    refine ⟨u,?_,hmu,?_⟩
    · simp only [runGrid,ht,Option.bind_some]
      exact hu
    · intro rg o hmiss
      rw [hfu rg o (by
        rintro ⟨j,hj,i,hi,hr,ho⟩
        exact hmiss ⟨j,List.mem_cons_of_mem idx hj,i,hi,hr,ho⟩)]
      apply hframe
      rintro ⟨i,hi,hr,ho⟩
      exact hmiss ⟨idx,List.mem_cons_self,i,hi,hr,ho⟩

end
end VeriTile.Bench.Repairs.ContextAttnLlama.FPSuccess
