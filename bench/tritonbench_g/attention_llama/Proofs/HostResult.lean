import bench.tritonbench_g.attention_llama.Proofs.FP8Kernel
import bench.tritonbench_g.attention_llama.Proofs.HostGrid

/- The source launcher selects q.device through the standard CUDA device
context. Formal device/stream observations refer to that effective owning
device; leaving the context restores the ambient caller device. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.HostResult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open LoopInvariant HostGrid HostMemory MaskCoverage
open scoped BigOperators
set_option maxHeartbeats 1000000

def contextFor {B heads : Nat} (c : Context) (i : Index B heads c.H c.D) : Context :=
  { c with initial := { c.initial with pids := fun ax =>
    if ax = 0 then i.query.val / 64 else
    if ax = 1 then i.batch.val * heads + i.head.val else 0 } }

def lane {B heads : Nat} {c : Context} (i : Index B heads c.H c.D) : Fin 64 :=
  ⟨i.query.val % 64, Nat.mod_lt _ (by decide)⟩

theorem row_context {B heads : Nat} (c : Context) (i : Index B heads c.H c.D) :
    (contextFor c i).row (lane i) = i.query.val := by
  simp only [Context.row,contextFor,lane,↓reduceIte]
  omega

structure Inputs (B heads H D : Nat) where
  query : Fin B → Fin heads → Fin H → Fin D → ℝ
  key : Fin B → Fin heads → Nat → Fin D → ℝ
  value : Fin B → Fin heads → Nat → Fin D → ℝ

noncomputable def score {B heads H D : Nat} (x : Inputs B heads H D)
    (scale : ℝ) (i : Index B heads H D) (k : Nat) : ℝ :=
  (∑ d : Fin D, x.query i.batch i.head i.query d * x.key i.batch i.head k d) * scale

noncomputable def expected {B heads : Nat} (c : Context) (x : Inputs B heads c.H c.D)
    (i : Index B heads c.H c.D) : ℝ :=
  TailMath.expected (score x c.scale i) (fun k => x.value i.batch i.head k i.channel)
    (logicalKeys c.N c.causal c.SP i.query.val)

/-- Ordinary tensor coordinate bindings. Inputs may share storage: no
disjointness between Q, K and V is required by these coordinate equations. -/
structure Binding {B heads : Nat} (c : Context) (x : Inputs B heads c.H c.D)
    : Prop where
  query : ∀ i d, (contextFor c i).query (lane i,d,PUnit.unit) =
    x.query i.batch i.head i.query d
  key : ∀ i k d, k < c.N → (contextFor c i).key k d = x.key i.batch i.head k d
  value : ∀ i k d, k < c.N → (contextFor c i).value k d = x.value i.batch i.head k d

theorem target_binding {B heads : Nat} (c : Context) (x : Inputs B heads c.H c.D)
    (h : Binding c x) (i : Index B heads c.H c.D) :
    RealKernel.target (contextFor c i) (lane i,i.channel,PUnit.unit) = expected c x i := by
  have hs : ∀ k, k < c.N → (contextFor c i).score (lane i) k = score x c.scale i k := by
    intro k hk
    simp only [Context.score,score]
    congr 1
    apply Finset.sum_congr rfl
    intro d _
    rw [h.query i d,h.key i k d hk]
  unfold RealKernel.target expected
  rw [row_context]
  unfold TailMath.expected TailMath.moment TailMath.mass
  congr 1
  · apply Finset.sum_congr rfl
    intro k hk
    have hn : k < c.N := (Finset.mem_filter.mp hk).1 |> Finset.mem_range.mp
    dsimp only
    rw [hs k hn,h.value i k i.channel hn]
  · apply Finset.sum_congr rfl
    intro k hk
    have hn : k < c.N := (Finset.mem_filter.mp hk).1 |> Finset.mem_range.mp
    rw [hs k hn]

def kernel (c : Context) (Out : RegionName) (soz soh som son : Nat) : ComputeKernel :=
  TypedSource.repaired c.Q c.K c.V Out c.scale c.sqz c.sqh c.sqm c.sqk
    c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn soz soh som son
    c.NH c.H c.N c.SP 64 64 c.D c.causal

/-- A runtime result names actual executions of the source. Existence is proved
below; this interface contains no expected-output equation. -/
structure Results {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat) where
  state : Index B heads c.H c.D → BlockState
  runs : ∀ i, exec (kernel c Out soz soh som son).eraseDType.toAlgKernel
    (contextFor c i).initial = some (state i)

theorem results_exist {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP = 0)
    (hout : ∀ i : Index B heads c.H c.D,
      Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D som son)
    (hu : ∀ rg o, c.initial.undef rg o = 0) :
    Nonempty (Results (B := B) (heads := heads) c Out soz soh som son) := by
  have hex : ∀ i : Index B heads c.H c.D, ∃ sf,
      exec (kernel c Out soz soh som son).eraseDType.toAlgKernel (contextFor c i).initial = some sf := by
    intro i
    obtain ⟨sf,hr,_,_⟩ := RealKernel.execution (contextFor c i) Out soz soh som son hD hN
      (by intro hc; simp only [contextFor] at hc ⊢; rw [hSP hc]; omega) (hout i) hu
    exact ⟨sf,hr⟩
  choose states runs using hex
  exact ⟨⟨states,runs⟩⟩

noncomputable def word {B heads : Nat} {c : Context} {Out : RegionName} {soz soh som son : Nat}
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (i : Index B heads c.H c.D) : ℝ :=
  (result.state i).readMem Out
    (alOutOffset (contextFor c i).initial c.NH soz soh som son 64 (lane i,i.channel,PUnit.unit))

theorem word_correct {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (x : Inputs B heads c.H c.D) (binding : Binding c x)
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hout : ∀ i : Index B heads c.H c.D,
      Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D som son)
    (hu : ∀ rg o, c.initial.undef rg o = 0) (i : Index B heads c.H c.D) :
    word result i = expected c x i := by
  obtain ⟨sf,hr,hread,_⟩ := RealKernel.execution (contextFor c i) Out soz soh som son hD hN
    (by intro hc; simp only [contextFor] at hc ⊢; rw [hSP hc]; omega) (hout i) hu
  have he : sf = result.state i := Option.some.inj (hr.symm.trans (result.runs i))
  subst sf
  have ha : alActive (contextFor c i).initial c.H 64 (lane i,i.channel,PUnit.unit) := by
    change (contextFor c i).row (lane i) < c.H
    rw [row_context]
    exact i.query.isLt
  exact (hread _ ha).trans (target_binding c x binding i)

/-- Source-produced words are committed at the fresh tensor's addresses in
the actual two-dimensional grid/lane schedule. Readback and the complementary
flat-memory frame follow from these stores and the allocator's injectivity. -/
theorem returned_values_and_frame {B heads : Nat} (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (x : Inputs B heads c.H c.D) (binding : Binding c x)
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hout : ∀ i : Index B heads c.H c.D,
      Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D som son)
    (hu : ∀ rg o, c.initial.undef rg o = 0)
    (address : Index B heads c.H c.D → Nat) (hinj : Function.Injective address)
    (memory : Nat → ℝ) :
    (∀ i, (returnTensor address (storeAll address (word result) memory
      (schedule B heads c.H c.D))).read i = expected c x i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (word result) memory (schedule B heads c.H c.D) location = memory location) := by
  obtain ⟨hv,hf⟩ := logical_result_and_frame address (word result) hinj memory
  exact ⟨fun i => (hv i).trans
    (word_correct c Out soz soh som son x binding result hD hN hSP hout hu i),hf⟩

end VeriTile.Bench.Repairs.AttentionLlama.HostResult
