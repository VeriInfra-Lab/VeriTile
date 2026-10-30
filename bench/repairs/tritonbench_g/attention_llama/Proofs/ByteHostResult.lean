import bench.repairs.tritonbench_g.attention_llama.Proofs.HostResult

namespace VeriTile.Bench.Repairs.AttentionLlama.ByteHostResult
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open FP8RealMemory
open FP8RealLoop (BytesFrom)
open LoopInvariant HostGrid HostMemory MaskCoverage HostResult
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

def kernel (c : Context) (Out : RegionName) (soz soh som son : Nat) : ComputeKernel :=
  TypedSource.repairedFP8 c.Q c.K c.V Out c.scale c.sqz c.sqh c.sqm c.sqk
    c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn soz soh som son
    c.NH c.H c.N c.SP 64 64 c.D c.causal

structure Domain {B heads : Nat} (c : Context) : Prop where
  keys : ∀ i : Index B heads c.H c.D,
    BytesFrom c.K (contextFor c i).initial (alBase (contextFor c i).initial c.NH c.skz c.skh)
      c.skn c.skk 0 (FP8RealSetup.stop (contextFor c i).initial c.N c.SP 64 c.causal) c.N 64 c.D
  values : ∀ i : Index B heads c.H c.D,
    BytesFrom c.V (contextFor c i).initial (alBase (contextFor c i).initial c.NH c.svz c.svh)
      c.svk c.svn 0 (FP8RealSetup.stop (contextFor c i).initial c.N c.SP 64 c.causal) c.N 64 c.D

structure Results {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat) where
  state : Index B heads c.H c.D → BlockState
  runs : ∀ i, exec (kernel c Out soz soh som son).eraseDType.toAlgKernel
    (contextFor c i).initial = some (state i)

theorem results_exist {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (domain : Domain (B := B) (heads := heads) c)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hout : ∀ i : Index B heads c.H c.D,
      Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D som son)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hu : ∀ rg o, c.initial.undef rg o = 0) :
    Nonempty (Results (B := B) (heads := heads) c Out soz soh som son) := by
  have hex : ∀ i : Index B heads c.H c.D, ∃ sf,
      exec (kernel c Out soz soh som son).eraseDType.toAlgKernel (contextFor c i).initial = some sf := by
    intro i
    obtain ⟨sf,hr,_,_⟩ := FP8Kernel.execution (contextFor c i) Out soz soh som son hD hN
      (by intro hc; simp only [contextFor] at hc ⊢; rw [hSP hc]; omega)
      (hout i) hQK hQV hOK hOV hu (domain.keys i) (domain.values i)
    exact ⟨sf,hr⟩
  choose states runs using hex
  exact ⟨⟨states,runs⟩⟩

/-- The decoded view changes only K/V memory; program IDs and output words
are preserved. This derives the real execution interface from byte execution. -/
noncomputable def realResults {B heads : Nat} (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (domain : Domain (B := B) (heads := heads) c)
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hu : ∀ rg o, c.initial.undef rg o = 0) :
    HostResult.Results (B := B) (heads := heads) (FP8Kernel.view c) Out soz soh som son where
  state := fun i => decoded c.K c.V (result.state i)
  runs := by
    intro i
    have h := FP8Execution.decoded_execution c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal (contextFor c i).initial
      hQK hQV hOK hOV hu (domain.keys i) (domain.values i)
    change Option.map (decoded c.K c.V)
      (exec (kernel c Out soz soh som son).eraseDType.toAlgKernel (contextFor c i).initial) = _ at h
    rw [result.runs i,Option.map_some] at h
    exact h.symm

noncomputable def word {B heads : Nat} {c : Context} {Out : RegionName} {soz soh som son : Nat}
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (i : Index B heads c.H c.D) : ℝ :=
  (result.state i).readMem Out
    (alOutOffset (contextFor c i).initial c.NH soz soh som son 64 (lane i,i.channel,PUnit.unit))

theorem returned_values_and_frame {B heads : Nat} (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (x : Inputs B heads c.H c.D)
    (binding : Binding (FP8Kernel.view c) x)
    (domain : Domain (B := B) (heads := heads) c)
    (result : Results (B := B) (heads := heads) c Out soz soh som son)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hout : ∀ i : Index B heads c.H c.D,
      Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D som son)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hu : ∀ rg o, c.initial.undef rg o = 0)
    (address : Index B heads c.H c.D → Nat) (hinj : Function.Injective address)
    (memory : Nat → ℝ) :
    (∀ i, (returnTensor address (storeAll address (word result) memory
      (schedule B heads c.H c.D))).read i = expected c x i) ∧
    (∀ location, (∀ i, location ≠ address i) →
      storeAll address (word result) memory (schedule B heads c.H c.D) location = memory location) := by
  let real := realResults c Out soz soh som son domain result hQK hQV hOK hOV hu
  have he : word result = HostResult.word real := by
    funext i
    unfold word HostResult.word
    change (result.state i).readMem Out _ = (decoded c.K c.V (result.state i)).readMem Out _
    unfold BlockState.readMem
    rw [outside_memory c.K c.V (result.state i) Out _ hOK hOV]
    rfl
  rw [he]
  exact HostResult.returned_values_and_frame (FP8Kernel.view c) Out soz soh som son
    x binding real hD hN hSP hout hu address hinj memory

end VeriTile.Bench.Repairs.AttentionLlama.ByteHostResult
