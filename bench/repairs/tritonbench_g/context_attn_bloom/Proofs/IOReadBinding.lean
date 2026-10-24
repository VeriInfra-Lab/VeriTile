import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOGeometry
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOReferenceAgreement

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000
variable {α : Type} [Inhabited α]

theorem read_query (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.query) :
    (s.mem r.query off).read .real=payload r extent xs 0 off := by
  have h := hin (0 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.query off).read .real=xs (0 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [payload,dif_pos ho] using h

theorem read_key (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.key) :
    (s.mem r.key off).read .real=payload r extent xs 1 off := by
  have h := hin (1 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.key off).read .real=xs (1 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [payload,dif_pos ho] using h

theorem read_value (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.value) :
    (s.mem r.value off).read .real=payload r extent xs 2 off := by
  have h := hin (2 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.value off).read .real=xs (2 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [payload,dif_pos ho] using h

theorem read_start (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.start) :
    (s.mem r.start off).read .int=integer r extent xs 0 off := by
  have h := hin (3 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.start off).read .int=xs (3 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [integer,dif_pos ho] using h

theorem read_sequence (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.sequence) :
    (s.mem r.sequence off).read .int=integer r extent xs 1 off := by
  have h := hin (4 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.sequence off).read .int=xs (4 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [integer,dif_pos ho] using h

theorem read_prompt (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.prompt) :
    (s.mem r.prompt off).read .int=integer r extent xs 2 off := by
  have h := hin (5 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.prompt off).read .int=xs (5 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [integer,dif_pos ho] using h

theorem read_request (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.request) :
    (s.mem r.request off).read .int=integer r extent xs 3 off := by
  have h := hin (6 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.request off).read .int=xs (6 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [integer,dif_pos ho] using h

theorem read_tokens (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (off : Nat) (ho : off<extent r.tokens) :
    (s.mem r.tokens off).read .int=integer r extent xs 4 off := by
  have h := hin (7 : Fin 8) (⟨off,ho⟩,PUnit.unit) trivial
  change (s.mem r.tokens off).read .int=xs (7 : Fin 8) (⟨off,ho⟩,PUnit.unit) at h
  simpa only [integer,dif_pos ho] using h

theorem input_agreement (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (valid : domain g r extent xs) (s : State α)
    (hin : (io g r extent M dots).InputBinding xs s) (idx : GridIndex (Host.grid g.arguments)) :
    IOReferenceAgreement.Agree r extent
      (FPReference.inputs (Host.layout g.arguments) r ((invocation g r M dots idx).initial s))
      (logicalInputs g r extent xs idx) := by
  have hb := valid.bounds idx
  refine ⟨rfl,rfl,rfl,rfl,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · exact read_start g r extent M dots xs s hin _ hb.start
  · change (s.mem r.sequence (Host.launchOf g.arguments idx).batch).read .int-
      (s.mem r.prompt (Host.launchOf g.arguments idx).batch).read .int=_
    rw [read_sequence g r extent M dots xs s hin _ hb.sequence,
      read_prompt g r extent M dots xs s hin _ hb.prompt]
    rfl
  · exact read_prompt g r extent M dots xs s hin _ hb.prompt
  · exact read_request g r extent M dots xs s hin _ hb.request
  · exact read_query g r extent M dots xs s hin
  · exact read_key g r extent M dots xs s hin
  · exact read_value g r extent M dots xs s hin
  · exact read_tokens g r extent M dots xs s hin

/-- Every finite input cell declared by the public IO is backed by the
caller's actual regional allocation; repeated input labels are preserved. -/
theorem input_capacity (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (M : Algebra α) (dots : FPSuccess.Dot α) (xs : Values α r extent)
    (bounds : RegionBounds) (hb : (io g r extent M dots).InBounds bounds xs) :
    ∀ region, region∈[r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.request,r.tokens] →
      extent region≤bounds region := by
  intro region hr
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with h|h|h|h|h|h|h|h
  · subst region
    by_contra hn
    have ho : bounds r.query<extent r.query := by omega
    have impossible := hb.2.1 (0 : Fin 8) (⟨bounds r.query,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.key<extent r.key := by omega
    have impossible := hb.2.1 (1 : Fin 8) (⟨bounds r.key,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.value<extent r.value := by omega
    have impossible := hb.2.1 (2 : Fin 8) (⟨bounds r.value,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.start<extent r.start := by omega
    have impossible := hb.2.1 (3 : Fin 8) (⟨bounds r.start,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.sequence<extent r.sequence := by omega
    have impossible := hb.2.1 (4 : Fin 8) (⟨bounds r.sequence,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.prompt<extent r.prompt := by omega
    have impossible := hb.2.1 (5 : Fin 8) (⟨bounds r.prompt,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.request<extent r.request := by omega
    have impossible := hb.2.1 (6 : Fin 8) (⟨bounds r.request,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible
  · subst region
    by_contra hn
    have ho : bounds r.tokens<extent r.tokens := by omega
    have impossible := hb.2.1 (7 : Fin 8) (⟨bounds r.tokens,ho⟩,PUnit.unit) trivial
    exact Nat.lt_irrefl _ impossible

#axiomsClean input_agreement
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.KernelIO
