import bench.tritonbench_g.context_attn_bloom.IO
import bench.tritonbench_g.context_attn_bloom.Proofs.RealInputAgreement

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000

theorem read_query (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.query) :
    (inputs s r).q off=(mathInputs r extent xs).q off := by
  have h := hin (0 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMem r.query off=xs (0 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.payload,dif_pos ho,toValues] using h

theorem read_key (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.key) :
    (inputs s r).k off=(mathInputs r extent xs).k off := by
  have h := hin (1 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMem r.key off=xs (1 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.payload,dif_pos ho,toValues] using h

theorem read_value (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.value) :
    (inputs s r).v off=(mathInputs r extent xs).v off := by
  have h := hin (2 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMem r.value off=xs (2 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.payload,dif_pos ho,toValues] using h

theorem read_start (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.start) :
    (inputs s r).start off=(mathInputs r extent xs).start off := by
  have h := hin (3 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMemValue .int r.start off=xs (3 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.integer,dif_pos ho,toValues] using h

theorem read_sequence (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.sequence) :
    (inputs s r).sequence off=(mathInputs r extent xs).sequence off := by
  have h := hin (4 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMemValue .int r.sequence off=xs (4 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.integer,dif_pos ho,toValues] using h

theorem read_prompt (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.prompt) :
    (inputs s r).prompt off=(mathInputs r extent xs).prompt off := by
  have h := hin (5 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMemValue .int r.prompt off=xs (5 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.integer,dif_pos ho,toValues] using h

theorem read_request (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.request) :
    (inputs s r).request off=(mathInputs r extent xs).request off := by
  have h := hin (6 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMemValue .int r.request off=xs (6 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.integer,dif_pos ho,toValues] using h

theorem read_tokens (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (hin : (io g r extent).InputBinding s xs)
    (off : Nat) (ho : off<extent r.tokens) :
    (inputs s r).tokens off=(mathInputs r extent xs).tokens off := by
  have h := hin (7 : Fin 8) ⟨off,ho⟩ trivial
  change s.readMemValue .int r.tokens off=xs (7 : Fin 8) ⟨off,ho⟩ at h
  simpa only [inputs,mathInputs,KernelIO.metadata,KernelIO.integer,dif_pos ho,toValues] using h

theorem input_agreement (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (valid : domain g r extent xs) (s : BlockState)
    (hin : (io g r extent).InputBinding s xs) (idx : GridIndex (Host.grid g.arguments)) :
    RealInputAgreement.Agree r extent (Host.launchOf g.arguments idx) (inputs s r) (mathInputs r extent xs) := by
  have hb := valid.bounds idx
  exact ⟨read_start g r extent xs s hin _ hb.start,read_sequence g r extent xs s hin _ hb.sequence,
    read_prompt g r extent xs s hin _ hb.prompt,read_request g r extent xs s hin _ hb.request,
    read_query g r extent xs s hin,read_key g r extent xs s hin,read_value g r extent xs s hin,
    read_tokens g r extent xs s hin⟩

theorem input_capacity (g : Raw.Geometry) (r : Regions) (extent : RegionBounds)
    (xs : Values r extent) (s : BlockState) (bounds : RegionBounds)
    (hb : (io g r extent).InBounds bounds s xs) :
    ∀ region,region∈[r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.request,r.tokens] →
      extent region≤bounds region := by
  intro region hr
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with h|h|h|h|h|h|h|h
  · subst region
    by_contra hn
    have ho : bounds r.query<extent r.query := by omega
    exact Nat.lt_irrefl _ (hb.1 (0 : Fin 8) ⟨bounds r.query,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.key<extent r.key := by omega
    exact Nat.lt_irrefl _ (hb.1 (1 : Fin 8) ⟨bounds r.key,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.value<extent r.value := by omega
    exact Nat.lt_irrefl _ (hb.1 (2 : Fin 8) ⟨bounds r.value,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.start<extent r.start := by omega
    exact Nat.lt_irrefl _ (hb.1 (3 : Fin 8) ⟨bounds r.start,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.sequence<extent r.sequence := by omega
    exact Nat.lt_irrefl _ (hb.1 (4 : Fin 8) ⟨bounds r.sequence,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.prompt<extent r.prompt := by omega
    exact Nat.lt_irrefl _ (hb.1 (5 : Fin 8) ⟨bounds r.prompt,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.request<extent r.request := by omega
    exact Nat.lt_irrefl _ (hb.1 (6 : Fin 8) ⟨bounds r.request,ho⟩ trivial)
  · subst region
    by_contra hn
    have ho : bounds r.tokens<extent r.tokens := by omega
    exact Nat.lt_irrefl _ (hb.1 (7 : Fin 8) ⟨bounds r.tokens,ho⟩ trivial)

#axiomsClean input_agreement
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
