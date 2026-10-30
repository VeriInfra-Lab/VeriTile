import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVNative

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]

theorem source_result_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (hc : s.pids 1<c.chunks) :
    ∃ t, FP.Structural.exec (engine (withDot (algebra ops.zero) gradientDot)) (kernel c scale) s=some t ∧
      (∀ i, keyActive c s i → lower ops literal ((t.mem "dq" (keyAddress c s i)).read .real)=
        Body.dq ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
          (history ops literal c s "h") i.1.val (feature c s i.2.1.val)) ∧
      (∀ i, keyActive c s i → lower ops literal ((t.mem "dk" (keyAddress c s i)).read .real)=
        Body.dk ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
          (history ops literal c s "dh") i.1.val (feature c s i.2.1.val)) ∧
      (∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i →
        lower ops literal ((t.mem "dv" (partialAddress c s f i)).read .real)=
          Body.dvPartial ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1)
            (history ops literal c s "dh") i.1.val (column c f i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"dq" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dk" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dv" ∨ ∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i → off≠partialAddress c s f i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory (algebra ops.zero) gradientDot c scale s
  refine ⟨t,ht,?_,?_,?_,?_⟩
  · intro i hi
    rw [hm,query_readback (algebra ops.zero) gradientDot c s scale i hi]
    exact query_body_local ops literal laws c s tag scale i
  · intro i hi
    rw [hm,key_readback (algebra ops.zero) gradientDot c s scale i hi]
    exact key_body_local ops literal laws c s tag scale i hc hi
  · intro f hf i hi
    rw [hm,partial_readback (algebra ops.zero) gradientDot c s scale f i hf hi]
    exact partial_body_local ops literal laws c s tag scale f i hc
  · intro r off hq hk hv
    rw [hm]
    exact frame (algebra ops.zero) gradientDot c s scale r off hq hk hv

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem source_result (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (hc : s.pids 1<c.chunks) :
    ∃ t, FP.Structural.exec (engine (withDot (algebra ops.zero) gradientDot)) (kernel c scale) s=some t ∧
      (∀ i, keyActive c s i → lower ops literal ((t.mem "dq" (keyAddress c s i)).read .real)=
        Body.dq ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
          (history ops literal c s "h") i.1.val (feature c s i.2.1.val)) ∧
      (∀ i, keyActive c s i → lower ops literal ((t.mem "dk" (keyAddress c s i)).read .real)=
        Body.dk ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
          (history ops literal c s "dh") i.1.val (feature c s i.2.1.val)) ∧
      (∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i →
        lower ops literal ((t.mem "dv" (partialAddress c s f i)).read .real)=
          Body.dvPartial ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1)
            (history ops literal c s "dh") i.1.val (column c f i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"dq" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dk" ∨ ∀ i, keyActive c s i → off≠keyAddress c s i) →
        (r≠"dv" ∨ ∀ f, f<c.valueBlocks → ∀ i, valueActive c s f i → off≠partialAddress c s f i) →
        t.mem r off=s.mem r off) :=
  source_result_local ops literal laws c s (ReadTags.of_tagged c s tag) scale hc

#axiomsClean source_result_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
