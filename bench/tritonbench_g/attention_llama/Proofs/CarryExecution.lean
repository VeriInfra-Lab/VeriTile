import bench.tritonbench_g.attention_llama.Proofs.CarryLoop
import bench.tritonbench_g.attention_llama.Proofs.Output

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source
open VeriTile.Bench.Repairs.AttentionLlama MaskCoverage
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem exec_surface (k : ComputeKernel) (s : State (Term α)) :
    exec algebra k s=run algebra k.surfaceBody s := by cases k <;> rfl

/-- Actual successful execution of the entire shared source, including its
preamble, dynamic range, dtype branches, final cast and masked output store. -/
theorem execution_of_reached (c : Config) (s : State (Term α)) (scale : Term α) (bytes : ReachedBytes c s) :
    ∃ f, exec algebra (kernel c) (bindScale s scale)=some f ∧
      f.mem=(stored c s (accumulated c s scale (count (stop c s)))).mem := by
  obtain ⟨u,hu,hc⟩ := range_execution_of_reached c s scale bytes (count (stop c s)) 0 (by omega)
    (entry c s scale).val (entry_context c s scale)
  let final := finish c s u scale (count (stop c s))
    (accumulated c s scale (count (stop c s))) hc
  refine ⟨final.val,?_,finish_memory c s u scale _ _ hc⟩
  rw [exec_surface,source_split,run_append,(entry c s scale).property]
  simp only [Option.bind_some,run,step,evalOp_unfold,entry_stop,Option.bind_some]
  change (range algebra "start_n" (0*64) (stop c s) 64 (parts c).body
    (entry c s scale).val).bind (run algebra (parts c).after)=some final.val
  rw [hu]
  exact final.property

/-- Source output and complete typed-cell frame. Only active output lanes
must be injective; masked padded addresses impose no layout condition. -/
theorem values_and_frame_of_reached (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ReachedBytes c s) (hinj : ActiveInjective c s) :
    ∃ f, exec algebra (kernel c) (bindScale s scale)=some f ∧
      (∀ i : TileIndex [64,c.D], rows s i.1<c.H →
        f.mem c.Out (address c s i)=
          .mk .fp16 (.half ((accumulated c s scale (count (stop c s))).accumulator i))) ∧
      (∀ r o, (r≠c.Out ∨ ∀ i : TileIndex [64,c.D], rows s i.1<c.H → o≠address c s i) →
        f.mem r o=s.mem r o) := by
  obtain ⟨f,he,hm⟩ := execution_of_reached c s scale bytes
  refine ⟨f,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact stored_read c s _ hinj i hi
  · intro r o ho
    rw [hm]
    exact stored_frame c s _ r o ho

/-- Compatibility with the previously accepted global representation theorem. -/
theorem execution (c : Config) (s : State (Term α)) (scale : Term α) (bytes : ByteRange s) :
    ∃ f, exec algebra (kernel c) (bindScale s scale)=some f ∧
      f.mem=(stored c s (accumulated c s scale (count (stop c s)))).mem :=
  execution_of_reached c s scale (bytes.reached c s)

theorem values_and_frame (c : Config) (s : State (Term α)) (scale : Term α)
    (bytes : ByteRange s) (hinj : ActiveInjective c s) :
    ∃ f, exec algebra (kernel c) (bindScale s scale)=some f ∧
      (∀ i : TileIndex [64,c.D], rows s i.1<c.H →
        f.mem c.Out (address c s i)=
          .mk .fp16 (.half ((accumulated c s scale (count (stop c s))).accumulator i))) ∧
      (∀ r o, (r≠c.Out ∨ ∀ i : TileIndex [64,c.D], rows s i.1<c.H → o≠address c s i) →
        f.mem r o=s.mem r o) :=
  values_and_frame_of_reached c s scale (bytes.reached c s) hinj

#axiomsClean execution_of_reached
#axiomsClean values_and_frame_of_reached
#axiomsClean execution
#axiomsClean values_and_frame
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
