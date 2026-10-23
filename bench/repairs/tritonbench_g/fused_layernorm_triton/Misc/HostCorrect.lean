import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Logical

/-! Contiguous FP32, offset-zero device-0 inputs; BF16 output and FP32 stats.
The independent Real moments include every logical column for arbitrary width.
Exact native numerical conformance is separate from this ideal interpretation. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.HistoricalCorrectness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
noncomputable section

abbrev python_checks_iff_preconditions := Contract.checkArguments_iff
abbrev native_index_bounds := @Host.native_bounds
abbrev native_index_exact := NativeIndex.source_offsets
abbrev returned_metadata := Host.metadata

/-- Whole guarded host result, reusing the original full Welford/grid proof.
InputContract is exactly the guard predicate. Tensor/storage/allocator bindings
are ordinary API contracts. Complete is the selected helper-tree leaf coverage. -/
specification fused_layernorm_real (rest : Algebra ℝ) (s : State ℝ) (read : Nat → ℝ)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (a : Allocations)
    (h : Contract.InputContract w b x) (api : Host.Inputs s read w b x W B X)
    (live : RegionName → Prop) (hX : live (region X)) (hW : live (region W)) (hB : live (region B))
    (fresh : a.Fresh live) (cs : List Configuration) (last : Configuration) (complete : last.tree.Complete) :
    Host.NativeSafe (Contract.rows x) (Contract.width x) ∧
    ∃ ret, Host.checked (ideal rest) realEqual w b x W B X s a cs last = some ret ∧
      ret.tensors = (outputTensor a (Contract.rows x) (Contract.width x),
        descriptor X [Contract.width x,1], meanTensor a (Contract.rows x), inverseTensor a (Contract.rows x)) ∧
      (∀ row, row < Contract.rows x → ret.state.mem a.mean row =
        .mk .real (Contract.mean (Contract.width x) (fun col => logicalRead read X (col + Contract.width x * row)))) ∧
      (∀ row, row < Contract.rows x → ret.state.mem a.inverse row =
        .mk .real (1 / Real.sqrt (Contract.variance (Contract.width x)
          (fun col => logicalRead read X (col + Contract.width x * row)) + 1e-5))) ∧
      (∀ row, row < Contract.rows x → ∀ col, col < Contract.width x →
        ret.state.mem a.output (col + Contract.width x * row) =
          .mk .bf16 (Contract.output (Contract.width x) (fun j => logicalRead read X (j + Contract.width x * row))
            (logicalRead read W) (logicalRead read B) col)) ∧
      (∀ q off, gridMiss a.output a.mean a.inverse (Contract.rows x) (Contract.width x) q off →
        ret.state.mem q off = s.mem q off) := by
  refine ⟨Host.native_safe s read w b x W B X api h, ?_⟩
  rw [Host.checked_eq _ _ _ _ _ _ _ _ _ _ _ _ h]
  obtain ⟨ret, run, metadata, mean, inverse, output, frame⟩ :=
    host_real_complete rest s _ _ _ a _ _ (Host.domain s read w b x W B X api h)
      live hX hW hB fresh cs last complete
  refine ⟨ret, run, metadata, ?_, ?_, ?_, frame⟩
  · intro row hr
    simpa only [descriptor, (Logical.moments s read w b x W B X api h row hr).1] using mean row hr
  · intro row hr
    simpa only [descriptor, logicalInverse, (Logical.moments s read w b x W B X api h row hr).2] using inverse row hr
  · intro row hr col hc
    simpa only [descriptor, Logical.output s read w b x W B X api h row col hr hc] using output row hr col hc


#axiomsClean fused_layernorm_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.HistoricalCorrectness
