import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.HistoricalEquivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
noncomputable section

/-- Full FP32 source-operation execution, including explicit BF16 store casts,
all chunks, all rows, arbitrary autotune trials and the actual returned tuple.
Primitive numerics and the selected ordered tree stay visible parameters; this
is not a claim that FP32/BF16 execution equals exact Real arithmetic. -/
specification fused_layernorm_source_fp32_bf16 {α : Type} [Inhabited α]
    (M : Algebra α) (eq32 : α → α → Bool) (s : State α) (read : Nat → α)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (a : Allocations)
    (h : Contract.InputContract w b x) (api : Host.Inputs s read w b x W B X)
    (live : RegionName → Prop) (hX : live (region X)) (hW : live (region W)) (hB : live (region B))
    (fresh : a.Fresh live) (cs : List Configuration) (last : Configuration) :
    Host.NativeSafe (Contract.rows x) (Contract.width x) ∧
    ∃ ret, Host.checked M eq32 w b x W B X s a cs last = some ret ∧
      ReturnedCorrect M eq32 s (descriptor X [Contract.width x,1]) (descriptor W [1]) (descriptor B [1])
        a (Contract.rows x) (Contract.width x) last ret ∧
      (∀ q, live q → ∀ off, ret.state.mem q off = s.mem q off) := by
  refine ⟨Host.native_safe s read w b x W B X api h, ?_⟩
  rw [Host.checked_eq _ _ _ _ _ _ _ _ _ _ _ _ h]
  exact host_complete M eq32 s _ _ _ a _ _ (Host.domain s read w b x W B X api h)
    live hX hW hB fresh cs last

abbrev python_checks_iff_preconditions := Contract.checkArguments_iff
abbrev all_native_indices := @Host.native_bounds
abbrev source_kernel_binding := Source.kernel_eq
abbrev source_host_binding := @Source.host_eq

#axiomsClean fused_layernorm_source_fp32_bf16
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.HistoricalEquivalence
