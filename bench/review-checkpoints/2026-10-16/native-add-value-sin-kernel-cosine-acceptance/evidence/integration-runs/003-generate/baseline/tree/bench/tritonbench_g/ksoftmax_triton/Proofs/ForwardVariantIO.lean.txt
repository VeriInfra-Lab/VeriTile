import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardCandidateReal

/-! Draft public assembly for root validation. Both IO objects execute the
actual complete fp32 quotation after explicit dtype erasure. Candidate IO is
transported through the independently proved equality of flattened runs.
No candidate trace-safety theorem is inferred from final-state equality.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- The two variants share the original read/write windows and independent
M read gate. Only the selected complete source kernel changes. -/
def fp32VariantIO (v : Bool) (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) : Masked2DKernelIO₂ :=
  { fp32ErasedIO Y X M ym yn xm xn sm K D log mode causal fp16 with
    kernel := (Candidate.source v Y X M ym yn xm xn sm K D
      log mode.python causal fp16).eraseDType
    projection := Candidate.erased_projects v Y X M ym yn xm xn sm K D
      log mode.python causal fp16 }

/-- The public original selector is the exact previously verified IO object,
not a replacement kernel chosen only for the theorem. -/
theorem fp32_original_io (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) :
    fp32VariantIO Bool.false Y X M ym yn xm xn sm K D log mode causal fp16 =
      fp32ErasedIO Y X M ym yn xm xn sm K D log mode causal fp16 := by
  rfl

/-- Successful finite flat execution, independent scalar formula on every
written lane, and the complete frame, for both source variants. The original
IO proof establishes its trace walk separately; this theorem transports the
IO conclusion using Candidate.flat_exec, not an alleged trace implication.
The original LOG/causal bottom payload is demoted only by the real observer. -/
specification ksoftmax_forward_fp32_variant_real_correctness (v : Bool)
    (Y X M : RegionName) (ym yn xm xn sm K D : Nat) (hD : 0 < D)
    (log : Bool) (mode : MaskMode) (causal fp16 : Bool) :
    fp32VariantIO v Y X M ym yn xm xn sm K D log mode causal fp16 ⊨
      fun _ row xs masks i => formula K row log causal mode xs masks i := by
  intro A hd hr hc batch row hx hm hy hscratch xs masks s hbatch hrow hu hxs hmasks
  obtain ⟨t, hrun, hvalue, hframe⟩ := ksoftmax_forward_fp32_real_correctness
    Y X M ym yn xm xn sm K D hD log mode causal fp16
    A hd hr hc batch row hx hm hy hscratch xs masks s hbatch hrow hu hxs hmasks
  refine ⟨t, ?_, hvalue, hframe⟩
  exact (Candidate.flat_exec A v Y X M ym yn xm xn sm K D
    log mode.python causal fp16 (A.flattenState s)).trans hrun

#axiomsClean fp32_original_io
#axiomsClean ksoftmax_forward_fp32_variant_real_correctness
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward
