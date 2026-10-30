import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.FPEquiv
import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Memory
import bench.repairs.tritonbench_g.chunk_gated_attention.FPEquiv

/-! Complete original Tensor applications, using the same existing raw source
execution and independent targets as the primary kernel interfaces. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.TensorBridge
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
open HostStorage RawStorage
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent closed form over valid logical times; the input gate is
already cumulative, and history records the state before each update. -/
abbrev mathematicalHistory := RealSource.History
abbrev mathematicalFinal := RealSource.Final
abbrev mathematicalPrefix := RealSource.Prefix

def InnerRuns (r : Admission.Inner) (read : Nat → ℝ) (scratch final : BlockState) : Prop :=
  DirectRealSequence.Runs (RealSource.layout r) RealSource.regions .fp32 .fp32 .fp32 .fp32 .fp32
    r.gateK r.seed.isSome r.final.isSome Bool.false
    (LaunchPlan.plans (r.config.heads*(r.config.PK/r.BK)*(r.config.PV/r.BV)))
    {RealSource.inputState r read scratch with undef := fun _ _ => 0} final

def PrefixRuns (r : Admission.Prefix) (BS : Nat) (read : Nat → ℝ)
    (scratch final : BlockState) : Prop :=
  DirectCumulativeSequence.Runs (RealSource.prefixLayout r BS) (r.B*r.H) "s" "o" Bool.false .fp32 .fp32
    (DirectCumulativeWhole.plans (RealSource.prefixLayout r BS) (r.B*r.H))
    {RealSource.prefixInput r read scratch with undef := fun _ _ => 0} final

/-- Every actual recurrent launch executes the shared DSL. Its valid output
Tensor elements have the independent mathematical history/final values, and
all addresses outside the two kernel output allocations are unchanged.
There is no Python-host-success or whole-result premise. -/
theorem inner_correct (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (h : InnerContract r heap p) (b : Bindings (innerDescriptors r heap p))
    (read : Nat → ℝ) (scratch : BlockState) :
    Spec.Real (∃ final, InnerRuns r read scratch final ∧
      (∀ i, i < innerCount r →
        logical (innerRealStores r heap p final read) (b.get p.history) (HostData.historyOffset r i) =
          mathematicalHistory r read i) ∧
      (r.final.isSome = Bool.true → ∀ i, i < finalCount r →
        logical (innerRealStores r heap p final read) (b.get p.final) (HostData.finalOffset r i) =
          mathematicalFinal r read i) ∧
      (∀ address, Outside p.history.byteBase (historyCapacity r) address →
        (r.final.isSome = Bool.true → Outside p.final.byteBase (finalCapacity r) address) →
        innerRealStores r heap p final read address = read address)) := by
  let final := RealSource.innerResult r h.checked read scratch
  refine ⟨final,(RealSource.inner_exists r h.checked read scratch).choose_spec.1,?_,?_,
    inner_real_frame r heap p final read⟩
  · intro i hi
    have bound := HostCoordinates.history_bound r h.checked i hi
    rw [b.logical_direct _ p.history (history_member r heap p) p.historyValid.2.2.2.2.1 _
      (by simpa [history_count r heap p] using bound)]
    rw [inner_real_history _ _ _ _ _ _ bound]
    exact RealSource.history r h.checked read scratch i hi
  · intro enabled i hi
    have bound := HostCoordinates.final_bound r h.checked i hi
    rw [b.logical_direct _ p.final (final_member r heap p enabled) (p.finalValid enabled).2.2.2.2.1 _
      (by simpa [final_count r heap p enabled] using bound)]
    rw [inner_real_final _ _ _ enabled _ _ _ bound]
    exact RealSource.final r h.checked read scratch enabled i hi

/-- All three BS choices (and hence all nine autotune configurations) produce
the within-chunk prefix at every valid output Tensor element. -/
theorem prefix_correct (r : Admission.Prefix) (BS : Nat) (heap : Memory)
    (p : PrefixPrepared r heap) (h : PrefixContract r BS heap p)
    (b : Bindings (prefixDescriptors r heap p)) (read : Nat → ℝ) (scratch : BlockState) :
    Spec.Real (∃ final, PrefixRuns r BS read scratch final ∧
      (∀ i, i < prefixCount r →
        logical (prefixRealStores r heap p final read) (b.get p.output) (HostData.prefixOffset r i) =
          mathematicalPrefix r read i) ∧
      (∀ address, Outside p.output.byteBase (prefixCapacity r) address →
        prefixRealStores r heap p final read address = read address)) := by
  let final := RealSource.prefixResult r h.checked BS h.selected read scratch
  refine ⟨final,(RealSource.prefix_exists r h.checked BS h.selected read scratch).choose_spec.1,?_,?_⟩
  · intro i hi
    have bound := HostCoordinates.prefix_bound r h.checked i hi
    rw [b.logical_direct _ p.output (prefix_member r heap p) p.outputValid.2.2.2.2.1 _
      (by simpa [prefix_count r heap p] using bound)]
    rw [prefixRealStores,RealPhysical.commit_read _ _ _ _ _ bound]
    exact RealSource.prefix_value r h.checked BS h.selected read scratch i hi
  · exact RealPhysical.commit_frame _ _ _ _

#specNonCircular mathematicalHistory avoiding [InnerRuns,RealSource.innerResult,VeriTile.Triton.exec]
#specNonCircular mathematicalFinal avoiding [InnerRuns,RealSource.innerResult,VeriTile.Triton.exec]
#specNonCircular mathematicalPrefix avoiding [PrefixRuns,RealSource.prefixResult,VeriTile.Triton.exec]
#axiomsClean inner_correct
#axiomsClean prefix_correct
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.TensorBridge

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.TensorBridge
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory HostStorage RawStorage
noncomputable section
set_option maxHeartbeats 4000000

def innerExecute (r : Admission.Inner) (heap : Memory) (scratch : State NativeGraph.Expr) :=
  HostFP.batches NativeValue.algebra (NativeGrid.program (innerKernel r))
    (NativeGrid.innerOwns r.config) r.config.plans (HostData.innerInput r heap scratch)
def prefixExecute (r : Admission.Prefix) (BS : Nat) (heap : Memory) (scratch : State NativeGraph.Expr) :=
  HostFP.batches NativeValue.algebra (NativeGrid.program (prefixKernel r BS))
    (NativeGrid.prefixOwns (r.config BS)) (r.config BS).plans (HostData.prefixInput r heap scratch)

/-- The independent native graph contains actual gate loads, exp2 lowering,
zero-plus-seed, explicit casts, and the ordered live-accumulator FMA chain.
No Real-equality claim is made for rounded arithmetic. -/
abbrev nativeHistory := HostData.historyWord
abbrev nativeFinal := HostData.finalWord
abbrev nativePrefix := HostData.prefixWord

/-- Shared-source execution is derived, then its FP32 stores are read through
the actual output Tensor API. Raw input loads and signed pointer derivations
are included as conclusions of the ordinary checked input contract. -/
theorem inner_fp_equiv (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (h : InnerContract r heap p) (b : Bindings (innerDescriptors r heap p))
    (actual reference : ScalarNative.Primitives) (laws : ScalarLaws actual reference)
    (scratch : State NativeGraph.Expr) :
    ∃ final, innerExecute r heap scratch = some final ∧
      (∀ i, i < innerCount r →
        (Torch.FP32.logicalValue (readBits (innerNativeStores actual r heap p final))
          (b.get p.history).tensor (b.api p.history (history_member r heap p))
          (HostData.historyOffset r i)).bits = nativeHistory reference r heap scratch i) ∧
      (∀ enabled : r.final.isSome = Bool.true, ∀ i, i < finalCount r →
        (Torch.FP32.logicalValue (readBits (innerNativeStores actual r heap p final))
          (b.get p.final).tensor (b.api p.final (final_member r heap p enabled))
          (HostData.finalOffset r i)).bits = nativeFinal reference r heap scratch i) ∧
      (∀ address, Outside p.history.byteBase (historyCapacity r) address →
        (r.final.isSome = Bool.true → Outside p.final.byteBase (finalCapacity r) address) →
        innerNativeStores actual r heap p final address = p.memory address) ∧
      FPEquiv.InnerBinding r heap p scratch := by
  let final := HostData.innerResult r heap scratch
  refine ⟨final,HostData.inner_source r heap scratch,?_,?_,
    inner_native_frame actual r heap p final,
    FPEquiv.inner_binding r h.checked h.active heap p h.pointers scratch⟩
  · intro i hi
    have bound := HostCoordinates.history_bound r h.checked i hi
    rw [b.logical_fp32 _ p.history (history_member r heap p)
      p.historyValid.2.2.2.2.1 p.historyValid.2.2.2.2.2.1 _
      (by simpa [history_count r heap p] using bound)]
    change read (innerNativeStores actual r heap p final) _ = _
    rw [inner_native_history _ _ _ _ _ _ bound]
    exact scalar_eval_eq actual reference laws _
  · intro enabled i hi
    have bound := HostCoordinates.final_bound r h.checked i hi
    rw [b.logical_fp32 _ p.final (final_member r heap p enabled)
      (p.finalValid enabled).2.2.2.2.1 (p.finalValid enabled).2.2.2.2.2.1 _
      (by simpa [final_count r heap p enabled] using bound)]
    change read (innerNativeStores actual r heap p final) _ = _
    rw [inner_native_final _ _ _ _ enabled _ _ bound]
    exact scalar_eval_eq actual reference laws _

theorem prefix_fp_equiv (r : Admission.Prefix) (BS : Nat) (heap : Memory)
    (p : PrefixPrepared r heap) (h : PrefixContract r BS heap p)
    (b : Bindings (prefixDescriptors r heap p))
    (actual reference : ScalarNative.Primitives) (laws : ScalarLaws actual reference)
    (scratch : State NativeGraph.Expr) :
    ∃ final, prefixExecute r BS heap scratch = some final ∧
      (∀ i, i < prefixCount r →
        (Torch.FP32.logicalValue (readBits (prefixNativeStores actual r heap p final))
          (b.get p.output).tensor (b.api p.output (prefix_member r heap p))
          (HostData.prefixOffset r i)).bits = nativePrefix reference r BS heap scratch i) ∧
      (∀ address, Outside p.output.byteBase (prefixCapacity r) address →
        prefixNativeStores actual r heap p final address = p.memory address) ∧
      FPEquiv.PrefixBinding r heap p BS scratch := by
  let final := HostData.prefixResult r BS heap scratch
  refine ⟨final,HostData.prefix_source r BS heap scratch,?_,?_,
    FPEquiv.prefix_binding r h.checked h.active BS h.selected heap p h.pointers scratch⟩
  · intro i hi
    have bound := HostCoordinates.prefix_bound r h.checked i hi
    rw [b.logical_fp32 _ p.output (prefix_member r heap p)
      p.outputValid.2.2.2.2.1 p.outputValid.2.2.2.2.2.1 _
      (by simpa [prefix_count r heap p] using bound)]
    change read (prefixNativeStores actual r heap p final) _ = _
    rw [prefixNativeStores,stores_read _ _ _ _ _ bound]
    exact scalar_eval_eq actual reference laws _
  · exact stores_frame _ _ _ _

#specNonCircular nativeHistory avoiding [innerExecute,FP.Structural.exec,HostFP.batches]
#specNonCircular nativeFinal avoiding [innerExecute,FP.Structural.exec,HostFP.batches]
#specNonCircular nativePrefix avoiding [prefixExecute,FP.Structural.exec,HostFP.batches]
#axiomsClean scalar_eval_eq
#axiomsClean inner_fp_equiv
#axiomsClean prefix_fp_equiv
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.TensorBridge
