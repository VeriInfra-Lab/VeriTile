import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.Kernels
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RealPhysical

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory HostStorage RawStorage
noncomputable section
set_option maxHeartbeats 4000000

theorem history_count (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap) :
    p.history.shape.prod = historyCapacity r := by
  rw [p.historyValid.2.2.2.2.2.2.2.2]
  simp [historyCapacity,Admission.Inner.config,Nat.mul_assoc]
theorem final_count (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (enabled : r.final.isSome = Bool.true) : p.final.shape.prod = finalCapacity r := by
  rw [(p.finalValid enabled).2.2.2.2.2.2.2.2]
  simp [finalCapacity,Admission.Inner.config,Nat.mul_assoc]
theorem prefix_count (r : Admission.Prefix) (heap : Memory) (p : PrefixPrepared r heap) :
    p.output.shape.prod = prefixCapacity r := by
  rw [p.outputValid.2.2.2.2.2.2.2.2]
  simp [prefixCapacity,Nat.mul_assoc]

theorem history_member (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap) :
    p.history ∈ innerDescriptors r heap p := by
  simp [innerDescriptors,InnerPrepared.tensors]
theorem final_member (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (enabled : r.final.isSome = Bool.true) : p.final ∈ innerDescriptors r heap p := by
  simp [innerDescriptors,InnerPrepared.tensors,enabled]
theorem prefix_member (r : Admission.Prefix) (heap : Memory) (p : PrefixPrepared r heap) :
    p.output ∈ prefixDescriptors r heap p := by simp [prefixDescriptors]

/-- Write back executed Real cells at actual output Tensor addresses. This
definition contains no mathematical target and performs no caller copy-back. -/
def innerRealStores (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (final : BlockState) (read : Nat → ℝ) : Nat → ℝ :=
  let history := RealPhysical.commit read p.history.byteBase (historyCapacity r) (final.readMem "h")
  if r.final.isSome then
    RealPhysical.commit history p.final.byteBase (finalCapacity r) (final.readMem "ht")
  else history

def prefixRealStores (r : Admission.Prefix) (heap : Memory) (p : PrefixPrepared r heap)
    (final : BlockState) (read : Nat → ℝ) : Nat → ℝ :=
  RealPhysical.commit read p.output.byteBase (prefixCapacity r) (final.readMem "o")

theorem inner_real_history (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (final : BlockState) (read : Nat → ℝ) (i : Nat) (hi : i < historyCapacity r) :
    innerRealStores r heap p final read (p.history.byteBase+4*i) = final.readMem "h" i := by
  unfold innerRealStores
  split
  · rename_i enabled
    rw [RealPhysical.commit_frame]
    · exact RealPhysical.commit_read _ _ _ _ i hi
    · apply disjoint_byte (p.finalFresh enabled p.history (by simp))
      rw [history_count r heap p]
      omega
  · exact RealPhysical.commit_read _ _ _ _ i hi

theorem inner_real_final (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (enabled : r.final.isSome = Bool.true) (final : BlockState) (read : Nat → ℝ)
    (i : Nat) (hi : i < finalCapacity r) :
    innerRealStores r heap p final read (p.final.byteBase+4*i) = final.readMem "ht" i := by
  simp only [innerRealStores,enabled,ite_true]
  exact RealPhysical.commit_read _ _ _ _ i hi

theorem inner_real_frame (r : Admission.Inner) (heap : Memory) (p : InnerPrepared r heap)
    (final : BlockState) (read : Nat → ℝ) (address : Nat)
    (history : Outside p.history.byteBase (historyCapacity r) address)
    (last : r.final.isSome = Bool.true → Outside p.final.byteBase (finalCapacity r) address) :
    innerRealStores r heap p final read address = read address := by
  unfold innerRealStores
  split
  · rename_i enabled
    rw [RealPhysical.commit_frame _ _ _ _ _ (last enabled)]
    exact RealPhysical.commit_frame _ _ _ _ _ history
  · exact RealPhysical.commit_frame _ _ _ _ _ history

/-- FP32 stores serialize the cells produced by the source execution. Scalar
primitive interpretations are explicit; loaded bit words remain identical. -/
def innerNativeStores (I : ScalarNative.Primitives) (r : Admission.Inner) (heap : Memory)
    (p : InnerPrepared r heap) (final : State NativeGraph.Expr) : Memory :=
  let history := stores p.memory p.history.byteBase (historyCapacity r)
    (fun i => ScalarNative.eval I ((final.mem "h" i).read .real))
  if r.final.isSome then stores history p.final.byteBase (finalCapacity r)
    (fun i => ScalarNative.eval I ((final.mem "ht" i).read .fp32)) else history

def prefixNativeStores (I : ScalarNative.Primitives) (r : Admission.Prefix) (heap : Memory)
    (p : PrefixPrepared r heap) (final : State NativeGraph.Expr) : Memory :=
  stores p.memory p.output.byteBase (prefixCapacity r)
    (fun i => ScalarNative.eval I ((final.mem "o" i).read .fp32))

theorem inner_native_history (I : ScalarNative.Primitives) (r : Admission.Inner) (heap : Memory)
    (p : InnerPrepared r heap) (final : State NativeGraph.Expr) (i : Nat) (hi : i < historyCapacity r) :
    read (innerNativeStores I r heap p final) (p.history.byteBase+4*i) =
      ScalarNative.eval I ((final.mem "h" i).read .real) := by
  unfold innerNativeStores
  split
  · rename_i enabled
    rw [stores_read_frame]
    · exact stores_read _ _ _ _ i hi
    · apply disjoint_cell (countB := historyCapacity r) _ hi
      rw [←history_count r heap p]
      exact p.finalFresh enabled p.history (by simp)
  · exact stores_read _ _ _ _ i hi

theorem inner_native_final (I : ScalarNative.Primitives) (r : Admission.Inner) (heap : Memory)
    (p : InnerPrepared r heap) (enabled : r.final.isSome = Bool.true)
    (final : State NativeGraph.Expr) (i : Nat) (hi : i < finalCapacity r) :
    read (innerNativeStores I r heap p final) (p.final.byteBase+4*i) =
      ScalarNative.eval I ((final.mem "ht" i).read .fp32) := by
  simp only [innerNativeStores,enabled,ite_true]
  exact stores_read _ _ _ _ i hi

theorem inner_native_frame (I : ScalarNative.Primitives) (r : Admission.Inner) (heap : Memory)
    (p : InnerPrepared r heap) (final : State NativeGraph.Expr) (address : Nat)
    (history : Outside p.history.byteBase (historyCapacity r) address)
    (last : r.final.isSome = Bool.true → Outside p.final.byteBase (finalCapacity r) address) :
    innerNativeStores I r heap p final address = p.memory address := by
  unfold innerNativeStores
  split
  · rename_i enabled
    rw [stores_frame _ _ _ _ _ (last enabled)]
    exact stores_frame _ _ _ _ _ history
  · exact stores_frame _ _ _ _ _ history

def readBits (heap : Memory) (address : Nat) : Torch.FP32.Word := ⟨read heap address⟩

#specNonCircular innerRealStores avoiding [RealSource.History,RealSource.Final]
#specNonCircular innerNativeStores avoiding [HostData.historyWord,HostData.finalWord,HostData.innerResult]
#axiomsClean inner_real_history
#axiomsClean inner_native_history
#axiomsClean inner_native_frame
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
