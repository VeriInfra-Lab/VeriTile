import bench.repairs.tritonbench_g.triton_softmax.Contract
import bench.repairs.tritonbench_g.triton_softmax.Proofs.NativeSource
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Addresses
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Host

namespace VeriTile.Bench.Repairs.TritonSoftmax.HostFPEquiv
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel
open VeriTile.Triton FP.Structural Contiguous
open VeriTile.Bench.Repairs.TritonSoftmax.NativeGraph
noncomputable section

/-- Production layout V1: all strides are runtime i64 arguments, including
unit column strides. Preserve the actual register/shuffle/shared-memory graph,
log2(e) multiplication, exp2 approximation and div.full order. -/
def reference (inst : Instructions) := NativeGraph.row inst 1

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

theorem reference_congr (inst : Instructions) (n B : Nat) (xs ys : Nat → Word) (c : Nat)
    (_hc : c<n) (same : ∀ j,j<n → xs j=ys j) : reference inst n B xs c=reference inst n B ys c := by
  have mask : masked n xs=masked n ys := by
    funext j
    by_cases hj : j<n <;> simp [masked,hj,same j]
  simp only [reference,NativeGraph.row,denominator,numerator,maximum,mask]

/-- Whole typed-source/native-graph execution on the original Tensor's storage.
The six numerical ports are scalar raw-word operations, never row callbacks. -/
specification softmax_wrapper_fp32 (inst : Instructions) (layout : Layout)
    (x : Object) (h : InputContract x.metadata)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (allocation : Allocation Floating.access layout x before) :
    ∃ final,runWrapper Floating.access layout (NativeGraph.evaluate inst 1) x before allocation=.ok allocation.output final ∧
      Returned Floating.access negateWord layout (reference inst) (descriptor x) (descriptor allocation.output) before final ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r) c) :=
  returned_tensor Floating.access negateWord layout (NativeGraph.evaluate inst 1) (reference inst)
    (NativeGraph.source inst 1) (reference_congr inst) (NativeGraph.source_terminates inst 1)
    x h before valid allocation

/-- Conformance transports only scalar instruction equality through the exact
ordered graph. Admission includes all raw FP32 bit patterns. -/
specification softmax_wrapper_if_local (actual specified : Instructions)
    (numerical : InstructionAgreement actual specified) (layout : Layout)
    (x : Object) (h : InputContract x.metadata)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (allocation : Allocation Floating.access layout x before) :
    ∃ final,runWrapper Floating.access layout (NativeGraph.evaluate actual 1) x before allocation=.ok allocation.output final ∧
      ∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference specified (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r) c := by
  obtain ⟨final,run,_,values⟩ := softmax_wrapper_fp32 actual layout x h before valid allocation
  exact ⟨final,run,fun r c hr hc => (values r c hr hc).trans
    (NativeGraph.row_agreement actual specified numerical 1 _ _ _ c)⟩

/-- Every caller backing byte is unchanged, including bytes outside Tensor
views. Shape, device, dtype and object identity belong to the actual allocation. -/
specification softmax_returned_object (inst : Instructions) (layout : Layout)
    (x caller : Object) (h : InputContract x.metadata)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (callerValid : Valid layout before (descriptor caller))
    (allocation : Allocation Floating.access layout x before)
    (encode : Cell Word → Nat → BitVec 8) :
    ∃ final,runWrapper Floating.access layout (NativeGraph.evaluate inst 1) x before allocation=.ok allocation.output final ∧
      allocation.output.metadata.shape=x.metadata.shape ∧
      allocation.output.metadata.device=x.metadata.device ∧
      allocation.output.metadata.dtype=.floating .fp32 ∧
      allocation.output.tensor.object≠x.tensor.object ∧
      (∀ byte,backingByte (worldAccess Floating.access) encode final caller.region byte=
        backingByte (worldAccess Floating.access) encode before caller.region byte) ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference inst (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r) c) := by
  obtain ⟨final,run,result,values⟩ := softmax_wrapper_fp32 inst layout x h before valid allocation
  exact ⟨final,run,allocation.shape,allocation.device,allocation.output.laws.dtype,
    allocation.freshObject,fun byte => caller_bytes Floating.access encode layout before final
      result.frame (descriptor caller) callerValid byte,values⟩

#axiomsClean softmax_returned_object

#specNonCircular reference avoiding [softmax_kernel,runWrapper,NativeGraph.evaluate,Floating.evaluate]
#axiomsClean softmax_wrapper_fp32
#axiomsClean softmax_wrapper_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonSoftmax.HostFPEquiv
