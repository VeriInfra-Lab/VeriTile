import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeSource
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Addresses
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Host

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.HostFPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section

/-- Actual compiler-selected V1/V2/V4 register layout. Input maximum precedes
subtraction, mask addition, log2(e) multiplication, exp2 and ordered sum/division. -/
def reference (inst : Instructions) (profile : Profile) := NativeGraph.row inst profile.vector

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

def memory : Memory (@Floating.access Word inferInstance) where
  replace := fun s cells => {s with mem := cells}
  cells := fun _ _ => rfl

specification softmax_wrapper_fp32 (inst : Instructions) (profile : Profile) (layout : Layout)
    (x : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (maskValid : ∀ m,mask=some m → Valid layout before (descriptor m))
    (allocation : Allocation Floating.access layout x before) :
    ∃ final,runWrapper Floating.access memory layout (NativeGraph.evaluate inst profile.vector)
        x mask dim before allocation=.ok allocation.output final ∧
      Preserves Floating.access before final ∧ Valid layout final (descriptor allocation.output) ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference inst profile (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r)
            (maskValues Floating.access negateWord before.program mask r) c) :=
  returned_tensor Floating.access memory negateWord layout (NativeGraph.evaluate inst profile.vector)
    (reference inst profile) (NativeGraph.source inst profile.vector) (NativeGraph.terminates inst profile.vector)
    x mask dim h before valid maskValid allocation

specification softmax_wrapper_if_local (actual specified : Instructions)
    (numerical : InstructionAgreement actual specified) (profile : Profile) (layout : Layout)
    (x : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (maskValid : ∀ m,mask=some m → Valid layout before (descriptor m))
    (allocation : Allocation Floating.access layout x before) :
    ∃ final,runWrapper Floating.access memory layout (NativeGraph.evaluate actual profile.vector)
        x mask dim before allocation=.ok allocation.output final ∧
      ∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference specified profile (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r)
            (maskValues Floating.access negateWord before.program mask r) c := by
  obtain ⟨final,run,_,_,values⟩ := softmax_wrapper_fp32 actual profile layout x mask dim h before valid maskValid allocation
  exact ⟨final,run,fun r c hr hc => (values r c hr hc).trans
    (NativeGraph.row_agreement actual specified numerical profile.vector _ _ _ _ c)⟩

specification softmax_returned_object (inst : Instructions) (profile : Profile) (layout : Layout)
    (x caller : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (before : World (State Word)) (valid : Valid layout before (descriptor x))
    (maskValid : ∀ m,mask=some m → Valid layout before (descriptor m))
    (callerValid : Valid layout before (descriptor caller))
    (allocation : Allocation Floating.access layout x before) (encode : Cell Word → Nat → BitVec 8) :
    ∃ final,runWrapper Floating.access memory layout (NativeGraph.evaluate inst profile.vector)
        x mask dim before allocation=.ok allocation.output final ∧
      allocation.output.metadata.shape=x.metadata.shape ∧
      allocation.output.metadata.device=x.metadata.device ∧
      allocation.output.metadata.dtype=.floating .fp32 ∧
      allocation.output.tensor.object≠x.tensor.object ∧
      (∀ byte,backingByte (worldAccess Floating.access) encode final caller.region byte=
        backingByte (worldAccess Floating.access) encode before caller.region byte) ∧
      (∀ r c,r<rows x.metadata → c<cols x.metadata →
        objectValue Floating.access negateWord final.program allocation.output r c=
          reference inst profile (cols x.metadata) (blockSize (cols x.metadata))
            (objectValue Floating.access negateWord before.program x r)
            (maskValues Floating.access negateWord before.program mask r) c) := by
  obtain ⟨final,run,frame,_,values⟩ := softmax_wrapper_fp32 inst profile layout x mask dim h before valid maskValid allocation
  exact ⟨final,run,allocation.shape,allocation.device,allocation.output.laws.dtype,allocation.freshObject,
    fun byte => caller_bytes Floating.access encode layout before final frame (descriptor caller) callerValid byte,values⟩

#specNonCircular reference avoiding [softmax_kernel,runWrapper,NativeGraph.evaluate,Floating.evaluate]
#axiomsClean softmax_wrapper_fp32
#axiomsClean softmax_wrapper_if_local
#axiomsClean softmax_returned_object
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.HostFPEquiv
