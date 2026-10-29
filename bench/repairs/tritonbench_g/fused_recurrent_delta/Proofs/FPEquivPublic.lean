import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.PhysicalLayout
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.WholeSource
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.ABI
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
noncomputable section
/-- Bit-for-bit complete current memory, reduced solely to four fixed-arity
FP32 instruction obligations. The independent source reference is recurrence
based; it does not invoke forward/backward loops or calls. -/
theorem fp_equiv (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word) (actual : Native.Instructions)
    (ports : Native.Obligations actual) (policy : Native.Policy) (m : Memory Native.Word) :
    Native.forward s f g scale actual policy m=WholeSource.forwardSoftwareReference s f g scale policy m ∧
    Native.backward s f g scale actual policy m=WholeSource.backwardSoftwareReference s f g scale policy m := by
  constructor
  · rw [Native.forward_transport s f g scale actual ports policy,WholeSource.software_forward]
  · rw [Native.backward_transport s f g scale actual ports policy,WholeSource.software_backward]
structure FPResult where
  forwardMemory : PhysicalLayout.Bytes
  backwardMemory : PhysicalLayout.Bytes
  outputs : ABI.Tensor×Option ABI.Tensor
  gradients : List (Option ABI.Tensor)
def runFP (c : ABI.Call) (gf gb : Trees) (actual : Native.Instructions)
    (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) : FPResult :=
  let f:=ABI.flags c
  let mf:=PhysicalLayout.forward c.shape f gf c.scale.launch actual pf initial
  let mb:=PhysicalLayout.backward c.shape f gb c.scale.launch actual pb mf
  ⟨mf,mb,ABI.forwardReturn c,ABI.backwardReturn c⟩

theorem whole_fp_bytes (c : ABI.Call) (gf gb : Trees) (actual : Native.Instructions)
    (ports : Native.Obligations actual) (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) :
    runFP c gf gb actual pf pb initial=runFP c gf gb Native.software pf pb initial := by
  unfold runFP
  dsimp only
  rw [PhysicalLayout.forward_transport c.shape (ABI.flags c) gf c.scale.launch actual ports pf,
    PhysicalLayout.backward_transport c.shape (ABI.flags c) gb c.scale.launch actual ports pb]

theorem fp_return_contract (c : ABI.Call) (gf gb : Trees) (actual : Native.Instructions)
    (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) :
    (runFP c gf gb actual pf pb initial).outputs=ABI.forwardReturn c ∧
    (runFP c gf gb actual pf pb initial).gradients=ABI.backwardReturn c ∧
    (runFP c gf gb actual pf pb initial).gradients.length=7 :=
  ⟨rfl,rfl,ABI.seven_backward_slots c⟩

theorem fp_forward_frame (c : ABI.Call) (gf gb : Trees) (actual : Native.Instructions)
    (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) (r : Region) (byte : Nat)
    (outside : ∀ loc,PhysicalLayout.At c.shape (ABI.flags c) r byte loc →
      ¬MemoryProof.ForwardCallWrites c.shape (ABI.flags c) loc.1) :
    (runFP c gf gb actual pf pb initial).forwardMemory r byte=initial r byte :=
  PhysicalLayout.publish_frame c.shape (ABI.flags c) _ initial _ r byte outside
theorem fp_backward_frame (c : ABI.Call) (gf gb : Trees) (actual : Native.Instructions)
    (pf pb : Native.Policy) (initial : PhysicalLayout.Bytes) (r : Region) (byte : Nat)
    (outside : ∀ loc,PhysicalLayout.At c.shape (ABI.flags c) r byte loc →
      ¬MemoryProof.BackwardWrites c.shape (ABI.flags c) loc.1) :
    (runFP c gf gb actual pf pb initial).backwardMemory r byte=
      (runFP c gf gb actual pf pb initial).forwardMemory r byte :=
  PhysicalLayout.publish_frame c.shape (ABI.flags c) _ _ _ r byte outside

#axiomsClean fp_equiv
#axiomsClean whole_fp_bytes
#axiomsClean fp_return_contract
#axiomsClean fp_forward_frame
#axiomsClean fp_backward_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta
