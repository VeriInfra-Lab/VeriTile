import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Physical
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPReturned
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Physical
open VeriTile Triton FP.Structural
open FPStorage (project)
noncomputable section
variable {α : Type} [Inhabited α]

theorem finite_address (abi : ABI c) (r : RegionName) (hr : tracked r) (i : Nat) (hi : i<size c r) :
    (abi.alloc r).base+i<abi.capacity := by
  have hl := abi.lengths r hr
  have hb := abi.finite r hr
  omega

theorem producer_preserves_input (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (hf : ProducerFresh abi) (scale : ℝ) (physical : Nat → α) (r : RegionName)
    (hr : tracked r) (hra : r≠"attention") (hrs : r≠"scratch") (i : Nat) (hi : i<(abi.alloc r).length) :
    producerReference M D abi scale physical ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i) := by
  unfold producerReference producerCommit
  split
  · exact commit_preserves abi "attention" r hf.1 hr (Ne.symm hra) _ physical i hi
  · rename_i hn
    have hs : Fresh abi "scratch" := hf.2 (by omega)
    rw [commit_preserves abi "attention" r hf.1 hr (Ne.symm hra) _ _ i hi,
      commit_preserves abi "scratch" r hs hr (Ne.symm hrs) _ physical i hi]

theorem consumer_preserves_input (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c)
    (hf : ConsumerFresh abi) (scale : ℝ) (physical : Nat → α) (r : RegionName)
    (hr : tracked r) (hro : r≠"o") (i : Nat) (hi : i<(abi.alloc r).length) :
    consumerReference M D abi scale physical ((abi.alloc r).base+i)=physical ((abi.alloc r).base+i) :=
  commit_preserves abi "o" r hf hr (Ne.symm hro) _ physical i hi

theorem consumer_tensor (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    consumerReference M D abi scale physical ((abi.alloc "o").base+h.val*(c.T*c.V)+t.val*c.V+v.val)=
      FPReturned.value M D c.consumer scale (project abi.alloc physical) h t v := by
  have hb : h.val*(c.T*c.V)+t.val*c.V+v.val<Host.outputSize c := by
    simpa only [Host.outputSize,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc]
      using Geometry.attention_bound h.isLt t.isLt v.isLt
  rw [show (abi.alloc "o").base+h.val*(c.T*c.V)+t.val*c.V+v.val=
      (abi.alloc "o").base+(h.val*(c.T*c.V)+t.val*c.V+v.val) by omega,
    consumer_readback M D abi scale physical _ hb]
  exact FPReturned.consumer_readback M D c scale _ h t v

theorem calls_compose (M : Algebra α) (D : FPOutput.DotPrimitive α) (abi : ABI c) (scale : ℝ)
    (physical : Nat → α) :
    (producerExecute M D abi scale physical).bind (consumerExecute M D abi scale)=
      some (consumerReference M D abi scale (producerReference M D abi scale physical)) := by
  rw [producer_source]
  exact consumer_source M D abi scale _

/-- FP32 word addressing refines the ordinary aligned byte ABI. -/
def byteAddress (word lane : Nat) := 4*word+lane
theorem byte_address_injective {w z i j : Nat} (hi : i<4) (hj : j<4)
    (he : byteAddress w i=byteAddress z j) : w=z ∧ i=j := by
  apply Geometry.packed_injective hi hj
  simpa [byteAddress,Nat.mul_comm] using he
theorem byte_address_bound {capacity word lane : Nat} (hw : word<capacity) (hl : lane<4) :
    byteAddress word lane<4*capacity := by
  simpa [byteAddress,Nat.mul_comm] using Geometry.packed_lt hw hl
theorem element_pointer (base off : Nat) : 4*(base+off)=4*base+4*off := by omega

#axiomsClean producer_preserves_input
#axiomsClean consumer_preserves_input
#axiomsClean consumer_tensor
#axiomsClean calls_compose
#axiomsClean byte_address_injective
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Physical
