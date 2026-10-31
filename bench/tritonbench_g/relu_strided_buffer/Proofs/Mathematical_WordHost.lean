import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralHost
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WordTile
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_HostIndex32

/-!
The complete original rank-one source host model specialized to raw FP32 words.
The original strict comparison, both explicit same-dtype casts and every source
branch/loop/CTA execute in StructuralHost. ZeroLiteralLaw is the only remaining
scalar primitive premise in the word result. The separate prepared-access
theorem derives finite index/mask/byte expressions from the same host arguments.
Native operand/instruction binding, byte-memory representation and concurrent
runtime conformance remain external; this helper does not assert those facts.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.WordHost
open VeriTile.Triton FP.Structural SignedView SignedGrid HostPlan WordTile
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Both source branches use these same prepared arguments. Signed-i32 index
and mask equality covers inactive lanes too; byte equality is required only
where the source actually loads or stores. -/
theorem prepared_accesses (input output : Buffer)
    {n inputCapacity outputCapacity pid j lane : Nat} (hn : 0 < n)
    (nBound : n < SignedIndex32.indexLimit) (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity)
    (inputBase outputBase : Nat)
    (inputAllocation : SignedByteAddress.AllocationDomain inputBase inputCapacity)
    (outputAllocation : SignedByteAddress.AllocationDomain outputBase outputCapacity)
    (launched : pid < (rankOneArguments input output n sx sy).C)
    (iteration : j < (rankOneArguments input output n sx sy).T)
    (inTile : lane < (rankOneArguments input output n sx sy).B) :
    let a := rankOneArguments input output n sx sy
    let logical := CappedGrid.logicalIndex a.B a.C pid j lane
    let index := SignedIndex32.index32 a.B a.C pid j lane
    index.toInt = (logical : Int) ∧
      SignedIndex32.mask32 n a.B a.C pid j lane = decide (logical < n) ∧
      (logical < n → ∀ byte, byte < 4 →
        (SignedByteAddress.dataPtr64 inputBase a.input + index.signExtend 64 *
          BitVec.ofInt 64 sx * 4 + BitVec.ofNat 64 byte).toNat =
          inputBase + 4 * a.input.address logical + byte ∧
        (SignedByteAddress.dataPtr64 outputBase a.output + index.signExtend 64 *
          BitVec.ofInt 64 sy * 4 + BitVec.ofNat 64 byte).toNat =
          outputBase + 4 * a.output.address logical + byte) := by
  have extent := HostIndex32.prepared_extent input output sx sy hn nBound
  have controls := HostIndex32.prepared_index_and_mask input output sx sy hn nBound
    launched iteration inTile
  refine ⟨controls.1, controls.2.1, ?_⟩
  intro active byte inElement
  exact ⟨SignedIndex32.active_byte_address_exact inputBase (input.view sx) hin inputAllocation
      extent launched iteration inTile active inElement,
    SignedIndex32.active_byte_address_exact outputBase (output.view sy) hout outputAllocation
      extent launched iteration inTile active inElement⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.WordHost
