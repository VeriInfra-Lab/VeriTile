import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Native

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 4000000

def readBits (heap : ByteMemory) (p : Nat) : VeriTile.Torch.FP32.Word := ⟨load32 heap p⟩

theorem byte_codec (heap : ByteMemory) (p : Nat) (word : Word)
    (bytes : ∀ i : Fin 4,heap (p+i.val)=byte word i) : load32 heap p=word := by
  unfold load32
  have h0 := bytes ⟨0,by decide⟩
  have h1 := bytes ⟨1,by decide⟩
  have h2 := bytes ⟨2,by decide⟩
  have h3 := bytes ⟨3,by decide⟩
  simp only [Nat.add_zero] at h0
  rw [h0,h1,h2,h3]
  apply BitVec.eq_of_toNat_eq
  have bound := word.isLt
  simp only [byte,BitVec.toNat_ofNat]
  norm_num at *
  omega

inductive InputRole where
  | query | key | value

def InputRole.payload (r : Request) : InputRole → Payload
  | .query => r.q | .key => r.k | .value => r.v
def InputRole.region : InputRole → RegionName
  | .query => regions.query | .key => regions.key | .value => regions.value

def inputAPI (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    (role : InputRole) → PayloadLaws (role.payload r)
  | .query => h.qAPI | .key => h.kAPI | .value => h.vAPI

theorem input_descriptor (r : Request) (heap : ByteMemory) (h : Contract r heap) (role : InputRole) :
    (role.payload r).tensor.metadata.isContiguous=Bool.true ∧
      (role.payload r).tensor.metadata.isNeg=Bool.false := by
  have all : (role.payload r).view∈Checked.all (views r) := by
    cases role <;> simp [InputRole.payload,views,Checked.all]
  have contiguous : (role.payload r).tensor.metadata.isContiguous=Bool.true := by
    cases role with
    | query => exact (h.checks.1.1 (views r).q (by simp)).2.2.1
    | key => exact (h.checks.1.2.1 (views r).k (by simp)).2.2.1
    | value => exact (h.checks.1.2.1 (views r).v (by simp)).2.2.1
  exact ⟨contiguous,(h.checks.1.2.2.2.2.2.1 _ all).2.2.1⟩

/-- All payload leaves are the actual shared Torch Tensor logical words.
Aliased payload/metadata/read roles observe the same original physical heap. -/
theorem logical_input (r : Request) (heap : ByteMemory) (h : Contract r heap) (role : InputRole)
    (i : Nat) (hi : i < (role.payload r).tensor.metadata.numel) :
    ((initial r heap).mem role.region i).read .real =
      NativeGraph.Expr.leaf
        ((VeriTile.Torch.FP32.logicalValue (readBits heap) (role.payload r).tensor (inputAPI r heap h role).tensor i).bits) := by
  have attrs := input_descriptor r heap h role
  rw [←logical_fp32 (readBits heap) (role.payload r) (inputAPI r heap h role) attrs.2 i,
    logical_direct (readBits heap) (role.payload r) (inputAPI r heap h role) attrs.1 i hi]
  cases role <;> simp [initial,base,regions,InputRole.region,InputRole.payload,readBits]

theorem logical_output (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (ops : NativeInstructions.ScalarOps) (after : ByteMemory)
    (idx : GridIndex r.call.grid) (i : TileIndex [r.call.layout.BM,r.call.layout.D])
    (on : r.call.active idx i)
    (bytes : ∀ j : Fin 4,after (address r (idx,i,j))=byte (referenceWord r ops heap (idx,i,j)) j) :
    (VeriTile.Torch.FP32.logicalValue (readBits after) r.out.tensor h.outAPI.tensor (flatIndex r.call idx i)).bits=
      referenceWord r ops heap (idx,i,⟨0,by decide⟩) := by
  have descriptor := h.checks.1.1 (views r).o (by simp)
  have attrs := h.checks.1.2.2.2.2.2.1 (views r).o (by simp [Checked.all])
  have shape : r.out.tensor.metadata.shape=[r.dimensions.queryRows,r.call.layout.heads,r.call.layout.D] := descriptor.1
  have contiguous : r.out.tensor.metadata.isContiguous=Bool.true := descriptor.2.2.1
  have negative : r.out.tensor.metadata.isNeg=Bool.false := attrs.2.2.1
  have flat := flat_index r.call r.dimensions (extent r) h.checks.2.1 idx i on
  have bound : flatIndex r.call idx i<r.out.tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using flat.2
  rw [←logical_fp32 (readBits after) r.out h.outAPI negative,
    logical_direct (readBits after) r.out h.outAPI contiguous _ bound]
  change load32 after (r.out.tensor.dataPtr.toNat+4*flatIndex r.call idx i)=_
  apply byte_codec
  intro j
  rw [←flat.1]
  exact bytes j

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
