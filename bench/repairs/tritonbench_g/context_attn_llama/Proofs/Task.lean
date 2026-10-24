import bench.repairs.tritonbench_g.context_attn_llama.Proofs.Returned
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Tensor

namespace VeriTile.Bench.Repairs.ContextAttnLlama.CheckedTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
open Bytes Returned
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _
abbrev read := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.read

/-- Nominal source-role names, independent of physical read-storage aliases. -/
def LogicalRoles (c : Call) : Prop :=
  ∀ rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out],rg∉metadataRegions c

structure Output (v : Checked.Views) where
  tensor : Tensor
  metadata : tensor.metadata=v.o.metadata
  laws : TensorLaws tensor

def flatIndex (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  ((c.start (Formula.batch c.layout (c.launchAt idx))+Formula.queryRow c.layout (c.launchAt idx) i.1.val)*c.layout.heads+
    Formula.head c.layout (c.launchAt idx))*c.layout.D+i.2.1.val

theorem flat_index (c : Call) (d : Checked.Dimensions) (extent : RegionBounds)
    (g : Checked.geometryChecks c d extent) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D])
    (on : c.active idx i) : c.offset idx i=flatIndex c idx i ∧ flatIndex c idx i < d.queryRows*c.layout.heads*c.layout.D := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Formula.batch c.layout (c.launchAt idx),Host.batch_bound c idx⟩).1
  have row : c.start (Formula.batch c.layout (c.launchAt idx))+Formula.queryRow c.layout (c.launchAt idx) i.1.val < d.queryRows := by
    change c.start (Formula.batch c.layout (c.launchAt idx))+c.length (Formula.batch c.layout (c.launchAt idx))≤d.queryRows at cap
    change Formula.queryRow _ _ _ < c.length _ at on
    omega
  have bound := Checked.bound3 d.queryRows c.layout.heads c.layout.D _ _ _ _ (le_refl _)
    row (Host.head_bound c idx) i.2.1.isLt
  rcases g.1 with ⟨_,_,_,orow,ohead,odim,_⟩
  have eq : c.offset idx i=flatIndex c idx i := by
    simp only [Call.offset,Call.output,orow,ohead,odim,Nat.mul_one,flatIndex]
    ring
  exact ⟨eq,by simpa only [flatIndex,Nat.add_mul,Nat.mul_assoc] using bound⟩

theorem logical_output (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c d v) (geometry : Checked.geometryChecks c d extent)
    (storage : Checked.StorageBinding c v B) (out : Output v)
    (ops : NativeInstructions.ScalarOps) (before after : Memory)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) (on : c.active idx i)
    (bytes : ∀ j : Fin 4,after (address c B (idx,i,j))=byte (referenceWord c B ops before (idx,i,j)) j) :
    (logicalValue (read after) out.tensor out.laws (flatIndex c idx i)).bits=
      referenceWord c B ops before (idx,i,⟨0,by omega⟩) := by
  have descriptor := tensors.1 v.o (by simp)
  have attrs := tensors.2.2.2.2.2.1 v.o (by simp [Checked.all])
  have index := flat_index c d extent geometry idx i on
  have bound : flatIndex c idx i < out.tensor.metadata.numel := by
    rw [out.metadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,descriptor.1,Nat.mul_assoc] using index.2
  rw [←VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct out.tensor out.laws after
    (by simpa [out.metadata] using descriptor.2.2.1) (by simpa [out.metadata] using attrs.2.2.1) _ bound]
  change load32 after (out.tensor.metadata.dataPtr.toNat+4*flatIndex c idx i)=_
  rw [out.metadata,←storage.base c.regions.out v.o (by simp [Checked.roleViews]),←index.1]
  apply VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.byte_codec
  intro j
  exact bytes j

structure Input (x : Checked.View) where
  tensor : Tensor
  metadata : tensor.metadata=x.metadata
  laws : TensorLaws tensor

theorem logical_source (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c d v) (storage : Checked.StorageBinding c v B)
    (rg : RegionName) (x : Checked.View) (role : (rg,x)∈Checked.roleViews c v)
    (payload : x∈[v.q,v.k,v.v]) (metadataLabel : rg∉metadataRegions c)
    (input : Input x) (before : Memory) (i : Nat) (bound : i < input.tensor.metadata.numel) :
    ((initial c B before).mem rg i).read .real=
      VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.Expr.leaf ((logicalValue (read before) input.tensor input.laws i).bits) := by
  have contig : x.metadata.isContiguous=Bool.true := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at payload
    rcases payload with rfl|rfl|rfl
    · exact (tensors.1 v.q (by simp)).2.2.1
    · exact (tensors.2.1 v.k (by simp)).2.2.1
    · exact (tensors.2.1 v.v (by simp)).2.2.1
  have attrs := tensors.2.2.2.2.2.1 x (Checked.role_member c v rg x role)
  simp only [initial,if_neg metadataLabel,Cell.read_same]
  rw [storage.base rg x role,←input.metadata]
  have eq := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct input.tensor input.laws before
    (by simpa [input.metadata] using contig) (by simpa [input.metadata] using attrs.2.2.1) i bound
  exact congrArg (fun word : VeriTile.Torch.FP32.Word => VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph.Expr.leaf word.bits) eq

/-- Complete direct FP32 call: supplied logical Tensor equals the independent
fixed-instruction reference, with all native/source grids, formed pointers,
output/read frames, empty-call behavior, None and restored device context. -/
theorem whole_tensor_fp32 (c : Call) (d : Checked.Dimensions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (checked : Checked.checkInputs c d v A.extent requestedMax start sequence prompt request pages=Bool.true)
    (before : Memory) (observed : Observed c B before start sequence prompt request pages)
    (storage : Checked.StorageBinding c v B) (labels : Checked.Labels c) (roles : LogicalRoles c)
    (ops : NativeInstructions.ScalarOps) (out : Output v) (callerDevice : Nat) :
    let ctx : Context := ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
    MachineFacts c d v A B before ∧
    ∃ result,invoke c B ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=out.tensor.object ∧ result.currentDevice=callerDevice ∧
      result.launchDevice=(if Needed c then some v.q.metadata.device.index else none) ∧
      FPSuccess.runGrid c (backend c) (order c) (initial c B before)=some result.graph ∧
      (∀ idx i,c.active idx i → (logicalValue (read result.bytes) out.tensor out.laws (flatIndex c idx i)).bits=
        referenceWord c B ops before (idx,i,⟨0,by omega⟩)) ∧
      (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → result.bytes a=before a) ∧
      (∀ rg∈c.inputs,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (¬Needed c → result.bytes=before) := by
  dsimp only
  have guard := (Checked.checkInputs_iff c d v A.extent requestedMax start sequence prompt request pages).mp checked
  have machine := machine_facts c d v A B requestedMax start sequence prompt request pages checked before observed storage labels
  obtain ⟨result,run,ret,obj,device,launch,source,referenceEq,bytes,frame,inputs,graphFrame,empty⟩ :=
    whole_bytes c d v A B requestedMax start sequence prompt request pages checked before observed storage labels ops
      ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
  refine ⟨machine,result,run,ret,obj,device,launch,source,?_,frame,inputs,empty⟩
  intro idx i on
  exact logical_output c d v B guard.1 guard.2.1 storage out ops before result.bytes idx i on
    (fun j => bytes (idx,i,j) on)

#axiomsClean logical_source
#axiomsClean whole_tensor_fp32
end
end VeriTile.Bench.Repairs.ContextAttnLlama.CheckedTensor
