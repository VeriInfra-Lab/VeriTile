import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Returned
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Tensor

namespace VeriTile.Bench.Repairs.ContextAttnNopad.CheckedTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Bytes Returned
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _
abbrev read := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.read

/-- Nominal source-role names, independent of physical read-storage aliases. -/
def LogicalRoles (r : Regions) : Prop :=
  ∀ rg∈[r.query,r.key,r.value,r.out],rg∉metadataRegions r

structure Output (v : Checked.Views) where
  tensor : Tensor
  metadata : tensor.metadata=v.o.metadata
  laws : TensorLaws tensor

def flatIndex (c : Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  ((c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val))*c.q.heads+(c.launch idx).head)*c.q.dims+i.2.1.val

theorem flat_index (c : Call) (r : Regions) (extent : RegionBounds)
    (g : Checked.geometryChecks c r extent) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (on : FPReference.laneActive c idx i) :
    FPReference.laneOffset c idx i=flatIndex c idx i ∧ flatIndex c idx i<c.q.rows*c.q.heads*c.q.dims := by
  rcases g with ⟨_,_,k,v,o,contiguous,capacity,_⟩
  have bounds := (capacity ⟨(c.launch idx).batch,launch_batch_lt c idx⟩).1
  change c.start (c.launch idx).batch+c.length (c.launch idx).batch≤c.q.rows at bounds
  have row : c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val)<c.q.rows := by
    change _ < c.length _ at on
    omega
  have bounded := Checked.bounded c.q contiguous (c.q.rows*c.q.heads*c.q.dims) (le_refl _) _ _ _
    row (launch_head_lt c idx) i.2.1.isLt
  have eq := Checked.offset_eq c.q contiguous
    (c.start (c.launch idx).batch+((c.launch idx).block*128+i.1.val)) (c.launch idx).head i.2.1.val
  constructor
  · simpa only [FPReference.laneOffset,o,flatIndex] using eq
  · simpa only [eq,flatIndex] using bounded

theorem logical_output (c : Call) (r : Regions) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c v) (geometry : Checked.geometryChecks c r extent)
    (storage : Checked.StorageBinding r v B) (out : Output v)
    (ops : NativeInstructions.ScalarOps) (before after : Memory)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) (on : FPReference.laneActive c idx i)
    (bytes : ∀ j : Fin 4,after (address c r B (idx,i,j))=byte (referenceWord c r B ops before (idx,i,j)) j) :
    (logicalValue (read after) out.tensor out.laws (flatIndex c idx i)).bits=
      referenceWord c r B ops before (idx,i,⟨0,by omega⟩) := by
  have descriptor := tensors.1 v.o (by simp)
  have attrs := tensors.2.2.2.1 v.o (by simp [Checked.all])
  have index := flat_index c r extent geometry idx i on
  have bound : flatIndex c idx i < out.tensor.metadata.numel := by
    rw [out.metadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,descriptor.1,Nat.mul_assoc] using index.2
  rw [← VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct out.tensor out.laws after
    (by simpa [out.metadata] using descriptor.2.2.1) (by simpa [out.metadata] using attrs.2.2.1) _ bound]
  change load32 after (out.tensor.metadata.dataPtr.toNat+4*flatIndex c idx i)=_
  rw [out.metadata,←storage.base r.out v.o (by simp [Checked.roleViews]),←index.1]
  apply VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.byte_codec
  intro j
  exact bytes j

/-- Direct contiguous input Tensor leaves use the original supplied storage;
no scratch/copyback or arbitrary logical offset is part of this interface. -/
structure Input (x : Checked.View) where
  tensor : Tensor
  metadata : tensor.metadata=x.metadata
  laws : TensorLaws tensor

theorem logical_source (c : Call) (r : Regions) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c v) (storage : Checked.StorageBinding r v B)
    (rg : RegionName) (x : Checked.View) (role : (rg,x)∈Checked.roleViews r v)
    (payload : x∈[v.q,v.k,v.v]) (metadataLabel : rg∉metadataRegions r)
    (input : Input x) (before : Memory) (i : Nat) (bound : i < input.tensor.metadata.numel) :
    ((initial c r B before).mem rg i).read .real=
      NativePhysical.Expr.leaf .fp32 ((logicalValue (read before) input.tensor input.laws i).bits) := by
  have member : x∈[v.q,v.k,v.v,v.o] := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at payload ⊢
    tauto
  have all : x∈Checked.all v := by
    simp only [Checked.all,List.mem_cons,List.not_mem_nil,or_false] at *
    tauto
  have descriptor := tensors.1 x member
  have attrs := tensors.2.2.2.1 x all
  simp only [initial,if_neg metadataLabel,Cell.read_same]
  rw [storage.base rg x role,←input.metadata]
  have eq := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct input.tensor input.laws before
    (by simpa [input.metadata] using descriptor.2.2.1) (by simpa [input.metadata] using attrs.2.2.1) i bound
  exact congrArg (fun word : VeriTile.Torch.FP32.Word => NativePhysical.Expr.leaf .fp32 word.bits) eq

/-- Canonical whole-task FP32 contract: supplied Tensor, independent fixed-MMA
reference, entire native/source grid, native projection, full byte/read-role
frames, empty calls, None and restored owning-device context. -/
theorem whole_tensor_fp32 (c : Call) (r : Regions) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (start length : Nat → Int)
    (checked : Checked.checkInputs c r v A.extent requestedMax start length=Bool.true)
    (before : Memory) (observed : Observed c r B before start length)
    (storage : Checked.StorageBinding r v B) (sizes : Checked.AllocationSizes r v A)
    (labels : Checked.Labels r) (roles : LogicalRoles r)
    (ops : NativeInstructions.ScalarOps) (out : Output v) (callerDevice : Nat) :
    let ctx : Context := ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
    MachineFacts c r v A B before ∧
    ∃ result,invoke c r B ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=out.tensor.object ∧ result.currentDevice=callerDevice ∧
      result.launchDevice=(if Needed c then some v.q.metadata.device.index else none) ∧
      FPReference.runGrid c r (backend c) (order c) (initial c r B before)=some result.graph ∧
      (∀ idx i,FPReference.laneActive c idx i →
        (logicalValue (read result.bytes) out.tensor out.laws (flatIndex c idx i)).bits=
          referenceWord c r B ops before (idx,i,⟨0,by omega⟩)) ∧
      (∀ a,(¬∃ i : Lane c,active c i ∧ address c r B i=a) → result.bytes a=before a) ∧
      (∀ rg∈readRegions r,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (¬Needed c → result.bytes=before) := by
  dsimp only
  have guard := (Checked.checkInputs_iff c r v A.extent requestedMax start length).mp checked
  have native := machine_facts c r v A B requestedMax start length checked before observed storage sizes labels
  obtain ⟨result,run,ret,obj,device,launch,source,referenceEq,bytes,frame,inputs,graphFrame,empty⟩ :=
    whole_bytes c r v A B requestedMax start length checked before observed storage labels ops
      ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
  refine ⟨native,result,run,ret,obj,device,launch,source,?_,frame,inputs,empty⟩
  intro idx i on
  exact logical_output c r v B guard.1 guard.2.1 storage out ops before result.bytes idx i on
    (fun j => bytes (idx,i,j) on)

#axiomsClean logical_source
#axiomsClean whole_tensor_fp32
end
end VeriTile.Bench.Repairs.ContextAttnNopad.CheckedTensor
