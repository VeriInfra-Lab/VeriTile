import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.Returned
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Tensor

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.CheckedTensor
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
open Bytes Returned
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _
abbrev read := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.read

def LogicalRoles (r : Regions) : Prop :=
  ∀ rg∈[r.query,r.key,r.value,r.out],rg∉Checked.metadataRegions r

structure Output (v : Checked.Views) where
  tensor : Tensor
  metadata : tensor.metadata=v.o.metadata
  laws : TensorLaws tensor

/-- Contiguous logical flat output coordinate, with source channel padding
removed by the actual active channel mask. -/
def flatIndex (c : Call) (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) : Nat :=
  ((c.input.start (Host.launchOf c.args idx).batch+queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.1.val).toNat*
    c.args.o.heads+(Host.launchOf c.args idx).head)*c.args.o.dims+i.2.1.val

theorem flat_index (c : Call) (extent : RegionBounds)
    (g : Checked.geometryChecks c.args c.regions extent) (m : Checked.metadataChecks c.args c.input)
    (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) (on : FPSuccess.laneActive c idx i) :
    FPSuccess.laneOffset c idx i=flatIndex c idx i ∧ flatIndex c idx i < c.args.o.rows*c.args.o.heads*c.args.o.dims := by
  rcases g with ⟨strides,hq,hk,grouped,qk,kv,ok,oh,vh,orows,vrows,dlo,dhi,rest⟩
  rcases strides.2.2.2.1 with ⟨sr,sh,sd⟩
  let l := Host.launchOf c.args idx
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have st : 0 ≤ c.input.start l.batch := md.1
  have cap : c.input.start l.batch+(c.input.sequence l.batch-c.input.prompt l.batch) ≤ c.args.q.rows := md.2.2.2.2.1
  have row0 : 0 ≤ c.input.start l.batch+queryRow (Host.layout c.args) l i.1.val := by positivity
  have rowBound : (c.input.start l.batch+queryRow (Host.layout c.args) l i.1.val).toNat < c.args.o.rows := by
    have active : (queryRow (Host.layout c.args) l i.1.val : Int) < c.input.sequence l.batch-c.input.prompt l.batch := on.1
    rw [orows]
    omega
  have hh : l.head < c.args.o.heads := by simpa only [oh] using Host.head_lt c.args idx
  have hd : i.2.1.val < c.args.o.dims := by simpa only [ok,←qk] using on.2
  have bound := Checked.bound3 _ _ _ _ _ _ _ (le_refl _) rowBound hh hd
  have eq : FPSuccess.laneOffset c idx i=flatIndex c idx i := by
    change ((c.input.start l.batch+queryRow (Host.layout c.args) l i.1.val)*c.args.o.rowStride+
      (l.head : Int)*c.args.o.headStride+(i.2.1.val : Int)*c.args.o.dimStride).toNat=
      ((c.input.start l.batch+queryRow (Host.layout c.args) l i.1.val).toNat*c.args.o.heads+l.head)*c.args.o.dims+i.2.1.val
    rw [sr,sh,sd]
    simp only [Nat.cast_one,mul_one]
    rw [←Int.toNat_of_nonneg row0]
    simp only [←Nat.cast_mul,←Nat.cast_add,Int.toNat_natCast]
    ring
  exact ⟨eq,by simpa only [flatIndex,Nat.add_mul,Nat.mul_assoc] using bound⟩

theorem logical_output (c : Call) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c.args v) (geometry : Checked.geometryChecks c.args c.regions extent)
    (metadata : Checked.metadataChecks c.args c.input) (storage : Checked.StorageBinding c.regions v B) (out : Output v)
    (profile : Profile) (ops : Native.WordISA) (before after : Memory)
    (idx : GridIndex (Host.grid c.args)) (i : RealIO.Lane (Host.layout c.args)) (on : FPSuccess.laneActive c idx i)
    (bytes : ∀ j : Fin 4,after (address c B (idx,i,j))=byte (referenceWord c B profile ops before (idx,i,j)) j) :
    (logicalValue (read after) out.tensor out.laws (flatIndex c idx i)).bits=
      referenceWord c B profile ops before (idx,i,⟨0,by omega⟩) := by
  have descriptor := tensors.1 (v.o,c.args.o) (by simp)
  have shape : v.o.metadata.shape=[c.args.o.rows,c.args.o.heads,c.args.o.dims] := descriptor.1
  have attrs := tensors.2.2.2.2.1 v.o (by simp [Checked.all])
  have index := flat_index c extent geometry metadata idx i on
  have bound : flatIndex c idx i < out.tensor.metadata.numel := by
    rw [out.metadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using index.2
  rw [←VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct out.tensor out.laws after
    (by simpa [out.metadata] using descriptor.2.2.2.1) (by simpa [out.metadata] using attrs.2.2.1) _ bound]
  change load32 after (out.tensor.metadata.dataPtr.toNat+4*flatIndex c idx i)=_
  rw [out.metadata,←storage.base c.regions.out v.o (by simp [Checked.roleViews]),←index.1]
  apply VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.byte_codec
  intro j
  exact bytes j

structure Input (x : Checked.View) where
  tensor : Tensor
  metadata : tensor.metadata=x.metadata
  laws : TensorLaws tensor

theorem logical_source (c : Call) (v : Checked.Views) (B : Windows extent)
    (tensors : Checked.tensorChecks c.args v) (storage : Checked.StorageBinding c.regions v B)
    (rg : RegionName) (x : Checked.View) (role : (rg,x)∈Checked.roleViews c.regions v)
    (payload : x∈[v.q,v.k,v.v]) (metadataLabel : rg∉Checked.metadataRegions c.regions)
    (input : Input x) (before : Memory) (i : Nat) (bound : i < input.tensor.metadata.numel) :
    ((initial c B before).mem rg i).read .real=
      Native.Expr.bits ((logicalValue (read before) input.tensor input.laws i).bits) := by
  have contig : x.metadata.isContiguous=Bool.true := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at payload
    rcases payload with rfl|rfl|rfl
    · exact (tensors.1 (v.q,c.args.q) (by simp)).2.2.2.1
    · exact (tensors.1 (v.k,c.args.k) (by simp)).2.2.2.1
    · exact (tensors.1 (v.v,c.args.v) (by simp)).2.2.2.1
  have attrs := tensors.2.2.2.2.1 x (Checked.role_member c.regions v rg x role)
  simp only [initial,if_neg metadataLabel,Cell.read_same]
  rw [storage.base rg x role,←input.metadata]
  have eq := VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor.direct input.tensor input.laws before
    (by simpa [input.metadata] using contig) (by simpa [input.metadata] using attrs.2.2.1) i bound
  exact congrArg (fun word : VeriTile.Torch.FP32.Word => Native.Expr.bits word.bits) eq

/-- The full direct caller Tensor is read back from the proved original-byte
stores and equals the independent native recurrence. The structural compiler
profile is used for every instruction evaluation and final store owner. -/
theorem whole_tensor_fp32 (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions) (roles : LogicalRoles c.regions)
    (profile : Profile) (launchScalar : LaunchScale.Bound c.args) (ops : Native.WordISA) (out : Output v) (callerDevice : Nat) :
    let ctx : Context := ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
    MachineFacts c v A B before ∧
    ∃ result,invoke c B profile ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=out.tensor.object ∧ result.currentDevice=callerDevice ∧
      result.launchDevice=(if Needed c then some v.q.metadata.device.index else none) ∧
      FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some result.graph ∧
      (∀ idx i,FPSuccess.laneActive c idx i → (logicalValue (read result.bytes) out.tensor out.laws (flatIndex c idx i)).bits=
        referenceWord c B profile ops before (idx,i,⟨0,by omega⟩)) ∧
      (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → result.bytes a=before a) ∧
      (∀ rg∈Checked.readRegions c.regions,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (¬Needed c → result.bytes=before) := by
  dsimp only
  have guard := (Checked.checkInputs_iff c.args c.regions v A.extent c.input requestedMax).mp checked
  obtain ⟨result,run,ret,obj,device,launch,source,referenceEq,bytes,frame,inputs,graphFrame,empty⟩ :=
    whole_bytes c v A B requestedMax checked before observed storage labels profile ops
      ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
  refine ⟨machine_facts c v A B requestedMax checked before observed storage labels,
    result,run,ret,obj,device,launch,source,?_,frame,inputs,empty⟩
  intro idx i on
  exact logical_output c v B guard.1 guard.2.1 guard.2.2.1 storage out profile ops before result.bytes idx i on
    (fun j => bytes (idx,i,j) on)

theorem whole_tensor_if_local (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions) (roles : LogicalRoles c.regions)
    (profile : Profile) (launchScalar : LaunchScale.Bound c.args) (actual refOps : Native.WordISA) (localLaws : LocalLaws actual refOps) (out : Output v) (callerDevice : Nat) :
    let ctx : Context := ⟨out.tensor.object,v.q.metadata.device.index,callerDevice⟩
    MachineFacts c v A B before ∧
    ∃ result,invoke c B profile actual ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=out.tensor.object ∧ result.currentDevice=callerDevice ∧
      result.launchDevice=(if Needed c then some v.q.metadata.device.index else none) ∧
      FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some result.graph ∧
      (∀ idx i,FPSuccess.laneActive c idx i → (logicalValue (read result.bytes) out.tensor out.laws (flatIndex c idx i)).bits=
        referenceWord c B profile refOps before (idx,i,⟨0,by omega⟩)) ∧
      (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → result.bytes a=before a) ∧
      (∀ rg∈Checked.readRegions c.regions,∀ off,off < A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (¬Needed c → result.bytes=before) := by
  have eq := localLaws.operations_eq actual refOps
  subst refOps
  exact whole_tensor_fp32 c v A B requestedMax checked before observed storage labels roles profile launchScalar actual out callerDevice

#axiomsClean whole_tensor_if_local
#axiomsClean logical_source
#axiomsClean whole_tensor_fp32
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.CheckedTensor
