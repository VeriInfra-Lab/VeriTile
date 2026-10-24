import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Contract
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.NativeHost
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Index
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Bytes

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Physical
open VeriTile Triton FP.Structural Contract
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
namespace Raw
abbrev State := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.Memory
abbrev read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.read
abbrev ops := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.ops
abbrev unpack := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.unpack
abbrev byte_of_read := VeriTile.Bench.Repairs.AdamUpdateTriton.Bytes.byte_of_read
end Raw
variable {σ C : Type} {mem : σ → RegionName → Nat → C}

/-- Ordinary input/allocator metadata. A contiguous window retains its complete
original backing identity while beginning at the tensor's actual data pointer. -/
structure TensorWindow (tensor : TensorInfo) where
  backing : Allocation
  window : Allocation
  backingId : backing.object=tensor.backing
  windowId : window.object=backing.object
  owner : window.owner=tensor.device.index
  backingOwner : backing.owner=window.owner
  pointer : window.baseByte=tensor.dataPtr.toNat
  extent : window.bytes=4*tensor.numel
  lower : backing.baseByte≤window.baseByte
  upper : window.baseByte+window.bytes≤backing.baseByte+backing.bytes

structure Binding {σ C : Type} {mem : σ → RegionName → Nat → C} (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps)
    (call : Host.Preparation σ C mem (request x w eps ok)) where
  output : TensorInfo
  inputWindow : TensorWindow x
  weightWindow : TensorWindow w
  outputWindow : TensorWindow output
  outputShape : output.shape=x.shape
  outputFP32 : output.dtype=.floating .fp32
  outputDevice : output.device=x.device
  outputContiguous : output.isContiguous=Bool.true
  outputNegative : output.isNeg=Bool.false
  outputObjectFresh : output.object≠x.object ∧ output.object≠w.object
  outputBackingFresh : output.backing≠x.backing ∧ output.backing≠w.backing
  inputRegion : call.input.region="x"
  weightRegion : call.weight.region="w"
  outputRegion : call.output.region="y"
  descriptorDevice : call.input.device=x.device.index
  /-- The kernel flatten view has the original pointer/backing, no copy. -/
  kernelInput : TensorInfo
  viewShape : kernelInput.shape=[(request x w eps ok).rows,(request x w eps ok).columns]
  viewPointer : kernelInput.dataPtr=x.dataPtr
  viewBacking : kernelInput.backing=x.backing
  viewDevice : kernelInput.device=x.device
  /-- Fresh output storage is disjoint from the entire original input backings. -/
  inputSeparate : inputWindow.backing.separate outputWindow.window
  weightSeparate : weightWindow.backing.separate outputWindow.window

/-- Entry words are decoded from actual four-byte source windows. Aliased read
roles are permitted and automatically read the same bytes when addresses agree. -/
def memory {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) (s : Raw.State) (r : RegionName) (i : Nat) : Native.Word :=
  if r="x" then Raw.read s (b.inputWindow.window.key i)
  else if r="w" then Raw.read s (b.weightWindow.window.key i)
  else Raw.read s (b.outputWindow.window.key i)

theorem outputSize {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) :
    b.outputWindow.window.words=(request x w eps ok).size := by
  rw [Allocation.words,b.outputWindow.extent]
  have shape : b.output.numel=x.numel := by simp [Torch.TensorMetadata.numel,b.outputShape]
  rw [shape,(request_facts x w eps ok).2.2.2]
  omega

/-- Physical realization of the source's already executed output stores.
This is a byte-level source-memory interpretation, not a Python copyback stage. -/
def sourceWrites (q : Host.Request) (output : Allocation) (words : Nat → Native.Word) (s : Raw.State) : Raw.State :=
  stores Raw.ops output.key words (List.range q.size) s

theorem source_read (q : Host.Request) (output : Allocation) (words : Nat → Native.Word)
    (s : Raw.State) (i : Nat) (active : i<q.size) :
    Raw.read (sourceWrites q output words s) (output.key i)=words i := by
  apply readback Raw.ops output.key words (List.range q.size) s i (List.mem_range.mpr active)
  · intro j member equal
    have eq:=congrArg Key.address equal
    simp only [Allocation.key] at eq
    have : j=i := by omega
    rw [this]
  · intro j member different
    apply Allocation.indices_disjoint
    intro same
    exact different (congrArg output.key same.symm)

theorem source_bytes (q : Host.Request) (output : Allocation) (words : Nat → Native.Word)
    (s : Raw.State) (i : Nat) (active : i<q.size) (byte : Fin 4) :
    sourceWrites q output words s output.owner ((output.key i).address+byte.val)=Raw.unpack (words i) byte := by
  exact (Raw.byte_of_read (sourceWrites q output words s) (output.key i) byte).trans
    (congrArg (fun word => Raw.unpack word byte) (source_read q output words s i active))

theorem source_frame (q : Host.Request) (output : Allocation) (words : Nat → Native.Word)
    (s : Raw.State) (owner address : Nat) (outside : ∀ i,i<q.size → ¬(output.key i).covers owner address) :
    sourceWrites q output words s owner address=s owner address :=
  bytes_frame Raw.ops output.key words (List.range q.size) s owner address (fun i hi => outside i (List.mem_range.mp hi))

structure Result where
  tensor : TensorInfo
  bytes : Raw.State
  source : Host.Result (State Native.Term)

def invoke (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation (State Native.Term) (Cell Native.Term) State.mem (request x w eps ok)} (b : Binding x w eps ok call) (prepared : Raw.State) : Option Result :=
  (Host.invoke (Native.algebra R) (request x w eps ok) call).map fun result =>
    ⟨b.output,sourceWrites (request x w eps ok) b.outputWindow.window
      (Native.observe I S result.state call.output.region) prepared,result⟩

/-- The allocation step may change only the fresh output window. This ordinary
allocator contract mentions no kernel value, successful execution or formula. -/
def AllocationFrame {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) (before prepared : Raw.State) : Prop :=
  ∀ owner address,(∀ i,i<(request x w eps ok).size → ¬(b.outputWindow.window.key i).covers owner address) →
    prepared owner address=before owner address

theorem whole_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (x w : TensorInfo) (eps : Epsilon) (ok : Admitted x w eps)
    (call : Host.Preparation (State Native.Term) (Cell Native.Term) State.mem (request x w eps ok))
    (b : Binding x w eps ok call) (before prepared : Raw.State)
    (entry : call.before=Native.entry (memory b before)) (allocation : AllocationFrame b before prepared) :
    ∃ result,invoke I R S b prepared=some result ∧ result.tensor=b.output ∧
      result.tensor.shape=x.shape ∧ result.tensor.dtype=.floating .fp32 ∧ result.tensor.device=x.device ∧
      (∀ row,row<(request x w eps ok).rows → ∀ col,col<(request x w eps ok).columns →
        Raw.read result.bytes (b.outputWindow.window.key (row*(request x w eps ok).columns+col))=
          Native.value I R S (memory b before) call.input.region call.weight.region (request x w eps ok) row col) ∧
      (∀ row,row<(request x w eps ok).rows → ∀ col,col<(request x w eps ok).columns → ∀ byte : Fin 4,
        result.bytes b.outputWindow.window.owner
          ((b.outputWindow.window.key (row*(request x w eps ok).columns+col)).address+byte.val)=
          Raw.unpack (Native.value I R S (memory b before) call.input.region call.weight.region
            (request x w eps ok) row col) byte) ∧
      (∀ owner address,(∀ i,i<(request x w eps ok).size → ¬(b.outputWindow.window.key i).covers owner address) →
        result.bytes owner address=before owner address) := by
  let q:=request x w eps ok
  obtain ⟨source,run,tensor,shape,dtype,device,values,frame⟩:=Native.whole_native I R S q (memory b before) call entry
  refine ⟨⟨b.output,sourceWrites q b.outputWindow.window (Native.observe I S source.state call.output.region) prepared,source⟩,
    by simp [invoke,run,q],rfl,b.outputShape,b.outputFP32,b.outputDevice,?_,?_,?_⟩
  · intro row hr col hc
    rw [source_read q _ _ _ _ (Index.active_bound q row col hr hc)]
    rw [←tensor]
    exact values row hr col hc
  · intro row hr col hc byte
    exact (source_bytes q b.outputWindow.window (Native.observe I S source.state call.output.region)
      prepared (row*q.columns+col) (Index.active_bound q row col hr hc) byte).trans
      (congrArg (fun word => Raw.unpack word byte) (by simpa [tensor] using values row hr col hc))
  · intro owner address outside
    exact (source_frame q _ _ _ owner address outside).trans (allocation owner address outside)

theorem backing_outside {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call)
    (backing : Allocation) (separate : backing.separate b.outputWindow.window)
    (owner address : Nat) (inside : backing.covers owner address) :
    ∀ i,i<(request x w eps ok).size → ¬(b.outputWindow.window.key i).covers owner address := by
  intro i hi overlap
  have contained:=Allocation.key_contains b.outputWindow.window i (by rw [outputSize b];exact hi)
  rcases inside with ⟨ownerEq,lo,upper⟩
  rcases overlap with ⟨otherOwner,otherLo,otherHi⟩
  rcases separate with different|before|after
  · exact different (ownerEq.symm.trans otherOwner)
  · have start:=contained.2.1
    omega
  · have finish:=contained.2.2
    omega

theorem original_backings_frame {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call)
    (before after : Raw.State)
    (frame : ∀ owner address,(∀ i,i<(request x w eps ok).size → ¬(b.outputWindow.window.key i).covers owner address) →
      after owner address=before owner address) :
    (∀ owner address,b.inputWindow.backing.covers owner address → after owner address=before owner address) ∧
    (∀ owner address,b.weightWindow.backing.covers owner address → after owner address=before owner address) := by
  constructor
  · intro owner address inside
    exact frame owner address (backing_outside b _ b.inputSeparate owner address inside)
  · intro owner address inside
    exact frame owner address (backing_outside b _ b.weightSeparate owner address inside)

/-- Device context selects the target's existing stream and restores the
caller's current device/stream map on ordinary success or error outcomes. -/
structure DeviceContext where
  device : Nat
  stream : Nat → Nat

def withDevice (target : Nat) (action : Nat → Nat → ρ) (before : DeviceContext) : DeviceContext × ρ :=
  (before,action target (before.stream target))

theorem device_restored (target : Nat) (action : Nat → Nat → ρ) (before : DeviceContext) :
    (withDevice target action before).1=before := rfl

theorem launch_owner {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) :
    b.inputWindow.window.owner=x.device.index ∧ b.weightWindow.window.owner=x.device.index ∧
    b.outputWindow.window.owner=x.device.index := by
  have checks:=(python_checks_iff x w eps).mp ok
  have devices : x.device=w.device := checks.2.2.2.2.2.2.1.1
  refine ⟨b.inputWindow.owner,?_,?_⟩
  · rw [b.weightWindow.owner,←devices]
  · rw [b.outputWindow.owner,b.outputDevice]

/-- Admitted CUDA row/lane arithmetic includes every masked padded offset. -/
theorem index_words (q : Host.Request) (row col : Nat) (hr : row<q.rows) (hc : col<16384) :
    ((BitVec.ofNat 32 row)*(BitVec.ofNat 32 q.columns)+BitVec.ofNat 32 col).toInt=(row*q.columns+col:Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  have bound:=Index.padded_index_bound q row col hr hc
  have nat : (BitVec.ofNat 32 (row*q.columns+col)).toNat=row*q.columns+col := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,nat]
  · push_cast
    rfl
  · rw [nat]
    norm_num at bound ⊢
    omega

theorem output_pointer {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) (row col : Nat)
    (hr : row<(request x w eps ok).rows) (hc : col<(request x w eps ok).columns) :
    (b.outputWindow.window.key (row*(request x w eps ok).columns+col)).address+4≤2^64 := by
  have index:=Index.active_bound (request x w eps ok) row col hr hc
  have backed:=Allocation.key_contains b.outputWindow.window _ (by rw [outputSize b];exact index)
  exact backed.2.2.trans b.outputWindow.window.representable

theorem input_pointer {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) (row col : Nat)
    (hr : row<(request x w eps ok).rows) (hc : col<(request x w eps ok).columns) :
    (b.inputWindow.window.key (row*(request x w eps ok).columns+col)).address+4≤2^64 := by
  have index:=Index.active_bound (request x w eps ok) row col hr hc
  have capacity : b.inputWindow.window.words=(request x w eps ok).size := by
    rw [Allocation.words,b.inputWindow.extent,←(request_facts x w eps ok).2.2.2]
    omega
  have backed:=Allocation.key_contains b.inputWindow.window _ (by rw [capacity];exact index)
  exact backed.2.2.trans b.inputWindow.window.representable

theorem weight_pointer {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation σ C mem (request x w eps ok)} (b : Binding x w eps ok call) (col : Nat)
    (hc : col<(request x w eps ok).columns) :
    (b.weightWindow.window.key col).address+4≤2^64 := by
  have checks:=(python_checks_iff x w eps).mp ok
  have count : w.numel=width x := checks.2.2.2.2.2.2.2.1.2.2
  have capacity : b.weightWindow.window.words=(request x w eps ok).columns := by
    rw [Allocation.words,b.weightWindow.extent,count,(request_facts x w eps ok).2.1]
    omega
  have backed:=Allocation.key_contains b.weightWindow.window col (by rw [capacity];exact hc)
  exact backed.2.2.trans b.weightWindow.window.representable

def hostInvoke (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation (State Native.Term) (Cell Native.Term) State.mem (request x w eps ok)}
    (b : Binding x w eps ok call) (prepared : Raw.State) (context : DeviceContext) : DeviceContext × Option Result :=
  if (request x w eps ok).size=0 then (context,invoke I R S b prepared)
  else withDevice x.device.index (fun _ _ => invoke I R S b prepared) context

theorem hostInvoke_result (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation (State Native.Term) (Cell Native.Term) State.mem (request x w eps ok)}
    (b : Binding x w eps ok call) (prepared : Raw.State) (context : DeviceContext) :
    hostInvoke I R S b prepared context=(context,invoke I R S b prepared) := by
  simp [hostInvoke,withDevice]

theorem empty_bytes (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    {x w : TensorInfo} {eps : Epsilon} {ok : Admitted x w eps}
    {call : Host.Preparation (State Native.Term) (Cell Native.Term) State.mem (request x w eps ok)}
    (b : Binding x w eps ok call) (prepared : Raw.State) (empty : (request x w eps ok).rows=0) :
    invoke I R S b prepared=some ⟨b.output,prepared,⟨call.output,call.prepared⟩⟩ := by
  simp [invoke,Host.empty_return _ _ call empty,sourceWrites,Host.Request.size,empty,stores]

/-- Even interrupted stores are local four-byte writes to output-owned cells. -/
def interrupted (q : Host.Request) (output : Allocation) (writes : List (Fin q.size × Native.Word)) (s : Raw.State) :=
  stores Raw.ops (fun i => output.key i.1.val) Prod.snd writes s

theorem partial_frame (q : Host.Request) (output : Allocation) (writes : List (Fin q.size × Native.Word))
    (s : Raw.State) (owner address : Nat) (outside : ∀ i,i<q.size → ¬(output.key i).covers owner address) :
    interrupted q output writes s owner address=s owner address :=
  bytes_frame Raw.ops _ _ writes s owner address (fun i _ => outside i.1.val i.1.isLt)

#axiomsClean whole_native
#axiomsClean original_backings_frame
#axiomsClean device_restored
#axiomsClean launch_owner
#axiomsClean index_words
#axiomsClean output_pointer
#axiomsClean input_pointer
#axiomsClean weight_pointer
#axiomsClean hostInvoke_result
#axiomsClean empty_bytes
#axiomsClean partial_frame
#specNonCircular Binding avoiding [invoke,Host.invoke,Grid.run,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Physical
