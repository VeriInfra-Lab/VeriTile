import bench.repairs.tritonbench_g.flash_attn.Proofs.NativeBinding
import bench.repairs.tritonbench_g.flash_attn.Proofs.Source

namespace VeriTile.Bench.Repairs.FlashAttn.Checked
noncomputable section
set_option maxHeartbeats 3000000

/-- The same complete source launch refines the source byte execution. -/
theorem source_bytes (x : Layout) (base : Region→Nat) (native : Native.Instructions)
    (scale : Word) (memory : ByteMemory) :
    Physical.load (Physical.byteExecute x (Native.ops native) scale (sourceSchedule x.shape)
      (callerBytes base memory))=
      Source.launch x (Native.ops native) scale (Physical.load (callerBytes base memory)) := by
  rw [Source.launch_execute,Physical.execute_load]

/-- Actual caller bytes and unmodified scalar payload feed the selected SM90
graph. The returned Tensor and scratch descriptor read the exact source words;
all exterior physical bytes and all original input allocations are preserved.
The source-to-pinned-graph compiler/ISA interpretation is an ordinary boundary,
not a compiler-internals proof or a native-to-Real numerical law. -/
theorem whole_native (q k v o : Tensor) (qo ko vo oo : SortedAxes) (scale : ScalarABI)
    (causal : Bool) (owner : Nat→Nat→Nat→Fin 4) (scratch : Scratch) (callerDevice : Nat)
    (hi : inputChecks q k v qo ko vo scale=Bool.true) (ho : tensorChecks o oo=Bool.true)
    (hp : scratchPointerChecks (Host.shape (arguments q k v causal scale owner)) scratch.dataPointer=Bool.true)
    (allocation : Allocations (arguments q k v causal scale owner) o scratch.allocation)
    (binding : ScratchBinding (arguments q k v causal scale owner) scratch)
    (fresh : Fresh q k v o scratch) (native : Native.Instructions) (memory : ByteMemory) :
    let a:=arguments q k v causal scale owner
    let x:=Host.layout a o.tensor
    let base:=pointers q k v o scratch
    let result:=invoke a q k v o scratch callerDevice native memory
    scale.word=scale.marshal scale.raw ∧
    Host.Contract a ∧
    result.tensor=o ∧ result.scratch=scratch ∧
    result.tensor.tensor.shape=q.tensor.shape ∧ result.tensor.tensor.device=q.tensor.device ∧
    result.tensor.tensor.dtypeFP32=Bool.true ∧
    result.launchDevice=q.tensor.device ∧ result.restoredDevice=callerDevice ∧
    (∀ r offset,Physical.load (callerBytes base memory) r offset=readWord memory (base r+4*offset)) ∧
    Physical.load (Physical.byteExecute x (Native.ops native) scale.word (sourceSchedule x.shape) (callerBytes base memory))=
      Source.launch x (Native.ops native) scale.word (Physical.load (callerBytes base memory)) ∧
    (∀ b h n d,(Work.output b h n d).active x.shape →
      result.read b h n d=outputValue x (Native.ops native) (Physical.load (callerBytes base memory)) scale.word b h n d) ∧
    (∀ b h n,(Work.logsum b h n).active x.shape →
      result.readScratch (b*q.tensor.shape.heads+h) n=
        logValue x (Native.ops native) (Physical.load (callerBytes base memory)) scale.word b h n) ∧
    (∀ location : Fin (2^64),¬StoredByte x base .o location.val → ¬StoredByte x base .l location.val →
      result.memory location=memory location) ∧
    (∀ t∈[q,k,v],∀ location : Fin (2^64),t.storageBase≤location.val →
      location.val<t.storageBase+t.tensor.storageBytes → result.memory location=memory location) := by
  let a:=arguments q k v causal scale owner
  have valid:=admitted_inputs q k v qo ko vo scale hi causal owner
  have dtype:=(admitted_tensor o oo ho).materialized.1
  refine ⟨scale.marshalled,valid,rfl,rfl,allocation.outputShape,allocation.outputDevice,dtype,rfl,rfl,
    caller_load _ memory,source_bytes _ _ native scale.word memory,?_,?_,?_,?_⟩
  · exact fun b h n d active=>returned_word a q k v o scratch valid allocation oo ho callerDevice native memory b h n d active
  · exact fun b h n active=>returned_scratch_word a q k v o scratch valid allocation oo ho binding fresh hp
      callerDevice native memory b h n active
  · exact fun location hout hl=>finish_frame _ _ native scale.word memory location hout hl
  · exact fun t ht location hlo hhi=>input_frame a q k v o scratch allocation oo ho binding fresh native memory t ht location hlo hhi

/-- A separately proved numerical relation on the precise native graph lifts
to reads through the actual returned Tensor descriptor. -/
theorem returned_relation (a : Host.Arguments) (q k v o : Tensor) (scratch : Scratch)
    (valid : Host.Contract a) (allocation : Allocations a o scratch.allocation)
    (order : SortedAxes) (ho : tensorChecks o order=Bool.true) (callerDevice : Nat)
    (native : Native.Instructions) (memory : ByteMemory) (relation : Nat→Nat→Nat→Nat→Word→Prop)
    (numeric : ∀ b h n d,(Work.output b h n d).active (Host.shape a) →
      relation b h n d (outputValue (Host.layout a o.tensor) (Native.ops native)
        (Physical.load (callerBytes (pointers q k v o scratch) memory)) a.scale b h n d)) :
    ∀ b h n d,(Work.output b h n d).active (Host.shape a) →
      relation b h n d ((invoke a q k v o scratch callerDevice native memory).read b h n d) := by
  intro b h n d active
  rw [returned_word a q k v o scratch valid allocation order ho callerDevice native memory b h n d active]
  exact numeric b h n d active

#axiomsClean source_bytes
#axiomsClean whole_native
#axiomsClean returned_relation
end
end VeriTile.Bench.Repairs.FlashAttn.Checked
