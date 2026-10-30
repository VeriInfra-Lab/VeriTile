import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Memory

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
open NativeGraph (Expr)
noncomputable section
set_option maxHeartbeats 3000000

def initial (r : Request) (heap : ByteMemory) : State Expr where
  mem := fun rg off => if rg=regions.start then .mk .int (indexValues r heap .start off)
    else if rg=regions.sequence then .mk .int (indexValues r heap .sequence off)
    else if rg=regions.prompt then .mk .int (indexValues r heap .prompt off)
    else if rg=regions.request then .mk .int (indexValues r heap .request off)
    else if rg=regions.pages then .mk .int (indexValues r heap .pages off)
    else .mk .real (.leaf (load32 heap (base r rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := r.call.grid.toNumPids
  undef := fun d _ _ => defaultValue d

def backend (r : Request) : Algebra Expr := NativeInstructions.backend r.call

def execute (r : Request) (heap : ByteMemory) : Option (State Expr) :=
  FPSuccess.runGrid r.call (backend r) (order r) (initial r heap)

/-- Independent online recurrence over the same original physical input words. -/
def reference (r : Request) (heap : ByteMemory) : State Expr :=
  FPReference.grid r.call (backend r) (NativeInstructions.support r.call) (order r) (initial r heap)

def outputWord (r : Request) (ops : NativeInstructions.ScalarOps) (s : State Expr) (i : Lane r) : Word :=
  let e := (s.mem regions.out (r.call.offset i.1 i.2.1)).read .real
  NativeInstructions.evaluate ops (NativeInstructions.plan r.call.layout.BM r.call.layout.D) e
    (NativeInstructions.owner r.call.layout.BM r.call.layout.D i.2.1.1.val i.2.1.2.1.val)

def referenceWord (r : Request) (ops : NativeInstructions.ScalarOps) (heap : ByteMemory) (i : Lane r) : Word :=
  outputWord r ops (reference r heap) i

def byte (word : Word) (i : Fin 4) : Byte := BitVec.ofNat 8 (word.toNat/256^i.val)

/-- Only executed output cells supply the stored payloads. -/
def nativeStores (r : Request) (ops : NativeInstructions.ScalarOps) (s : State Expr) (heap : ByteMemory) : ByteMemory :=
  writeback r (fun i => byte (outputWord r ops s i) i.2.2) heap

theorem initial_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    FPSuccess.Metadata r.call (initial r heap) := by
  refine ⟨?_,?_,?_⟩
  · intro b hb
    simpa [initial,Request.call,regions] using (h.checks.2.2.2.1 ⟨b,hb⟩).1
  · intro b hb
    simpa [initial,Request.call,regions] using (h.checks.2.2.2.1 ⟨b,hb⟩).2.1
  · intro b hb
    simpa [initial,Request.call,regions] using (h.checks.2.2.2.1 ⟨b,hb⟩).2.2.1

theorem source_reference (r : Request) (heap : ByteMemory) :
    ∃ final,execute r heap=some final ∧ final.mem=(reference r heap).mem :=
  FPReference.whole r.call (backend r) (NativeInstructions.support r.call) (order r) (initial r heap)

theorem native_bytes (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (ops : NativeInstructions.ScalarOps) :
    ∃ final,execute r heap=some final ∧
      (∀ i : Lane r,active r i → nativeStores r ops final heap (address r i)=byte (referenceWord r ops heap i) i.2.2) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → nativeStores r ops final heap a=heap a) ∧
      ReadFrame r heap (nativeStores r ops final heap) ∧
      (¬Needed r → nativeStores r ops final heap=heap) := by
  obtain ⟨final,run,memory⟩ := source_reference r heap
  refine ⟨final,run,?_,writeback_frame r _ heap,writeback_inputs r heap h _ heap,empty_writeback r _ heap⟩
  intro i hi
  rw [nativeStores,writeback_readback r heap h _ heap i hi]
  unfold outputWord referenceWord
  rw [memory]
  rfl

#specNonCircular referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,execute,nativeStores]
#specNonCircular nativeStores avoiding [reference,referenceWord]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
