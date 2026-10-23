import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.Execution
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.SourceGraph

namespace VeriTile.Bench.Repairs.ContextAttnMistral.WholeFP
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Checked Execution
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Host-observed signed metadata is the same raw caller storage supplied to
the kernel. This is an ordinary local input binding, with no result premise. -/
structure MetadataBinding (api : API Word Store Byte) (c : Host.Call)
    (buffers : NativeIndex.Buffers) (m : Store) : Prop where
  start : ∀ b,b<c.batches →
    (api.read m ((buffers.metadataBuffer .start).base+4*b)).toInt=(c.start b : Int)
  length : ∀ b,b<c.batches →
    (api.read m ((buffers.metadataBuffer .length).base+4*b)).toInt=(c.length b : Int)

abbrev graphInputs := SourceGraph.operands

/-- The compiler-bound source fragments, evaluated at each actual store owner.
All matrix work is implemented by fixed k8 WGMMA instruction chains. -/
def source (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (input : Recurrence.Inputs Word) (q : Coord c) : Word :=
  NativeInstructions.evaluate ops plan
    (SourceGraph.value binding input (batch q) (head q) (row q) (dim q)) (owner q)

/-- Independent recurrence target, using only initial logical input words. -/
def target (ops : ScalarOps) (plan : NativePhysical.Plan) (owner : Coord c → Nat)
    (input : Recurrence.Inputs Word) (q : Coord c) : Word :=
  NativeInstructions.evaluate ops plan
    (Recurrence.expected (NativePhysical.backend plan) c (graphInputs input)
      (batch q) (head q) (row q) (dim q)) (owner q)

theorem source_target (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (input : Recurrence.Inputs Word) (q : Coord c) : source ops plan binding owner input q=target ops plan owner input q := by
  unfold source target
  rw [SourceGraph.value_expected]

/-- Actual selected store events also consume the compiler binding's store
operand, rather than silently using an unrelated output-address function. -/
def selectedStep (api : API Word Store Byte) (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (tokens : Nat) (buffers : NativeIndex.Buffers) (m : Store) (q : Coord c) : Store :=
  api.write m ((buffers.data .output).base+
    4*binding.store.selected (batch q) (head q) (row q) (dim q))
    (source ops plan binding owner (inputs api 0 c tokens buffers m) q)

def selectedExecute (api : API Word Store Byte) (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (tokens : Nat) (buffers : NativeIndex.Buffers) (order : List (Coord c)) (m : Store) : Store :=
  order.foldl (selectedStep api ops plan binding owner tokens buffers) m

theorem selected_execute_eq (api : API Word Store Byte) (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (tokens : Nat) (buffers : NativeIndex.Buffers) (order : List (Coord c)) (m : Store) :
    selectedExecute api ops plan binding owner tokens buffers order m=
      execute api 0 c tokens buffers (source ops plan binding owner) order m := by
  unfold selectedExecute execute
  congr 1
  funext state q
  simp only [selectedStep,step,binding.store_map,address,offset]

def selectedDispatch (api : API Word Store Byte) (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (tokens : Nat) (request : Request) (buffers : NativeIndex.Buffers)
    (order : List (Coord c)) (m : Store) : Except Unit (Option Unit × Tensor × Store) :=
  match validate c tokens request buffers with
  | .error _ => .error ()
  | .ok Bool.false => .ok (none,request.tensor .output,m)
  | .ok Bool.true => .ok (none,request.tensor .output,
      selectedExecute api ops plan binding owner tokens buffers order m)

theorem selected_dispatch_eq (api : API Word Store Byte) (ops : ScalarOps) (plan : NativePhysical.Plan)
    (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (tokens : Nat) (request : Request) (buffers : NativeIndex.Buffers)
    (order : List (Coord c)) (m : Store) :
    selectedDispatch api ops plan binding owner tokens request buffers order m=
      dispatch api 0 c tokens request buffers (source ops plan binding owner) order m := by
  unfold selectedDispatch dispatch
  cases validate c tokens request buffers with
  | error _ => rfl
  | ok launch => cases launch <;> simp only [selected_execute_eq] <;> rfl

/-- Full raw-word and physical-byte result of the selected scalar/store graph.
The compiler boundary supplies only ordered trees, operand routes and owners;
no premise asserts successful graph execution or the desired output. -/
theorem checked_native (api : API Word Store Byte) (ops : ScalarOps)
    (plan : NativePhysical.Plan) (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (accepted : check c tokens request buffers=Bool.true)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store) (metadata : MetadataBinding api c buffers m) :
    NativeIndex.Correspondence c .fp32 buffers ∧
    (∀ q,api.read (execute api 0 c tokens buffers (source ops plan binding owner) order m) (address buffers q)=
      target ops plan owner (inputs api 0 c tokens buffers m) q) ∧
    (∀ i,Outside buffers order i →
      api.byte (execute api 0 c tokens buffers (source ops plan binding owner) order m) i=api.byte m i) ∧
    (∀ (q : Coord c) j,j<4 → address buffers q+j<2^64) := by
  have a := check_iff.mp accepted
  refine ⟨NativeIndex.correspondence c .fp32 buffers (native_ready a),?_,?_,?_⟩
  · intro q
    rw [readback api 0 a _ order m q (complete q),source_target]
  · exact byte_frame api 0 a _ order m
  · intro q j hj;exact (word_finite a q j hj).1

/-- The actual signed metadata reads remain fixed throughout the dynamic
store trace, including when metadata aliases other read-only raw-word views. -/
theorem metadata_preserved (api : API Word Store Byte) (ops : ScalarOps)
    (plan : NativePhysical.Plan) (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (a : Admission c tokens request buffers) (order : List (Coord c)) (m : Store)
    (metadata : MetadataBinding api c buffers m) :
    MetadataBinding api c buffers (execute api 0 c tokens buffers (source ops plan binding owner) order m) := by
  constructor
  · intro b hb;rw [metadata_frame api 0 a];exact metadata.start b hb;exact hb
  · intro b hb;rw [metadata_frame api 0 a];exact metadata.length b hb;exact hb

#axiomsClean metadata_preserved
/-- Only local rounded scalar and fixed WGMMA instruction conformance is
assumed. Exceptional IEEE words, finite sentinel guards and padded products
remain in the graph. There is no exact-Real floating assumption. -/
theorem fpequiv (api : API Word Store Byte) (actual standard : ScalarOps)
    (laws : NativeInstructions.ScalarLaws actual standard)
    (plan : NativePhysical.Plan) (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (accepted : check c tokens request buffers=Bool.true)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store) (metadata : MetadataBinding api c buffers m) :
    (∀ q,api.read (execute api 0 c tokens buffers (source actual plan binding owner) order m) (address buffers q)=
      target standard plan owner (inputs api 0 c tokens buffers m) q) ∧
    (∀ i,Outside buffers order i →
      api.byte (execute api 0 c tokens buffers (source actual plan binding owner) order m) i=api.byte m i) := by
  rw [laws.operations_eq actual standard]
  have h := checked_native api standard plan binding owner accepted order complete m metadata
  exact ⟨h.2.1,h.2.2.1⟩

/-- Successful empty/launch checked dispatch, exact caller descriptor and
None return. No launch-only pointer or alignment fact is needed for emptiness. -/
theorem checked_words_call (api : API Word Store Byte) (actual standard : ScalarOps)
    (laws : NativeInstructions.ScalarLaws actual standard)
    (plan : NativePhysical.Plan) (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (launch : Bool) (accepted : validate c tokens request buffers=.ok launch)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store) (metadata : MetadataBinding api c buffers m) :
    ∃ after, selectedDispatch api actual plan binding owner tokens request buffers order m=
      .ok (none,request.tensor .output,after) ∧
      ((launch=Bool.false ∧ after=m) ∨
       (launch=Bool.true ∧
        (∀ q,api.read after (address buffers q)=target standard plan owner (inputs api 0 c tokens buffers m) q) ∧
        (∀ i,Outside buffers order i → api.byte after i=api.byte m i))) := by
  rw [selected_dispatch_eq]
  cases launch
  · exact ⟨m,dispatch_empty _ _ _ _ _ _ _ _ _ accepted,Or.inl ⟨rfl,rfl⟩⟩
  · have a := (validate_launch.mp accepted).2
    have result := fpequiv api actual standard laws plan binding owner (check_iff.mpr a) order complete m metadata
    exact ⟨_,dispatch_launch _ _ _ _ _ _ _ _ _ accepted,Or.inr ⟨rfl,result⟩⟩

#axiomsClean checked_words_call
/-- Complete byte-level checked endpoint. All four bytes of every written
FP32 word are the independently predicted raw-word encoding. -/
theorem checked_call (api : API Word Store (BitVec 8)) (codec : WordCodec api)
    (actual standard : ScalarOps) (laws : NativeInstructions.ScalarLaws actual standard)
    (plan : NativePhysical.Plan) (binding : SourceGraph.Binding c plan) (owner : Coord c → Nat)
    (launch : Bool) (accepted : validate c tokens request buffers=.ok launch)
    (order : List (Coord c)) (complete : ∀ q,q∈order) (m : Store)
    (metadata : MetadataBinding api c buffers m) :
    ∃ after, selectedDispatch api actual plan binding owner tokens request buffers order m=
      .ok (none,request.tensor .output,after) ∧
      ((launch=Bool.false ∧ after=m) ∨
       (launch=Bool.true ∧
        (∀ q,api.read after (address buffers q)=target standard plan owner (inputs api 0 c tokens buffers m) q) ∧
        (∀ q (j : Fin 4),api.byte after (address buffers q+j.val)=
          wordByte (target standard plan owner (inputs api 0 c tokens buffers m) q) j) ∧
        (∀ i,Outside buffers order i → api.byte after i=api.byte m i))) := by
  obtain ⟨after,run,result⟩ := checked_words_call api actual standard laws plan binding owner launch accepted order complete m metadata
  refine ⟨after,run,?_⟩
  rcases result with empty|⟨live,words,frame⟩
  · exact Or.inl empty
  · exact Or.inr ⟨live,words,fun q j=>by rw [codec.bytes,words q],frame⟩

#axiomsClean checked_call
#auditStmt checked_call
#axiomsClean selected_execute_eq
#axiomsClean selected_dispatch_eq
#axiomsClean source_target
#axiomsClean checked_native
#axiomsClean fpequiv
#auditStmt fpequiv
#specNonCircular target avoiding [FP.Structural.exec,Recurrence.sourceValue,execute,source]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.WholeFP
