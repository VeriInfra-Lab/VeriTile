import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.RealOriginalBinding
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Proofs.RawNativeResults
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Tensor
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Contract

/-! Complete checked Tensor applications and source-to-physical address correspondence.
The raw helpers retain the full original physical-memory theorem scope. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API)
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
open Checked NativeIndex Execution KernelPhysical
noncomputable section

/-- Actual complete quoted Real kernel over the finite grid. Only ordinary
input bindings, bounds, and allocation separation are hypotheses. -/
theorem correct (c : Host.Call) (tokens : Nat) (request : Request) (buffers : NativeIndex.Buffers)
    (accepted : checkInputs c tokens request buffers=Bool.true)
    (allocation : FlatAlloc) (before : BlockState) (input : Logical.Tensors)
    (binding : RoleBinding c tokens allocation before) (reads : Logical.Resolved c before input)
    (separate : allocation.Disjoint)
    (complete : ∀ rg,rg∉allocation.regions → allocation.extent rg=0) :
    ∃ after,Nonempty (Kernel.GridLaunchedOrdinary
      (allocation.flattenKernel (Program.kernel c.layout c.regions .fp32 Bool.false
        (Host.scale c.layout.D)).eraseDType.toAlgKernel)
      c.grid (allocation.flattenState before) after) ∧
      (∀ q : Coord c,after.readMem allocation.flat (allocation.addr c.regions.out (offset q))=
        realTarget c input (batch q) (head q) (row q) (dim q)) ∧
      BlockState.WriteWithin (c.writes allocation) (allocation.flattenState before) after ∧
      (∀ rg,rg∈c.inputs → ∀ off,off<allocation.extent rg →
        after.mem allocation.flat (allocation.addr rg off)=
          (allocation.flattenState before).mem allocation.flat (allocation.addr rg off)) := by
  have valid := host_valid (checkInputs_iff.mp accepted).geometry binding
  obtain ⟨result,execution,output,frame,stable⟩ := Host.whole_call c allocation before valid separate complete Bool.false
  refine ⟨result.2,execution.2,?_,frame,stable⟩
  intro q
  exact (output q.val.1 q.val.2.1 (row q) q.property q.val.2.2.1.isLt (dim q)).trans
    (Logical.expected_eq c allocation before input valid reads _ _ _ _
      q.val.1.isLt q.val.2.1.isLt q.property (dim q).isLt)

#axiomsClean correct

/-- Actual quoted source-store address equals the physical API address. -/
theorem source_address (c : Host.Call) (api : API Word Store Byte) (base : Base) (buffers : Buffers)
    (binding : Binding c buffers base) (before : Store) (observed : Observed c api base before)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) (on : c.active idx i) :
    base c.regions.out+4*KernelReference.offset c.layout c.regions
      (KernelReference.atIndex (initial c api base before) idx) i=
      address buffers (KernelCoverage.coordinate c idx i on) := by
  rw [KernelFootprint.offset_matches c (initial c api base before)
    (KernelTensor.metadata c api base before observed) idx i,binding.output]
  rfl

alias input_tensor := KernelTensor.input_tensor

#axiomsClean source_address
#axiomsClean input_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorBridge

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory (API)
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
open Checked NativeIndex Execution KernelPhysical
noncomputable section
set_option maxHeartbeats 4000000

abbrev nativeTarget {Store Byte : Type} := @KernelPhysical.target Store Byte
abbrev PrimitiveLaws := NativeInstructions.ScalarLaws

structure NativeResult (c : Host.Call) (api : API Word Store (BitVec 8)) (base : Base)
    (buffers : Buffers) (reference : ScalarOps) (before : Store)
    (views : KernelTensor.Views c tokens request buffers) (result : KernelPhysical.Result Store) : Prop where
  recurrence : result.graph.mem=(KernelPhysical.reference c api base before).mem
  metadata : KernelFootprint.Metadata c result.graph
  sourceFrame : ∀ rg off,(¬∃ q : Coord c,rg=c.regions.out ∧ off=offset q) →
    result.graph.mem rg off=(initial c api base before).mem rg off
  words : ∀ q : Coord c,api.read result.memory (address buffers q)=nativeTarget c api base reference before q
  tensor : ∀ q : Coord c,(VeriTile.Torch.FP32.logicalValue (KernelTensor.read api result.memory)
    (views.tensor .output) (views.laws .output) (offset q)).bits=nativeTarget c api base reference before q
  bytes : ∀ (q : Coord c) (j : Fin 4),api.byte result.memory (address buffers q+j.val)=
    wordByte (nativeTarget c api base reference before q) j
  frame : ∀ i,Outside buffers (all c) i → api.byte result.memory i=api.byte before i
  inputWords : ∀ role,role≠.output → ∀ i,i<size c tokens role →
    api.read result.memory ((buffers.data role).base+4*i)=api.read before ((buffers.data role).base+4*i)
  metadataWords : ∀ role b,b<c.batches →
    api.read result.memory ((buffers.metadataBuffer role).base+4*b)=
      api.read before ((buffers.metadataBuffer role).base+4*b)
  nativePointers : Correspondence c .fp32 buffers
  outputAddress : ∀ q : Coord c,(queryPointer c .fp32 buffers .output (point q) (row q%128) (dim q)).toNat=address buffers q
  finiteBytes : ∀ (q : Coord c) (j : Fin 4),address buffers q+j.val<2^64 ∧
    (BitVec.ofNat 64 (address buffers q+j.val)).toNat=address buffers q+j.val

/-- Source execution is proved here, not assumed. The reference consumes only
initial words and a fixed shape-selected instruction profile. The only
numerical hypotheses compare local scalar instructions and fixed K8 WGMMA. -/
theorem fp_equiv (c : Host.Call) (api : API Word Store (BitVec 8)) (codec : WordCodec api)
    (base : Base) (buffers : Buffers) (actual reference : ScalarOps)
    (localLaws : PrimitiveLaws actual reference)
    (tokens : Nat) (request : Request) (accepted : checkInputs c tokens request buffers=Bool.true)
    (binding : Binding c buffers base) (views : KernelTensor.Views c tokens request buffers)
    (before : Store) (observed : Observed c api base before) :
    ∃ result,KernelPhysical.run c api base buffers actual before=some result ∧
      KernelReference.runGrid c (KernelNative.backend c.layout.D c.layout.window.toNat)
        (KernelReference.allPrograms c) (initial c api base before)=some result.graph ∧
      NativeResult c api base buffers reference before views result := by
  have equalOps := localLaws.operations_eq actual reference
  subst reference
  have admitted := checkInputs_iff.mp accepted
  obtain ⟨result,executed,source,recurrence,words,frame⟩ := KernelPhysical.words c api base buffers actual tokens request accepted before
  obtain ⟨final,finalSource,finalMetadata,sourceFrame⟩ := KernelFootprint.whole_frame c
    (KernelNative.backend c.layout.D c.layout.window.toNat) (KernelReference.allPrograms c)
    (initial c api base before) (KernelTensor.metadata c api base before observed)
    ⟨binding.outputSeparate _ (by simp),binding.outputSeparate _ (by simp)⟩
  have same : final=result.graph := by rw [source] at finalSource;exact Option.some.inj finalSource.symm
  subst final
  have memory : result.memory=stores c api buffers actual result.graph before := by
    have h := executed
    simp [KernelPhysical.run,source] at h
    exact (congrArg KernelPhysical.Result.memory h).symm
  refine ⟨result,executed,source,⟨recurrence,finalMetadata,?_,words,?_,?_,frame,?_,?_,
    correspondence c .fp32 buffers (native_ready admitted),source_pointer admitted,?_⟩⟩
  · intro rg off outside
    exact sourceFrame rg off (fun h => outside ((KernelCoverage.writes_iff c rg off).mp h))
  · intro q
    rw [KernelTensor.output_tensor c api buffers admitted views result.memory q]
    exact words q
  · intro q j
    rw [codec.bytes,words]
  · intro role notOutput i bound
    rw [memory]
    change api.read (execute api 0 c tokens buffers (fun _ q => word c actual result.graph q) (all c) before) _=_
    rw [execute_static api 0 admitted]
    apply VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.run_read_frame
    intro e he
    obtain ⟨q,_,rfl⟩ := List.mem_map.mp he
    have outputBound := offset_bound admitted.geometry q
    have separate := admitted.separate role notOutput
    rw [admitted.extents,admitted.extents] at separate
    change VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.Apart (address buffers q) _
    unfold address VeriTile.Bench.Repairs.LayernormFwdTriton.Memory.Apart
    omega
  · intro role b bound
    rw [memory]
    exact metadata_frame api 0 admitted (fun _ q => word c actual result.graph q) (all c) before role b bound
  · intro q j
    exact word_finite admitted q j j.isLt

/-- An empty grid is the identity independently of launch-only Admission. -/
theorem empty (c : Host.Call) (api : API Word Store Byte) (base : Base) (buffers : Buffers)
    (ops : ScalarOps) (before : Store) (noWork : c.batches=0 ∨ c.heads=0 ∨ c.blocks=0) :
    KernelPhysical.run c api base buffers ops before=some ⟨initial c api base before,before⟩ := by
  simp only [KernelPhysical.run,KernelCoverage.empty_programs c noWork,KernelReference.runGrid,Option.bind_some]
  change some (KernelPhysical.Result.mk (initial c api base before)
    (stores c api buffers ops (initial c api base before) before))=_
  rw [empty_stores c api buffers ops _ before (KernelCoverage.empty_coordinates c noWork)]

/-- Actual source integer mask for the compiler's constexpr-one branch. -/
theorem window_one_mask (row key : Int) : (key≤row ∧ row-1<key) ↔ key=row := by omega

theorem window_one_offdiagonal (block step row column : Int)
    (hr : 0≤row ∧ row<128) (hc : 0≤column ∧ column<128) (different : row≠column) :
    ¬(step*128+column≤block*128+row ∧ block*128+row-1<step*128+column) := by
  rw [window_one_mask]
  omega

/-- Exact source-store coverage and unique writers support the main endpoint. -/
alias write_coverage := KernelCoverage.writes_iff
alias source_writer_unique := KernelCoverage.source_writer_unique

#specNonCircular nativeTarget avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid,KernelPhysical.run,KernelPhysical.stores]
#axiomsClean fp_equiv
#axiomsClean empty
#axiomsClean window_one_mask
#axiomsClean window_one_offdiagonal
#axiomsClean write_coverage
#auditStmt fp_equiv
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.TensorBridge
