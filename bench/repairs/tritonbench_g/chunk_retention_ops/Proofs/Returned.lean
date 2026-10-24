import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Context
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.SourceIdentity
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

def forwardTensors (storage : Storage c) (rf : Bool) : Contract.Tensor × Option Contract.Tensor :=
  (storage.tensor .output,if rf then some (storage.tensor .finalState) else none)
def backwardTensors (storage : Storage c) (ri : Bool) : List (Option Contract.Tensor) :=
  [some (storage.tensor .dq),some (storage.tensor .dk),some (storage.tensor .dv),
   if ri then some (storage.tensor .initialGradient) else none,none,none,none]

/-- Returned handles are the actual allocator/argument objects used by the
stores. Same-dtype Tensor.to is a no-copy identity; Tensor.sum's new object is
the .dv descriptor, separate from the .dvPart scratch object. -/
def forwardEntry (abi : MemoryABI σ) (ports : Numeric.ScalarPorts) (c : Shape)
    (storage : Storage c) (scale : Word) (compiler : Compiler c tree realScale seed rf terminal ri)
    (state : σ) := (forwardTensors storage rf,nativeExecute abi ports c storage.binding scale compiler.forward state)
def backwardEntry (abi : MemoryABI σ) (ports : Numeric.ScalarPorts) (c : Shape)
    (storage : Storage c) (scale : Word) (compiler : Compiler c tree realScale seed rf terminal ri)
    (checkpoint : Bool) (state : σ) :=
  (backwardTensors storage ri,nativeExecute abi ports c storage.binding scale
    (recomputeCode c tree checkpoint++compiler.backward) state)

theorem forward_model (abi : MemoryABI σ) (models : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws models ports) (c : Shape) (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (state : σ) :
    storage.binding.view 0 (abi.words (forwardEntry abi ports c storage scale compiler state).2)=
      Source.afterO (wordOps (Numeric.referenceISA models)) c scale tree rf (storage.binding.view 0 (abi.words state)) := by
  simp only [forwardEntry,native_reference abi models ports laws,compiler.forward_eq,forward_graph]
  exact Separate.forward_physical _ _ _ _ _ _ _

theorem output_value (ops : Ops α) (c : Shape) (b : Binding c) (scale : α) (tree : SumTree)
    (rf : Bool) (before : Memory α) (h t j : Nat) (hh : h<c.heads) (ht : t<c.T) (hj : j<c.V) :
    Separate.forwardPhysical ops c b scale tree rf before (b.base .output+c.valueOffset h t j)=
      Reference.output ops c (inputsOfHeap (b.view ops.zero before)) scale h t j := by
  have valid : (⟨.output,h,0,t,j⟩:Address).Valid c := ⟨hh,rfl,ht,hj⟩
  have eq := congrFun (Separate.forward_physical ops c b scale tree rf before) ⟨.output,h,0,t,j⟩
  rw [b.view_read ops.zero _ _ valid (by simp [inputRegions]),Source.o_value ops c scale tree rf _ h t j hh ht hj] at eq
  exact eq

theorem final_value (ops : Ops α) (c : Shape) (b : Binding c) (scale : α) (tree : SumTree)
    (before : Memory α) (h e j : Nat) (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    Separate.forwardPhysical ops c b scale tree Bool.true before (b.base .finalState+c.stateOffset h e j)=
      Reference.finalState ops c (inputsOfHeap (b.view ops.zero before)) h e j := by
  have valid : (⟨.finalState,h,0,e,j⟩:Address).Valid c := ⟨hh,rfl,he,hj⟩
  have eq := congrFun (Separate.forward_physical ops c b scale tree Bool.true before) ⟨.finalState,h,0,e,j⟩
  rw [b.view_read ops.zero _ _ valid (by simp [inputRegions]),Source.afterO,
    Source.execute_frame ops c scale tree .o _ (Source.o_footprint c) _ _ (by simp [Phase.writes]),
    Source.final_value ops c scale tree _ h e j hh he hj] at eq
  exact eq

theorem returned_byte (abi : MemoryABI σ) (storage : Storage c) (state : σ) (a : Address)
    (valid : a.Valid c) (present : storage.present a.region=Bool.true) (lane : Nat) (hl : lane<4) :
    abi.bytes state ((storage.tensor a.region).metadata.dataPtr.toNat+4*a.offset c+lane)=
      wordByte (abi.words state (storage.binding.address a)) lane := by
  have bound := a.offset_bound c valid
  obtain ⟨pointer,_,_,_⟩ := storage.live_span a.region present (by omega)
  rw [←pointer]
  have eq : 4*storage.base a.region+4*a.offset c+lane=4*(storage.binding.address a)+lane := by
    simp only [Binding.address,Storage.binding]
    ring
  rw [eq,byte_at_word abi state _ lane hl]

theorem forward_output_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports)
    (c : Shape) (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (state : σ)
    (present : storage.present .output=Bool.true) :
    ∀ h t j,h<c.heads → t<c.T → j<c.V → ∀ lane,lane<4 →
      abi.bytes (forwardEntry abi ports c storage scale compiler state).2
        ((forwardEntry abi ports c storage scale compiler state).1.1.metadata.dataPtr.toNat+
          4*c.valueOffset h t j+lane)=
      wordByte (Reference.output (wordOps (Numeric.referenceISA models)) c
        (inputsOfHeap (storage.binding.view 0 (abi.words state))) scale h t j) lane := by
  intro h t j hh ht hj lane hl
  change abi.bytes _ ((storage.tensor .output).metadata.dataPtr.toNat+4*c.valueOffset h t j+lane)=_
  have scalarByte := returned_byte abi storage (forwardEntry abi ports c storage scale compiler state).2
    ⟨.output,h,0,t,j⟩ ⟨hh,rfl,ht,hj⟩ present lane hl
  simp only [Address.offset] at scalarByte
  rw [scalarByte]
  simp only [forwardEntry,native_reference abi models ports laws,compiler.forward_eq,forward_graph]
  simpa only [Binding.address,Address.offset] using
    congrArg (fun value => wordByte value lane)
      (output_value (wordOps (Numeric.referenceISA models)) c storage.binding scale tree rf (abi.words state) h t j hh ht hj)

theorem forward_final_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports)
    (c : Shape) (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed Bool.true terminal ri) (state : σ)
    (present : storage.present .finalState=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V → ∀ lane,lane<4 →
      abi.bytes (forwardEntry abi ports c storage scale compiler state).2
        ((storage.tensor .finalState).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte (Reference.finalState (wordOps (Numeric.referenceISA models)) c
        (inputsOfHeap (storage.binding.view 0 (abi.words state))) h e j) lane := by
  intro h e j hh he hj lane hl
  have scalarByte := returned_byte abi storage (forwardEntry abi ports c storage scale compiler state).2
    ⟨.finalState,h,0,e,j⟩ ⟨hh,rfl,he,hj⟩ present lane hl
  simp only [Address.offset] at scalarByte
  rw [scalarByte]
  simp only [forwardEntry,native_reference abi models ports laws,compiler.forward_eq,forward_graph]
  simpa only [Binding.address,Address.offset] using
    congrArg (fun value => wordByte value lane)
      (final_value (wordOps (Numeric.referenceISA models)) c storage.binding scale tree (abi.words state) h e j hh he hj)

theorem backward_model (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports) (c : Shape)
    (fstorage storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (checkpoint : Bool) (before later : σ)
    (versions : Versions c
      (fstorage.binding.view 0 (abi.words (forwardEntry abi ports c fstorage scale compiler before).2))
      (storage.binding.view 0 (abi.words later)) checkpoint) :
    Separate.BackwardPost (wordOps (Numeric.referenceISA models)) c
      (Separate.joined (fstorage.binding.view 0 (abi.words before)) (storage.binding.view 0 (abi.words later)))
      scale tree (storage.binding.view 0 (abi.words (backwardEntry abi ports c storage scale compiler checkpoint later).2)) := by
  simp only [backwardEntry,native_reference abi models ports laws,compiler.backward_eq]
  exact backward_saved _ c scale tree compiler.sumCoverage rf ri checkpoint fstorage storage _ _ _
    (forward_model abi models ports laws c fstorage scale compiler before) versions

def gradientTarget (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α) (tree : SumTree)
    (a : Address) : α :=
  if a.region=.dq then Reference.dq ops c xs scale a.head a.row a.col
  else if a.region=.dk then Reference.dk ops c xs scale a.head a.row a.col
  else Reference.dv ops c xs scale tree a.head a.row a.col

theorem backward_gradient_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports) (c : Shape)
    (fstorage storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal ri) (checkpoint : Bool) (before later : σ)
    (versions : Versions c
      (fstorage.binding.view 0 (abi.words (forwardEntry abi ports c fstorage scale compiler before).2))
      (storage.binding.view 0 (abi.words later)) checkpoint)
    (a : Address) (valid : a.Valid c) (tag : a.region∈[Region.dq,Region.dk,Region.dv])
    (present : storage.present a.region=Bool.true) (lane : Nat) (hl : lane<4) :
    abi.bytes (backwardEntry abi ports c storage scale compiler checkpoint later).2
      ((storage.tensor a.region).metadata.dataPtr.toNat+4*a.offset c+lane)=
    wordByte (gradientTarget (wordOps (Numeric.referenceISA models)) c
      (Separate.joined (fstorage.binding.view 0 (abi.words before)) (storage.binding.view 0 (abi.words later)))
      scale tree a) lane := by
  rw [returned_byte abi storage _ a valid present lane hl]
  have post := backward_model abi models ports laws c fstorage storage scale compiler checkpoint before later versions
  have readonly : a.region∉inputRegions := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at tag
    rcases tag with h|h|h <;> simp [h,inputRegions]
  have read := storage.binding.view_read (0:Word)
    (abi.words (backwardEntry abi ports c storage scale compiler checkpoint later).2) a valid readonly
  rw [←read]
  rcases a with ⟨region,h,s,t,j⟩
  simp only [List.mem_cons,List.not_mem_nil,or_false] at tag
  rcases tag with rfl|rfl|rfl <;> simp only [Address.Valid] at valid
  all_goals rcases valid with ⟨hh,rfl,ht,hj⟩
  · simp only [gradientTarget,↓reduceIte]
    rw [post.query h hh t ht j hj]
  · simp only [gradientTarget,reduceCtorEq,↓reduceIte]
    rw [post.key h hh t ht j hj]
  · simp only [gradientTarget,reduceCtorEq,↓reduceIte]
    rw [post.value h hh t ht j hj]

#axiomsClean forward_output_bytes
#axiomsClean forward_final_bytes
#axiomsClean backward_model
#axiomsClean backward_gradient_bytes
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
