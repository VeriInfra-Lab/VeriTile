import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Returned
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

theorem initial_value (ops : Ops α) (c : Shape) (storage : Storage c) (scale : α) (tree : SumTree)
    (checkpoint : Bool) (later : Memory α) (h e j : Nat)
    (hh : h<c.heads) (he : e<c.K) (hj : j<c.V) :
    execute ops c storage.binding scale (recomputeCode c tree checkpoint++backwardCode c tree Bool.true)
      later (storage.base .initialGradient+c.stateOffset h e j)=
    ops.cast (Reference.reverseCarry ops c (inputsOfHeap (storage.binding.view ops.zero later)) scale h c.chunks e j) := by
  have valid : (⟨.initialGradient,h,0,e,j⟩:Address).Valid c := ⟨hh,rfl,he,hj⟩
  let prepared := execute ops c storage.binding scale (recomputeCode c tree checkpoint) later
  have same := congrFun (Separate.backward_physical ops c storage.binding scale tree Bool.true prepared)
    ⟨.initialGradient,h,0,e,j⟩
  rw [storage.binding.view_read ops.zero _ _ valid (by simp [inputRegions]),
    Separate.backwardAPI_initial ops c scale tree _ h e j hh he hj] at same
  have prep := prepare_physical ops c storage scale tree checkpoint later
  change storage.binding.view ops.zero prepared=_ at prep
  rw [prep,prepare_inputs] at same
  rw [execute_append,backward_graph]
  exact same

theorem backward_initial_bytes (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports) (c : Shape)
    (storage : Storage c) (scale : Word)
    (compiler : Compiler c tree realScale seed rf terminal Bool.true) (checkpoint : Bool) (later : σ)
    (present : storage.present .initialGradient=Bool.true) :
    ∀ h e j,h<c.heads → e<c.K → j<c.V → ∀ lane,lane<4 →
      abi.bytes (backwardEntry abi ports c storage scale compiler checkpoint later).2
        ((storage.tensor .initialGradient).metadata.dataPtr.toNat+4*c.stateOffset h e j+lane)=
      wordByte ((wordOps (Numeric.referenceISA models)).cast
        (Reference.reverseCarry (wordOps (Numeric.referenceISA models)) c
          (inputsOfHeap (storage.binding.view 0 (abi.words later))) scale h c.chunks e j)) lane := by
  intro h e j hh he hj lane hl
  have scalarByte := returned_byte abi storage (backwardEntry abi ports c storage scale compiler checkpoint later).2
    ⟨.initialGradient,h,0,e,j⟩ ⟨hh,rfl,he,hj⟩ present lane hl
  simp only [Address.offset] at scalarByte
  rw [scalarByte]
  simp only [backwardEntry,native_reference abi models ports laws,compiler.backward_eq]
  simpa only [Binding.address,Address.offset,Storage.binding] using
    congrArg (fun value => wordByte value lane)
      (initial_value (wordOps (Numeric.referenceISA models)) c storage scale tree checkpoint (abi.words later) h e j hh he hj)

#axiomsClean backward_initial_bytes
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
