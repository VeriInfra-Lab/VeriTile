import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPSequence
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.StorageImage

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPPhysicalFrame
open VeriTile.Triton Storage
open FP.Structural
noncomputable section
variable {α β : Type} [Inhabited α]

theorem forward (M : Algebra α) (h : Call) (storage : Placement h .forward)
    (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β) (initial : FPGridForward.Heap α) :
    ∃ final,FPGridForward.run M h initial=some final ∧
      ∀ addr,¬written storage addr →
        StorageImage.materialize storage encode entry (fun r o => final (r,o)) addr=
          StorageImage.materialize storage encode entry (fun r o => initial (r,o)) addr := by
  obtain ⟨final,he,_,hf⟩:=FPGridForward.complete M h initial
  refine ⟨final,he,?_⟩
  apply StorageImage.source_frame
  intro q o outside
  apply hf
  intro row
  rintro (⟨j,eq⟩|eq)
  · have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt⟩
  · have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using row.isLt⟩

theorem backward (M : Algebra α) (h : Call) (storage : Placement h .backward)
    (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β) (initial : FPGridForward.Heap α) :
    ∃ final,FPSequence.run M h initial=some final ∧
      ∀ addr,¬written storage addr →
        StorageImage.materialize storage encode entry (fun r o => final (r,o)) addr=
          StorageImage.materialize storage encode entry (fun r o => initial (r,o)) addr := by
  obtain ⟨final,he,_,_,hf⟩:=FPSequence.complete M h initial
  refine ⟨final,he,?_⟩
  apply StorageImage.source_frame
  intro q o outside
  apply hf
  · intro row
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [FPGridBackward.address,Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt⟩
  · intro pid
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [FPGridWeight.address,Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using j.property⟩

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPPhysicalFrame
