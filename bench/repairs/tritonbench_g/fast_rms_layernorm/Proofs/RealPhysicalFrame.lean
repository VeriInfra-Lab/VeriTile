import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealSequence
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.StorageImage

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealPhysicalFrame
open VeriTile.Triton Storage
noncomputable section
variable {β : Type}

theorem forward (h : Call) (storage : Placement h .forward)
    (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β) (initial : RealGridForward.Heap) :
    ∃ final,RealGridForward.run h initial=some final ∧
      ∀ addr,¬written storage addr →
        StorageImage.materialize storage encode entry (fun r o => final (r,o)) addr=
          StorageImage.materialize storage encode entry (fun r o => initial (r,o)) addr := by
  obtain ⟨final,he,_,hf⟩:=RealGridForward.complete h initial
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

theorem backward (h : Call) (storage : Placement h .backward)
    (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β) (initial : RealGridForward.Heap) :
    ∃ final,RealSequence.run h initial=some final ∧
      ∀ addr,¬written storage addr →
        StorageImage.materialize storage encode entry (fun r o => final (r,o)) addr=
          StorageImage.materialize storage encode entry (fun r o => initial (r,o)) addr := by
  obtain ⟨final,he,_,_,hf⟩:=RealSequence.complete h initial
  refine ⟨final,he,?_⟩
  apply StorageImage.source_frame
  intro q o outside
  apply hf
  · intro row
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [RealGridBackward.address,Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using Native.offset_bound h row.val j.val row.isLt j.isLt⟩
  · intro pid
    rintro ⟨j,eq⟩
    have qr:=congrArg Prod.fst eq
    have oo:=congrArg Prod.snd eq
    simp only [RealGridWeight.address,Prod.fst,Prod.snd] at qr oo
    subst q; subst o
    exact outside ⟨by simp [outputs],by simpa [cells] using j.property⟩

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealPhysicalFrame
