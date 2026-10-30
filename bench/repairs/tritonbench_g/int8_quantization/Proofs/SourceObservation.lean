import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceGrid

namespace VeriTile.Bench.Repairs.Int8Quantization.Revision002.Grid
open VeriTile.Triton FP.Structural Schedule
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem readback (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (block : 0< p.blockRows) (channels : 0< p.channels) (blocks : 0< p.blocks)
    (order : List Program) (unique : (order.map Program.index).Nodup) (entry : State α)
    (w : Program) (member : w∈order) (i : Lane p) (valid : active p w i) :
    (model M ops p positive order entry).mem outRegion (offset p w i)=
      .mk .int (quantized M ops (parameters p w) positive (atProgram entry w) i) := by
  obtain ⟨before,after,rfl⟩:=List.mem_iff_append.mp member
  have afterNot : w.index∉after.map Program.index := by
    have nd : (before.map Program.index++w.index::after.map Program.index).Nodup := by
      simpa using unique
    exact (List.nodup_cons.mp (List.nodup_append.mp nd).2.1).1
  rw [model_append]
  change (model M ops p positive after
    (finalState M ops (parameters p w) positive (atProgram (model M ops p positive before entry) w))).mem
      outRegion (offset p w i)=_
  rw [frame M ops p positive after _ outRegion (offset p w i) (Or.inl (by decide)) (Or.inr (by
    intro v hv j validj equal
    have group:=(address_injective p block channels blocks w v i j valid validj equal).1
    apply afterNot
    exact List.mem_map.mpr ⟨v,hv,group.symm⟩))]
  have stored:=source_readback M ops (parameters p w) positive
    (atProgram (model M ops p positive before entry) w) i valid
  rw [show (finalState M ops (parameters p w) positive (atProgram (model M ops p positive before entry) w)).mem
    outRegion (offset p w i)=_ from stored]
  rw [(values_stable M ops p positive w entry (model M ops p positive before entry)
    (input_preserved M ops p positive before entry)).1]

theorem scale_readback (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (order : List Program) (unique : (order.map Program.index).Nodup)
    (entry : State α) (w : Program) (member : w∈order) :
    (model M ops p positive order entry).mem scaleRegion w.index=
      .mk .real (savedScale M ops (parameters p w) positive (atProgram entry w)) := by
  obtain ⟨before,after,rfl⟩:=List.mem_iff_append.mp member
  have afterNot : w.index∉after.map Program.index := by
    have nd : (before.map Program.index++w.index::after.map Program.index).Nodup := by simpa using unique
    exact (List.nodup_cons.mp (List.nodup_append.mp nd).2.1).1
  rw [model_append]
  change (model M ops p positive after
    (finalState M ops (parameters p w) positive (atProgram (model M ops p positive before entry) w))).mem
      scaleRegion w.index=_
  rw [frame M ops p positive after _ scaleRegion w.index (Or.inr (by
    intro v hv equal
    exact afterNot (List.mem_map.mpr ⟨v,hv,equal.symm⟩))) (Or.inl (by decide))]
  have stored:=source_scale M ops (parameters p w) positive
    (atProgram (model M ops p positive before entry) w)
  rw [show (finalState M ops (parameters p w) positive (atProgram (model M ops p positive before entry) w)).mem
    scaleRegion w.index=_ from stored]
  rw [(values_stable M ops p positive w entry (model M ops p positive before entry)
    (input_preserved M ops p positive before entry)).2]

def launch (M : Algebra α) (ops : Model.Primitives α) (p : Parameters) (entry : State α) :=
  execute M ops p (Schedule.programs 0 (p.batches*p.blocks)) entry

theorem whole_source (M : Algebra α) (ops : Model.Primitives α) (p : Parameters)
    (positive : 0< p.width) (entry : State α) :
    launch M ops p entry=some (model M ops p positive (Schedule.programs 0 (p.batches*p.blocks)) entry) :=
  execute_model M ops p positive _ entry

theorem coordinate_covered (p : Parameters) (block : 0< p.blockRows) (channels : 0< p.channels)
    (blocks : p.blocks=Geometry.blocks p.length p.blockRows) (cover : p.blockRows*p.channels≤ p.width)
    (batch row channel : Nat) (hb : batch< p.batches) (hr : row< p.length) (hc : channel< p.channels) :
    ∃ w∈Schedule.programs 0 (p.batches*p.blocks),∃ i : Lane p,
      active p w i ∧ offset p w i=batch*p.length*p.channels+row*p.channels+channel := by
  obtain ⟨group,lane,mask,rowEq,address⟩:=Geometry.coordinate p.batches p.length p.channels p.blockRows
    batch row channel p.width block channels hb hr hc cover
  rw [←blocks] at group rowEq address
  obtain ⟨w,hw,index⟩:=Schedule.complete (p.batches*p.blocks) (batch*p.blocks+row/p.blockRows) group
  refine ⟨w,hw,(⟨row%p.blockRows*p.channels+channel,lane⟩,PUnit.unit),?_,?_⟩
  · unfold active
    refine ⟨mask,?_⟩
    simpa only [index,rowEq] using hr
  · simpa only [offset,index] using address

#axiomsClean readback
#axiomsClean scale_readback
#axiomsClean whole_source
#axiomsClean coordinate_covered
end
end VeriTile.Bench.Repairs.Int8Quantization.Revision002.Grid
