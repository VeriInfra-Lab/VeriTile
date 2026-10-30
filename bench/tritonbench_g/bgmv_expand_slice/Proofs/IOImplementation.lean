import bench.tritonbench_g.bgmv_expand_slice.Proofs.IOSafety
import bench.tritonbench_g.bgmv_expand_slice.Proofs.IOValues

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

theorem source_implements (M : Algebra A) (R : Reduction A) (project : A → A) (h : Call) :
    (sourceIO M R project h).ImplementsOn Valid
      (fun xs => result h (fun b n => project
        (localValue M R h (input xs b) (weight xs b n) (previous xs b n)))) := by
  intro xs valid bounds hb s hin
  have metadata := index_binding M R project h xs s hin
  refine ⟨schedule_safe M R project h xs valid bounds hb _ s metadata,?_⟩
  have gridResult : ∃ final,runSchedule ((Geometry.programs h).map (invocation M R h)) s=some final ∧
      (∀ p : Geometry.Program h,(PublicExecution.data s).indices p.batch≠-1 →
        final.mem "Y" (FPGrid.address h p)=.mk .real (FPGrid.value M R h p (PublicExecution.data s))) ∧
      (∀ name offset,PublicExecution.Outside h s (Geometry.programs h) name offset →
        final.mem name offset=s.mem name offset) := by
    by_cases positive : 0<h.N
    · obtain ⟨final,ran,values,frame⟩ := PublicExecution.grid_execution M R h (Geometry.programs h) s
        (fun b hb => by rw [metadata positive ⟨b,hb⟩];exact valid ⟨b,hb⟩)
        (by simpa [FPGrid.address,Geometry.outputOffset,
          VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry.outputOffset]
          using Geometry.programs_unique h)
      exact ⟨final,ran,fun p => values p (Geometry.programs_complete h p),frame⟩
    · have zero : h.N=0 := by omega
      have empty : Geometry.programs h=[] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro p _
        have := p.rowInside
        omega
      exact ⟨s,by simp [empty,runSchedule],fun p => by have := p.rowInside;omega,
        fun _ _ _ => rfl⟩
  obtain ⟨final,ran,values,frame⟩ := gridResult
  refine ⟨final,ran,?_,?_⟩
  · intro o j active
    have oz : o=⟨0,by simp [sourceIO]⟩ := Fin.ext (by have := o.isLt;change o.val<1 at this;change o.val=0;omega)
    subst o
    rcases j with ⟨b,n,u⟩
    have positive : 0<h.N := by have := n.isLt;omega
    have hi := metadata positive b
    have stored := values ⟨b.val,n.val,b.isLt,n.isLt⟩ (by rwa [hi])
    change mapCell project (final.mem "Y" (b.val*h.columns+n.val+h.slice))=
      .mk .real (project (localValue M R h (input xs b) (weight xs b n) (previous xs b n)))
    rw [show final.mem "Y" (b.val*h.columns+n.val+h.slice)=_ from stored]
    simp only [mapCell,mapValue]
    exact congrArg (fun v => Cell.mk .real (project v)) (value_binding M R project h xs s hin b n active)
  · intro name offset outside
    apply frame name offset
    by_cases other : name≠"Y"
    · exact Or.inl other
    · have same : name="Y" := by simpa using other
      subst name
      right
      intro p member active
      have positive : 0<h.N := by have := p.rowInside;omega
      have hi := metadata positive ⟨p.batch,p.batchInside⟩
      have miss := outside.1 ⟨0,by simp [sourceIO]⟩
        (⟨p.batch,p.batchInside⟩,⟨p.row,p.rowInside⟩,PUnit.unit) (by rwa [hi] at active)
      exact miss.resolve_left (by simp [sourceIO])

#axiomsClean source_implements
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicIO
