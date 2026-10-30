import bench.tritonbench_g.flash_attn.Proofs.IORealMath

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
open VeriTile Triton FP.Structural FlashAttn FlashAttn.ActualSource
noncomputable section

theorem real_implements (x : Layout) (hs : x.shape.Valid) (ho : OutputInjective x) (scale : ℝ) :
    (realIO x scale).ImplementsOn (Logical.RealInputs x) (Logical.mathematical x scale) := by
  intro xs domain bounds hb s hin
  refine ⟨schedule_safe KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
    xs bounds hb _ (fun j hj => (program_valid x j).mp hj) s,?_⟩
  obtain ⟨final,he,hm⟩ := schedule_memory KernelReal.algebra KernelReal.dot x (some scale)
    (programs (parameters x)) s
  refine ⟨final,he,?_,?_⟩
  · intro port i _
    fin_cases port
    · change mapCell id (final.mem "O" (FlashAttn.address x.o i.1.val i.2.1.val i.2.2.1.val i.2.2.2.1.val))=_
      rw [hm,grid_output_injective KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
        xs s hin hs ho i,real_output_formula x scale xs s hin hs ho domain i]
      rfl
    · change mapCell id (final.mem "L" (FlashAttn.lAddress x.shape i.1.val i.2.1.val i.2.2.1.val))=_
      rw [hm,grid_log_tensor KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id)
        xs s hin i,real_log_formula x scale xs s hin hs domain i]
      rfl
  · intro r off outside
    rw [hm]
    apply grid_frame
    exact (outside_iff KernelReal.algebra KernelReal.dot id x (some scale) (fun _ _ => id) xs r off).mp outside

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.PublicIO
