import bench.tritonbench_g.token_attn_llama2.Proofs.IOSafety
import bench.tritonbench_g.token_attn_llama2.Proofs.FormulaBinding

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOImplements
open VeriTile Triton FP.Structural Whole IOBinding KernelNative
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
variable {α β : Type} [Inhabited α] [Inhabited β]

/-- Reuse the established source/grid execution theorem. The independent graph
result is proved before observation, and the entire raw typed frame survives. -/
theorem graph (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (work : List Program) (order : (programs c.grid).Perm work)
    (encode : RegionName → Nat → α → β) (xs : Interface.Inputs α c)
    (s : State β) (v : Valid c m) (bound : Reads c m xs encode s) :
    ∃ final, runSchedule (work.map (Interface.invocation M c scale)) s=some final ∧
      (∀ (b : Fin c.batches) (h : Fin c.heads) (t : Fin c.maxLength), t.val<m.len b.val →
        final.mem "Out" (h.val*c.outHead+(m.begin b.val+t.val)*c.outSeq) =
          Cell.mk .fp32 (Reference.score M encode c m scale xs b h t)) ∧
      (∀ r o, Whole.Outside (c.parameters scale) c.grid work s r o → final.mem r o=s.mem r o) := by
  have within : ∀ pid∈work, pid∈programs c.grid := fun _ hp => order.mem_iff.mpr hp
  obtain ⟨final,ran,values,frame⟩ := NativeGrid.run_correct M (c.parameters scale)
    (parameters_separated c scale) c.grid work s (GridIO.unique bound v scale work within)
  refine ⟨final,(GridIO.runs M c scale work s).trans ran,?_,frame⟩
  intro b h t ht
  have covers := grid_covers c m v b.val h.val t.val b.isLt h.isLt ht
  have member := order.mem_iff.mp covers.1
  have active : active (c.parameters scale) (start c.grid (b.val,h.val,t.val/32) s)
      (lane c scale t.val) := by
    apply (active_eq bound v scale _ b.isLt _).mpr
    simpa only [lane,covers.2] using ht
  have hv := values (b.val,h.val,t.val/32) member (lane c scale t.val) active
  rw [offset_eq bound scale _ b.isLt _,FormulaBinding.score_eq M c m scale encode xs s v bound b h t] at hv
  simpa only [lane,covers.2] using hv

theorem observed (M : Program → Algebra β) (c : Call) (m : Metadata c) (scale : ℝ)
    (work : List Program) (order : (programs c.grid).Perm work)
    (encode : RegionName → Nat → α → β) (observe : β → α) (v : Valid c m)
    (formula : Interface.Inputs α c → Interface.Outputs α c)
    (formula_eq : ∀ xs (b : Fin c.batches) (h : Fin c.heads) (t : Fin c.maxLength),
      t.val<m.len b.val → observe (Reference.score M encode c m scale xs b h t) =
        formula xs ⟨0,by simp [Interface.outputs]⟩ (b,h,t,PUnit.unit)) :
    (Interface.io M c m scale work encode observe).ImplementsOn
      (Interface.MetadataValues c m) formula := by
  intro xs metadata bounds bounded s input
  have reads := IOBinding.reads M c m scale work encode observe xs s metadata input
  have within : ∀ pid∈work, pid∈programs c.grid := fun _ hp => order.mem_iff.mpr hp
  refine ⟨IOSafety.schedule M c m scale work encode observe xs v bounds bounded work within s reads,?_⟩
  obtain ⟨final,ran,values,frame⟩ := graph M c m scale work order encode xs s v reads
  refine ⟨final,ran,?_,?_⟩
  · intro o j active
    fin_cases o
    change mapCell observe (final.mem "Out" (j.2.1.val*c.outHead+(m.begin j.1.val+j.2.2.1.val)*c.outSeq)) = _
    rw [values j.1 j.2.1 j.2.2.1 active]
    change Cell.mk .fp32 (observe (Reference.score M encode c m scale xs j.1 j.2.1 j.2.2.1)) = _
    rw [formula_eq xs j.1 j.2.1 j.2.2.1 active]
    have eta : (j.1,j.2.1,j.2.2.1,PUnit.unit) = j := by
      rcases j with ⟨b,h,t,u⟩
      cases u
      rfl
    rw [eta]
    rfl
  · intro r o outside
    exact frame r o (GridIO.frame reads v scale M observe work within r o outside)

theorem real (c : Call) (m : Metadata c) (v : Valid c m)
    (work : List Program) (order : (programs c.grid).Perm work) :
    (Interface.io (fun _ => realAlgebra) c m (idealScale c.dim) work (fun _ _ x => x) id).ImplementsOn
      (Interface.MetadataValues c m) (Reference.real c m) :=
  observed _ c m _ work order _ _ v _ (fun xs b h t ht => FormulaBinding.real_score c m xs b h t ht)


#axiomsClean graph
#axiomsClean real
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.IOImplements
