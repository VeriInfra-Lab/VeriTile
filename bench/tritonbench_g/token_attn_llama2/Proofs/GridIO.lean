import bench.tritonbench_g.token_attn_llama2.Proofs.IOBinding
import bench.tritonbench_g.token_attn_llama2.Proofs.GridRun

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.GridIO
open VeriTile Triton FP.Structural Whole IOBinding
open VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section
set_option maxHeartbeats 1000000
variable {α β : Type} [Inhabited β]

theorem invocation_initial (M : Program → Algebra β) (c : Call) (scale : ℝ)
    (pid : Program) (s : State β) :
    (Interface.invocation M c scale pid).initial s = start c.grid pid s := rfl

theorem runs (M : Program → Algebra β) (c : Call) (scale : ℝ)
    (work : List Program) (s : State β) :
    runSchedule (work.map (Interface.invocation M c scale)) s =
      NativeGrid.runPrograms M (c.parameters scale) c.grid work s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid work ih =>
    simp only [List.map_cons,runSchedule,NativeGrid.runPrograms,Invocation.run,
      invocation_initial,Interface.invocation]
    congr 1
    funext next
    exact ih next

theorem unique {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ)
    (work : List Program) (within : ∀ pid∈work, pid∈programs c.grid) :
    Unique (c.parameters scale) c.grid work s := by
  intro a ha b hb i j hi hj he
  obtain ⟨ba,ha',na⟩ := (mem_programs c.grid a.1 a.2.1 a.2.2).mp (within a ha)
  obtain ⟨bb,hb',nb⟩ := (mem_programs c.grid b.1 b.2.1 b.2.2).mp (within b hb)
  have hta := (active_eq h v scale a ba i).mp hi
  have htb := (active_eq h v scale b bb j).mp hj
  rw [offset_eq h scale a ba i,offset_eq h scale b bb j] at he
  have hsa := (v.starts ⟨a.1,ba⟩).2
  have hsb := (v.starts ⟨b.1,bb⟩).2
  obtain ⟨heHead,heToken⟩ := v.outputInjective a.2.1 ha' b.2.1 hb'
    _ (by dsimp at hsa; omega) _ (by dsimp at hsb; omega) he
  obtain ⟨heBatch,heLane⟩ := v.segmentsInjective a.1 ba b.1 bb
    _ hta _ htb heToken
  have hiBound : i.1.val<32 := i.1.isLt
  have hjBound : j.1.val<32 := j.1.isLt
  have heBlock : a.2.2=b.2.2 := by omega
  constructor
  · exact Prod.ext heBatch (Prod.ext heHead heBlock)
  · apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _

theorem frame {c : Call} {m : Metadata c} {xs : Interface.Inputs α c}
    {encode : RegionName → Nat → α → β} {s : State β}
    (h : Reads c m xs encode s) (v : Valid c m) (scale : ℝ)
    (M : Program → Algebra β) (observe : β → α) (work : List Program)
    (within : ∀ pid∈work, pid∈programs c.grid)
    (r : RegionName) (o : Nat)
    (outside : (Interface.io M c m scale work encode observe).Outside xs r o) :
    Whole.Outside (c.parameters scale) c.grid work s r o := by
  by_cases hr : r="Out"
  · right
    intro pid member i active
    obtain ⟨hb,hh,_⟩ := (mem_programs c.grid pid.1 pid.2.1 pid.2.2).mp (within pid member)
    have hn := (active_eq h v scale pid hb i).mp active
    have hl := length_bound c m v pid.1 hb
    have hnMax : pid.2.2*32+i.1.val<c.maxLength := by omega
    rw [offset_eq h scale pid hb i]
    have ho := outside.1 ⟨0,by simp [Interface.io,Interface.outputs]⟩
      (⟨pid.1,hb⟩,⟨pid.2.1,hh⟩,⟨pid.2.2*32+i.1.val,hnMax⟩,PUnit.unit) hn
    exact ho.resolve_left (by simpa [Interface.io,Interface.outputs] using hr)
  · exact Or.inl hr

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.GridIO
