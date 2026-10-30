import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Source

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

/-- Readback of a source store uses the same snapshot in both decode rounds. -/
theorem stores_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B pid : Nat) (snapshot s : State α)
    (j : Fin B) (active : pid*B+j.val<N) :
    (stores M p Y X N B pid snapshot s).mem Y (pid*B+j.val)=
      .mk p.output.toTileDType (ofFloat p.output
        (cast M p (toFloat p.input ((snapshot.mem X (pid*B+j.val)).read p.input.toTileDType)))) := by
  unfold stores
  apply State.masked_scatter_readback Y (fun i : TileIndex [B] => pid*B+i.1.val)
    (fun i => .mk p.output.toTileDType (ofFloat p.output
      (cast M p (toFloat p.input ((snapshot.mem X (pid*B+i.1.val)).read p.input.toTileDType)))))
    (fun i => pid*B+i.1.val<N) s (j,PUnit.unit) active
  intro a b equal
  dsimp only at equal
  have first : a.1=b.1 := Fin.ext (by omega)
  exact Prod.ext first (Subsingleton.elim _ _)

theorem memory_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α)
    (j : Fin B) (active : s.pids 0*B+j.val<N) :
    memory M p Y X N B s Y (s.pids 0*B+j.val)=
      .mk p.output.toTileDType (ofFloat p.output
        (cast M p (toFloat p.input ((s.mem X (s.pids 0*B+j.val)).read p.input.toTileDType)))) := by
  unfold memory
  split <;> apply stores_readback <;> exact active

/-- The exact source execution and its observed value are tied together. -/
theorem executes_readback {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (Y X : RegionName) (N B : Nat) (s : State α) :
    ∃ final,FP.Structural.exec M (kernel p Y X N B) s=some final ∧
      ∀ j : Fin B,s.pids 0*B+j.val<N → final.mem Y (s.pids 0*B+j.val)=
      .mk p.output.toTileDType (ofFloat p.output
        (cast M p (toFloat p.input ((s.mem X (s.pids 0*B+j.val)).read p.input.toTileDType)))) := by
  obtain ⟨final,run,mem⟩ := executes M p Y X N B s
  refine ⟨final,run,?_⟩
  intro j active
  rw [mem]
  exact memory_readback M p Y X N B s j active

#axiomsClean executes_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
