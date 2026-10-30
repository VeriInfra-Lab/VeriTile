import bench.tritonbench_g.attention_llama.Proofs.NativeFiniteGrid

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativePublicIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage
open Backing FiniteInputs GridGeometry
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Assemble source-success, actual access safety, native outputs and the
exact raw-cell frame with the shared ordinary-grid IO judgment. -/
theorem implements (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (hD : 0<c.D) (hN : 0<c.N)
    (hinj : Function.Injective (IOLayout.output (B:=B) c)) :
    (NativeIO.io A C arg B c).ImplementsOn (IOLayout.nativeDomain B c)
      (Mathematical.native A C arg B c) := by
  intro vals domain s inputs
  obtain ⟨states,executed,values,framed⟩:=NativeFiniteGrid.rows A C arg B c hD hN hinj vals s inputs domain
  let io:=NativeIO.io A C arg B c
  let p : Fin (io.schedule vals).length→NativeIO.Program B c := fun k=>
    NativeIO.program B c ⟨k.val,by simpa only [io,NativeIO.io,schedule_length] using k.isLt⟩
  let rows:=fun k=>states (p k)
  have runs : ∀ k,(io.schedule vals)[k].run s=some (rows k) := by
    intro k
    rw [schedule_get]
    exact executed (p k)
  have frames : ∀ k r o,¬io.writes vals k r o → (rows k).mem r o=s.mem r o := by
    intro k r o outside
    exact framed (p k) r o outside
  have disjoint : ∀ k j r o,io.writes vals k r o → io.writes vals j r o → k=j := by
    intro k j r o hk hj
    have same:=writes_disjoint c hinj (p k) (p j) r o hk hj
    have sameIndex:=(Fintype.equivFin (NativeIO.Program B c)).symm.injective same
    have hv:=congrArg (fun z : Fin (Fintype.card (NativeIO.Program B c))=>z.val) sameIndex
    exact Fin.ext hv
  let final:=OrdinaryRun.merge (io.writes vals) s rows
  refine ⟨final,⟨OrdinaryRun.of_rows (io.schedule vals) (io.writes vals) s rows runs frames disjoint⟩,?_,?_,?_⟩
  · intro capacity bounds k
    rw [schedule_get]
    exact NativeFiniteGrid.safe A C arg B c hD vals s inputs domain capacity bounds (p k)
  · intro o i active
    fin_cases o
    let k : Fin (io.schedule vals).length:=
      ⟨(Fintype.equivFin (NativeIO.Program B c)) (owner (index i)),by
        simpa only [io,NativeIO.io,schedule_length] using
          ((Fintype.equivFin (NativeIO.Program B c)) (owner (index i))).isLt⟩
    have pk : p k=owner (index i) := by simp [p,k,NativeIO.program]
    have hit : io.writes vals k c.Out (IOLayout.output c i) := by
      change NativeIO.writes B c (p k) _ _
      rw [pk]
      exact owner_writes c i
    change mapCell (Native.lower A c.fp8 c.D
      (VeriTile.Bench.Repairs.AttentionLlama.NativeSchedule.lane i.2.2.2.1.val))
        (final.mem c.Out (IOLayout.output c i))=
          Cell.mk .fp16 (Mathematical.native A C arg B c vals 0 i)
    rw [OrdinaryRun.merge_read (io.writes vals) s rows disjoint k _ _ hit]
    change mapCell _ ((states (p k)).mem _ _)=_
    rw [pk]
    exact values i
  · intro r o outside
    apply OrdinaryRun.merge_frame
    intro k hit
    obtain ⟨i,_,_,hr,ho⟩:=hit
    have miss:=outside.1 0 i trivial
    exact miss.elim (fun h=>h hr) (fun h=>h ho)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativePublicIO
