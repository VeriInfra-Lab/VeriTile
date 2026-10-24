import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PairGrouped
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PairFinal
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.GroupedIOResult
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.FinalIO
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardPairIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
variable {α : Type} [Inhabited α]

namespace Grouped

def variant (m : Model α) (stride M N G B group : Nat) :
    ReadProgramIO.Variant (Interface.Grouped.io m stride M N G B group (fun _ _ => id)) where
  schedule := fun _ => [job m (PairGrouped.kernel stride M N G B) group]
  observe := (Interface.Grouped.io m stride M N G B group (fun _ _ => id)).observe

def original (m : Model α) (stride M N G B group : Nat) :=
  ReadProgramIO.Implementation.original
    (Interface.Grouped.io m stride M N G B group (fun _ _ => id)) (fun _ => True)

def candidate (m : Model α) (stride M N G B group : Nat) :=
  ReadProgramIO.Implementation.ofVariant (variant m stride M N G B group) (fun _ => True)

/-- Complete grouped-row loop, all DX result reads (including unvisited rows),
both partial sums, source safety/success and the original narrow write frame. -/
theorem equivalent (m : Model α) (stride M N G B group : Nat) :
    (Interface.Grouped.io m stride M N G B group (fun _ _ => id)).Equivalent
      (variant m stride M N G B group) := by
  intro vals _ bounds boundsOK initial binding
  obtain ⟨safe,t,ran,values,frame⟩ :=
    GroupedIOResult.implements m stride M N G B group (fun _ _ => id)
      vals trivial bounds boundsOK initial binding
  have csafe : ScheduleSafe bounds ((variant m stride M N G B group).schedule vals) initial := by
    change ScheduleSafe bounds [job m (PairGrouped.kernel stride M N G B) group] initial
    rw [scheduleSafe_singleton]
    apply (PairGrouped.safe_iff m _ stride M N G B bounds).mpr
    simpa only [Interface.Grouped.io,Interface.Grouped.layout,scheduleSafe_singleton] using safe
  have eq : runSchedule ((variant m stride M N G B group).schedule vals) initial =
      runSchedule ((Interface.Grouped.io m stride M N G B group (fun _ _ => id)).schedule vals) initial := by
    simp only [variant,Interface.Grouped.io,Interface.Grouped.layout,runSchedule_singleton,
      Invocation.run,job,Invocation.initial]
    exact PairGrouped.exact m _ stride M N G B
  exact ⟨safe,csafe,t,t,ran,eq.trans ran,fun _ _ _ => rfl,frame,frame⟩
end Grouped

namespace Final

def variant (m : Model α) (M N BM BN pid : Nat) :
    ProgramIO.Variant (Interface.Final.io m M N BM BN pid (fun _ _ => id)) where
  schedule := fun _ => [job m (PairFinal.kernel M N BM BN) pid]
  observe := (Interface.Final.io m M N BM BN pid (fun _ _ => id)).observe

def original (m : Model α) (M N BM BN pid : Nat) :=
  ProgramIO.Implementation.original
    (Interface.Final.io m M N BM BN pid (fun _ _ => id)) (fun _ => True)

def candidate (m : Model α) (M N BM BN pid : Nat) :=
  ProgramIO.Implementation.ofVariant (variant m M N BM BN pid) (fun _ => True)

/-- Complete final tile loop, both gradient outputs, actual source access
safety/success and the full original exterior frame. -/
theorem equivalent (m : Model α) (M N BM BN pid : Nat) :
    (Interface.Final.io m M N BM BN pid (fun _ _ => id)).Equivalent
      (variant m M N BM BN pid) := by
  intro vals _ bounds boundsOK initial binding
  obtain ⟨safe,t,ran,values,frame⟩ :=
    FinalIO.implements m M N BM BN pid (fun _ _ => id)
      vals trivial bounds boundsOK initial binding
  have csafe : ScheduleSafe bounds ((variant m M N BM BN pid).schedule vals) initial := by
    change ScheduleSafe bounds [job m (PairFinal.kernel M N BM BN) pid] initial
    rw [scheduleSafe_singleton]
    apply (PairFinal.safe_iff m _ M N BM BN bounds).mpr
    simpa only [Interface.Final.io,scheduleSafe_singleton] using safe
  have eq : runSchedule ((variant m M N BM BN pid).schedule vals) initial =
      runSchedule ((Interface.Final.io m M N BM BN pid (fun _ _ => id)).schedule vals) initial := by
    simp only [variant,Interface.Final.io,runSchedule_singleton,Invocation.run,job,Invocation.initial]
    exact PairFinal.exact m _ M N BM BN
  exact ⟨safe,csafe,t,t,ran,eq.trans ran,fun _ _ _ => rfl,frame,frame⟩
end Final

#axiomsClean Grouped.equivalent
#axiomsClean Final.equivalent
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardPairIO
