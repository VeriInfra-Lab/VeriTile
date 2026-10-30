import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Native
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Native

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.Native
abbrev Word := BitVec 32
def software : Instructions :=
  ⟨VeriTile.Bench.Repairs.ChunkGateRecurrence.SourceFP.scalar.add,
   VeriTile.Bench.Repairs.ChunkGateRecurrence.SourceFP.scalar.mul,
   VeriTile.Bench.Repairs.ChunkGateRecurrence.SourceFP.scalar.fma⟩

/-- Exactly three local scalar word laws. Compiler choices remain the
value-independent node schedule, not a recurrence or whole-output port. -/
structure LocalConformance (p : Instructions) : Prop where
  add : ∀ a b,p.addRN a b=software.addRN a b
  mul : ∀ a b,p.mulRN a b=software.mulRN a b
  fma : ∀ a b c,p.fmaRN a b c=software.fmaRN a b c

theorem software_eq (p : Instructions) (laws : LocalConformance p) : p=software := by
  cases p with
  | mk add mul fma =>
    have a : add=software.addRN := funext fun x => funext fun y => laws.add x y
    have b : mul=software.mulRN := funext fun x => funext fun y => laws.mul x y
    have c : fma=software.fmaRN := funext fun x => funext fun y => funext fun z => laws.fma x y z
    cases a;cases b;cases c
    rfl

theorem eval_software (p : Instructions) (laws : LocalConformance p) (schedule : Schedule) (term : Term) :
    eval p schedule term=eval software schedule term := by rw [software_eq p laws]

#axiomsClean eval_software
#specNonCircular LocalConformance avoiding [Execution.gridRun,state,dx,dg,eval]
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.Native
