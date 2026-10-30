import bench.tritonbench_g.kv_cache_filling.Proofs.Interface
import bench.tritonbench_g.kv_cache_filling.Proofs.Readback
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.KvCacheFilling.Interface
open VeriTile.Triton FP.Structural QuantProfile CopyExecution QuantExecution
noncomputable section
set_option maxHeartbeats 4000000

private theorem map_cell_id {α : Type} (c : FP.Structural.Cell α) : mapCell id c = c := by
  cases c with | mk d v => cases d <;> rfl

theorem implements {α : Type} [Inhabited α] (a : Ops α) (r : Request) (d : Domain r)
    (j : Job) (version : Bool) (pid : Nat) :
    (programIO a r j version pid).ImplementsOn (inputDomain r j) (target a r) := by
  intro vals dom bounds hb s hin
  let t := (invocation a r version pid).initial s
  have hs : (t.mem "src" (t.pids 0)).read .int = (sourceRow r j : Int) :=
    input_source a r j version pid vals dom s hin
  have hd : (t.mem "dst" (t.pids 0)).read .int = (destinationRow r j : Int) :=
    input_destination a r j version pid vals dom s hin
  let e := SharedExecution.executed a r d t (sourceRow r j) (destinationRow r j) hs hd version
  have ex := e.property
  refine ⟨?_,e.val,?_,?_,?_⟩
  · exact (scheduleSafe_singleton bounds _ s).mpr
      (source_safe a r d j version pid vals dom bounds hb s hin)
  · exact (runSchedule_singleton _ _).trans ex
  · have targetEq : result r (SharedTarget.value a r
        (inputRow (Shared.parameters r) t (sourceRow r j) Bool.true)
        (inputRow (Shared.parameters r) t (sourceRow r j) Bool.false)) = target a r vals :=
      result_congr a r d (input_key a r j version pid vals s hin)
        (input_value a r j version pid vals s hin)
    intro o k active
    have values := congrFun (congrFun targetEq o) k
    change mapCell id (e.val.mem (outputs r)[o].buf (outputAddress r j o k)) =
      FP.Structural.Cell.mk (outputs r)[o].dtype (target a r vals o k)
    rw [map_cell_id,← values]
    fin_cases o
    · simpa only [result,SharedTarget.value,rowValue,sourceRowTarget,outputAddress,outputs,
        List.getElem_cons_zero,List.getElem_cons_succ,output,capacity,lanes,Shared.parameters] using
        cache_cell a r d t e.val (sourceRow r j) (destinationRow r j) hs hd version ex Bool.true
          k.1.val k.1.isLt
    · simpa only [result,SharedTarget.value,rowValue,sourceRowTarget,outputAddress,outputs,
        List.getElem_cons_zero,List.getElem_cons_succ,output,capacity,lanes,Shared.parameters] using
        cache_cell a r d t e.val (sourceRow r j) (destinationRow r j) hs hd version ex Bool.false
          k.1.val k.1.isLt
    · simpa only [result,SharedTarget.value,rowValue,sourceRowTarget,outputAddress,outputs,floatValue,
        List.getElem_cons_zero,List.getElem_cons_succ,metaRegion,capacity,lanes,Shared.parameters] using
        metadata_cell a r d t e.val (sourceRow r j) (destinationRow r j) hs hd version ex active Bool.true
          k.1.val k.1.isLt
    · simpa only [result,SharedTarget.value,rowValue,sourceRowTarget,outputAddress,outputs,floatValue,
        List.getElem_cons_zero,List.getElem_cons_succ,metaRegion,capacity,lanes,Shared.parameters] using
        metadata_cell a r d t e.val (sourceRow r j) (destinationRow r j) hs hd version ex active Bool.false
          k.1.val k.1.isLt
  · intro reg i outside
    exact SharedReadback.frame a r d t e.val (sourceRow r j) (destinationRow r j) hs hd version ex reg i
      (outside_source a r j version pid vals reg i outside)

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.Interface
