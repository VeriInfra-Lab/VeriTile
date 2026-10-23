import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.HostContract
namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs
set_option maxHeartbeats 3000000

variable {α β Store Byte : Type} [Inhabited α] {api : RopeEmbedding.Memory.API β Store Byte}
  {embed : β→α} {rq rk : RopeEmbedding.Request} {before : Store} {p : PairPreparation api embed rq rk before}

theorem checked_inputs (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true) (qnonempty : rq.size≠0) (knonempty : rk.size≠0) :
    apart p.first.call.input.base rq.size p.inputK.base rk.size := by
  simp only [inputCheck,Bool.and_eq_true] at accepted
  have sep := accepted.1.1
  simp only [RopeEmbedding.Checked.separate,decide_eq_true_eq] at sep
  have qlow := laws.firstInputLower.resolve_right qnonempty
  have klow := laws.secondInputLower.resolve_right knonempty
  have qhigh := p.first.call.inputFits.1
  have khigh := (p.second before).call.inputFits.1
  rw [p.secondInput] at khigh
  unfold apart
  omega

theorem StorageLaws.firstApartK (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true) :
    apart p.first.call.output.base rq.size p.inputK.base rk.size := by
  by_cases qe : rq.size=0
  · simp [apart,qe,laws.firstEmpty qe]
  by_cases ke : rk.size=0
  · simp [apart,ke,laws.secondInputEmpty ke]
  cases reused : p.first.call.reused
  · have fresh := laws.firstFreshK reused
    have qlow := laws.firstOutputLower.resolve_right qe
    have klow := laws.secondInputLower.resolve_right ke
    have qhigh := p.first.call.outputFits.1
    have khigh := (p.second before).call.inputFits.1
    rw [p.secondInput] at khigh
    unfold apart; omega
  · rw [(p.first.call.reuseIdentity reused).2]
    exact checked_inputs laws accepted qe ke

theorem StorageLaws.outputApart (laws : StorageLaws api embed rq rk before p)
    (accepted : inputCheck api embed rq rk before p=Bool.true) (after : Store) :
    apart p.first.call.output.base rq.size (p.second after).call.output.base rk.size := by
  by_cases qe : rq.size=0
  · simp [apart,qe,laws.firstEmpty qe]
  by_cases ke : rk.size=0
  · simp [apart,ke,laws.secondEmpty after ke]
  cases reused : (p.second after).call.reused
  · have fresh := laws.secondFreshQ after reused
    have qlow := laws.firstOutputLower.resolve_right qe
    have klow := (laws.secondOutputLower after).resolve_right ke
    have qhigh := p.first.call.outputFits.1
    have khigh := (p.second after).call.outputFits.1
    unfold apart; omega
  · rw [((p.second after).call.reuseIdentity reused).2,p.secondInput]
    exact laws.firstApartK accepted

#axiomsClean checked_inputs
#axiomsClean StorageLaws.firstApartK
#axiomsClean StorageLaws.outputApart
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastRopeEmbedding
