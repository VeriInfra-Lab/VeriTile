import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyLoop
import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyPrologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.TwoLoops

/-! Compose safety for both actual helper calls and the intervening pointer
reset, following the same complete execution witnesses. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyMiddle
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000

theorem safe (bounds : RegionBounds) (mem s : BlockState) (c : Config) (BM BN BD : Nat)
    (diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (f : Iteration.Fixed mem s c BM BN BD) (hc : Iteration.Carries s seed)
    (caller : FirstSetup.Caller s c BN BD)
    (hb1 : SafetyLoop.Bounds bounds c BN BD (FirstSetup.firstLower c.startM BM diagonal)
      (FirstSetup.firstUpper c.startM BM c.p.context diagonal))
    (hb2 : SafetyLoop.Bounds bounds c BN BD (c.startM*BM) (min ((c.startM+1)*BM) c.p.context)) :
    Stmt.TraceSafeList bounds (TwoLoops.code BM BN BD c.p.headDim diagonal) s := by
  obtain ⟨sa,hastep,halo,hahi,haptr,haf⟩ := FirstSetup.executes s c BM BN BD diagonal caller f.context f.keyStride f.valueStride
  have fa := TwoLoops.fixed_setup f haf
  have hca := TwoLoops.carry_setup hc haf
  have hasafe := SafetyLoop.statement_safe bounds mem sa c _ _ Bool.false diagonal seed hN hb1 halo hahi ⟨fa,hca,haptr⟩
  obtain ⟨sb,hbloop,hbr,hbf⟩ := Loop.executes mem sa c BM BN BD
    (FirstSetup.firstLower c.startM BM diagonal) (FirstSetup.firstUpper c.startM BM c.p.context diagonal)
    Bool.false diagonal seed hN ⟨fa,hca,haptr⟩
  have hbstep := TwoLoops.statement BM BN BD c.p.headDim _ _ Bool.false diagonal sa sb halo hahi hbloop
  have hbframe : Frame TwoLoops.writes s sb :=
    (haf.mono (by intro name hn; simp only [TwoLoops.writes,List.mem_append]; tauto)).trans
      (hbf.mono (by intro name hn; simp only [TwoLoops.writes,List.mem_append]; tauto))
  have cb := TwoLoops.caller_frame caller hbframe
  obtain ⟨sc,hcstep,hclo,hchi,hcptr,hcf⟩ := Setup.executes sb c BM BN BD Bool.true
    cb.row cb.key cb.scale cb.value hbr.fixed.context hbr.fixed.keyStride hbr.fixed.valueStride
  have fc := TwoLoops.fixed_setup hbr.fixed hcf
  have hcc := TwoLoops.carry_setup hbr.carry hcf
  have hcsafe := SafetyLoop.statement_safe bounds mem sc c _ _ Bool.true diagonal
    (TwoLoops.firstResult mem c BM BN BD diagonal seed) hN hb2 hclo hchi ⟨fc,hcc,hcptr⟩
  rw [TwoLoops.code,List.append_assoc,List.append_assoc]
  apply append_of_run (SafetyPrologue.first_setup_safe bounds s BM BN BD diagonal) hastep
  simp only [List.cons_append,List.nil_append]
  apply cons_of_run hasafe hbstep
  apply append_of_run (SafetyPrologue.setup_safe bounds sb BM BN BD Bool.true) hcstep
  exact Stmt.TraceSafeList.cons_intro hcsafe (fun _ _ => Stmt.TraceSafeList.nil_intro)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyMiddle
