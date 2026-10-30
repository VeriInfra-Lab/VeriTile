import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SafetyParts
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Loop
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Setup

/-! Trace safety of every actual helper iteration, derived along the checked
execution invariant. Inactive addresses and the final unused advance are free. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyLoop
open VeriTile.Triton Memory Preparation AccessSafety
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
noncomputable section
set_option maxHeartbeats 1000000

structure Bounds (bounds : RegionBounds) (c : Config) (BN BD lower upper : Nat) : Prop where
  keys : ∀ i < Loop.count lower upper BN, KeyBound bounds c BN BD (Loop.cursor lower BN i)
  scales : ∀ i < Loop.count lower upper BN, ksbase c BN+Loop.cursor lower BN i/BN < bounds c.p.KS
  values : ∀ i < Loop.count lower upper BN, ValueBound bounds c BN BD (Loop.cursor lower BN i)

theorem loop_safe_from {BM BN BD : Nat} (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper) (i : Nat) (hi : i ≤ Loop.count lower upper BN)
    (hr : Loop.Registers mem s c BM BN BD (Loop.cursor lower BN i) second
      (Loop.after mem c BM BN BD lower second diagonal seed i)) :
    Stmt.forRangeTraceSafe bounds "start_n" (Loop.cursor lower BN i) upper BN
      (sourceIteration BM BN BD c.p.headDim second diagonal) s := by
  rw [Stmt.forRangeTraceSafe,if_neg (by omega)]
  by_cases hlt : i < Loop.count lower upper BN
  · rw [if_pos ((VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor_lt lower upper BN i hN).mpr hlt)]
    let si := s.setReg "start_n" .nat [] (Tile.scalar (Loop.cursor lower BN i))
    have hp : Pointers si c BN BD (Loop.cursor lower BN i) second := by
      refine ⟨?_,?_,?_⟩
      · cases second <;> simpa [si,keyName,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.keyName] using hr.pointers.key
      · cases second <;> simpa [si,scaleName,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.scaleName] using hr.pointers.scale
      · cases second <;> simpa [si,valueName,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory.valueName] using hr.pointers.value
    refine ⟨SafetyParts.iteration_safe bounds si c BM BN BD (Loop.cursor lower BN i) second diagonal hp
      (by simpa [si] using hr.fixed.keys) (by simp [si]) (by simpa [si] using hr.fixed.context)
      (hb.keys i hlt) (hb.scales i hlt) (hb.values i hlt),?_⟩
    obtain ⟨t,et,ht,_⟩ := Loop.invariant_step mem s c BM BN BD lower second diagonal seed hN i hr
    rw [et]
    simpa only [VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor_succ] using
      loop_safe_from bounds mem t c lower upper second diagonal seed hN hb (i+1) (by omega) ht
  · rw [if_neg (fun hcur => hlt ((VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor_lt lower upper BN i hN).mp hcur))]
    trivial
termination_by Loop.count lower upper BN-i

theorem loop_safe {BM BN BD : Nat} (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper)
    (hr : Loop.Registers mem s c BM BN BD lower second seed) :
    Stmt.forRangeTraceSafe bounds "start_n" lower upper BN
      (sourceIteration BM BN BD c.p.headDim second diagonal) s := by
  simpa only [Loop.cursor,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor,Nat.zero_mul,Nat.add_zero] using
    loop_safe_from bounds mem s c lower upper second diagonal seed hN hb 0 (Nat.zero_le _)
      (by simpa only [Loop.after,VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula.closed_zero,
        Loop.cursor,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealLoop.cursor,Nat.zero_mul,Nat.add_zero] using hr)

theorem statement_safe {BM BN BD : Nat} (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (lower upper : Nat) (second diagonal : Bool) (seed : Carry BM BD) (hN : 0 < BN)
    (hb : Bounds bounds c BN BD lower upper)
    (hl : s.regs .nat [] (Setup.lowerName second) = some (Tile.scalar lower))
    (hu : s.regs .nat [] (Setup.upperName second) = some (Tile.scalar upper))
    (hr : Loop.Registers mem s c BM BN BD lower second seed) :
    Stmt.TraceSafe bounds (loop BM BN BD c.p.headDim second diagonal) s := by
  rw [loop,Stmt.TraceSafe]
  refine ⟨by simp [Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],by simp [Op.SafeAt.eq_def],?_⟩
  change s.regs .nat [] (if second then "second_lo" else "first_lo") = some (Tile.scalar lower) at hl
  change s.regs .nat [] (if second then "second_hi" else "first_hi") = some (Tile.scalar upper) at hu
  simp only [evalOp_ref,hl,hu,evalOp_constNat,Tile.scalar]
  exact loop_safe bounds mem s c lower upper second diagonal seed hN hb hr

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyLoop
