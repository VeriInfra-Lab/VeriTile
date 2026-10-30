import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.RealIndexBridge

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem run_prelude (s : BlockState) :
    ∃ t,stepStmts prelude s=some t ∧ Ready (s.pids 0) t := by
  simp [-Tile.scalar_eta,prelude,stepStmts,stepStmt,evalOp,
    Ready]

theorem loop (input output : RegionName) (T BT pid cur stop : Nat) (a b : BlockState)
    (related : Rel a b) (ready : Ready pid b) :
    ∃ x y,
      stepForRangeIntAux "i_t" cur stop 1 (actualBody input output T BT) a=some x ∧
      stepForRangeAux "i_t" cur stop 1 (ccLoopBody input output T BT) b=some y ∧
      Rel x y ∧ Ready pid y := by
  by_cases hlt : cur<stop
  · obtain ⟨midA,midB,ha,hb,hr,hready⟩ := body input output T BT pid cur a b related ready
    obtain ⟨x,y,hx,hy,hrel,hready⟩ := loop input output T BT pid (cur+1) stop midA midB hr hready
    refine ⟨x,y,?_,?_,hrel,hready⟩
    · rw [stepForRangeIntAux,if_neg (by decide : ¬(1:Nat)=0),if_pos (by exact_mod_cast hlt),ha]
      simpa using hx
    · rw [stepForRangeAux,if_neg (by decide : ¬(1:Nat)=0),if_pos hlt,hb]
      exact hy
  · refine ⟨a,b,?_,?_,related,ready⟩
    · rw [stepForRangeIntAux,if_neg (by decide : ¬(1:Nat)=0),if_neg (by exact_mod_cast hlt)]
    · rw [stepForRangeAux,if_neg (by decide : ¬(1:Nat)=0),if_neg hlt]
termination_by stop-cur
 decreasing_by omega

theorem complete_related (input output : RegionName) (T BT : Nat) (s : BlockState) :
    ∃ x y,exec (actual input output T BT) s=some x ∧
      exec (reference input output T BT) s=some y ∧ Rel x y := by
  obtain ⟨initial,hpre,hready⟩ := run_prelude s
  obtain ⟨x,y,hx,hy,hr,_⟩ := loop input output T BT (s.pids 0) 0 ((T+BT-1)/BT)
    initial initial ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩ hready
  simp only [Nat.cast_zero] at hx
  refine ⟨x,y,?_,?_,hr⟩
  · change stepStmts (actual input output T BT).body s=some x
    rw [actual_body]
    rw [stepStmts.append_some hpre]
    simp only [stepStmts,stepStmt,evalOp.eq_def,bind,pure,Option.bind,Tile.scalar,hx]
  · change stepStmts (reference input output T BT).body s=some y
    rw [reference_body]
    rw [stepStmts.append_some hpre]
    simp only [stepStmts,stepStmt,ccStopOp,evalOp.eq_def,bind,pure,Option.bind,Tile.scalar,
      Tile.bop,NumericDType.add,NumericDType.sub,NumericDType.div,Broadcast.leftIndex,Broadcast.rightIndex,hy]

/-- Actual repaired source, independent inclusive Real prefix, and full
cell frame. The existing complete mathematical proof is reused only after
the changed source's complete execution relation has been established. -/
theorem complete (input output : RegionName) (T : Nat) (cfg : Config)
    (s : BlockState) (separate : output≠input) :
    ∃ t,exec (repairedKernel .fp32 .fp32 input output T cfg.block).eraseDType.toAlgKernel s=some t ∧
      (∀ i,i<T → t.readMem output (s.pids 0*T+i)=
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i),s.readMem input (s.pids 0*T+j)) ∧
      (∀ rg off,(rg≠output ∨ ∀ i,i<T → off≠s.pids 0*T+i) → t.mem rg off=s.mem rg off) := by
  obtain ⟨x,y,hx,hy,hr⟩ := complete_related input output T cfg.block s
  obtain ⟨z,hz,hv,hf⟩ := RealRow.complete input output .fp32 .fp32 T cfg s separate
  have same : z=y := Option.some.inj (hz.symm.trans hy)
  subst z
  refine ⟨x,hx,?_,?_⟩
  · intro i hi
    simpa only [BlockState.readMem,hr.1] using hv i hi
  · intro rg off hn
    exact (congrFun (congrFun hr.1 rg) off).trans (hf rg off hn)

#axiomsClean loop
#axiomsClean complete_related
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealIndexBridge
