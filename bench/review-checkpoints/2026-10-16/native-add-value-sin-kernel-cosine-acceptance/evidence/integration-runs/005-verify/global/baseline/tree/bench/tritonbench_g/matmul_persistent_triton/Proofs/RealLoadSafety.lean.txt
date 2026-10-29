import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealAccess

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoadSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Safety of all five actual load-stage statements, following the evaluated
offset/pointer registers and preserving the exact signed K-tail masks. -/
theorem safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (rows : Fin p.BM → Nat) (columns : Fin p.BN → Nat) (ki : Nat)
    (hM : 0<p.M) (hN : 0<p.N) (hb : RealAccess.Bounds p bounds)
    (hk : s.regs .int [] "ki" = some (Tile.scalar (ki : Int)))
    (hm : s.regs .int [p.BM] "offs_am" = some (Tile.vec fun i => (RealTiles.clamp p.M (rows i) : Int)))
    (hn : s.regs .int [p.BN] "offs_bn" = some (Tile.vec fun j => (RealTiles.clamp p.N (columns j) : Int)))
    (hl : s.regs .nat [p.BK] "offs_k_for_mask" = some (Tile.vec Fin.val)) :
    Stmt.TraceSafeList bounds (RealProgram.loads p) s := by
  let t1 := s.setReg "offs_k" .int [p.BK] (Tile.vec fun k => (ki*p.BK+k.val : Nat))
  let t2 := t1.setReg "a_ptrs" .ptr [p.BM,p.BK] (RealLoads.aAddresses p rows ki)
  let t3 := t2.setReg "b_ptrs" .ptr [p.BK,p.BN] (RealLoads.bAddresses p columns ki)
  have h1 : stepStmt (.assign .int [p.BK] "offs_k" (RealLoads.kOffsets p)) s = some t1 := by
    simp only [stepStmt,RealLoads.k_evaluates p s ki hk,bind,Option.bind_some]
    rfl
  have h2 : stepStmt (.assign .ptr [p.BM,p.BK] "a_ptrs" (RealLoads.aPointers p)) t1 = some t2 := by
    rw [stepStmt,RealLoads.a_pointer_evaluates p t1 rows ki (by simp [t1,hm]) (by simp [t1])]
    rfl
  have h3 : stepStmt (.assign .ptr [p.BK,p.BN] "b_ptrs" (RealLoads.bPointers p)) t2 = some t3 := by
    rw [stepStmt,RealLoads.b_pointer_evaluates p t2 columns ki
      (by simp [t2,t1,hn]) (by simp [t2,t1])]
    rfl
  rw [RealLoads.matches_source,RealLoads.code,TraceProofs.cons_of_run_iff h1]
  refine ⟨?_,?_⟩
  · simp [Stmt.TraceSafe,RealLoads.kOffsets,Op.SafeAt.eq_def]
  rw [TraceProofs.cons_of_run_iff h2]
  refine ⟨?_,?_⟩
  · simp [Stmt.TraceSafe,RealLoads.aPointers,Op.SafeAt.eq_def]
  rw [TraceProofs.cons_of_run_iff h3]
  refine ⟨?_,?_⟩
  · simp [Stmt.TraceSafe,RealLoads.bPointers,Op.SafeAt.eq_def]
  apply TraceProofs.assign_cons
  · exact RealAccess.a_safe p bounds t3 rows ki hM hb
      (by simp [t3,t2]) (by simp [t3,t2,t1,hk]) (by simp [t3,t2,t1,hl])
  · intro a
    apply TraceProofs.assign_cons
    · exact RealAccess.b_safe p bounds (t3.setReg "a" .real [p.BM,p.BK] a) columns ki hN hb
        (by simp [t3]) (by simp [t3,t2,t1,hk]) (by simp [t3,t2,t1,hl])
    · intro _
      exact Stmt.TraceSafeList.nil_intro

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealLoadSafety
