import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopContext

/-! A mathematical range template and its generic execution lemmas. The complete repaired quotation is proved separately in Proofs/RealClosedLoop.lean. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopSource
open VeriTile.Triton

def statement (BM BN BD context window complement even : Nat) (offset size : Int) : Stmt :=
  .forRangeInt "start_n" (.ref .int [] "lo") (.ref .int [] "hi") (.constNat BN)
    (RealBody.body BM BN BD context window complement even offset size)

def loop? (index : RegName) : List Stmt → Option Stmt
  | [] => none
  | (.forRangeInt name start stop stride body) :: rest =>
      if name = index then some (.forRangeInt name start stop stride body) else loop? index rest
  | _ :: rest => loop? index rest

theorem loop_erase (index : RegName) (code : List Stmt) :
    loop? index (Stmt.eraseDTypeList code) = (loop? index code).map Stmt.eraseDType := by
  induction code with
  | nil => simp only [Stmt.eraseDTypeList, loop?, Option.map_none]
  | cons st rest ih =>
    cases st <;> simp only [Stmt.eraseDTypeList, Stmt.eraseDType, loop?]
    all_goals try exact ih
    split <;> simp_all [Stmt.eraseDType]

theorem erased_loop (k : ComputeKernel) (index : RegName)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    loop? index k.eraseDType.toAlgKernel.body = (loop? index k.toAlgKernel.body).map Stmt.eraseDType := by
  rw [ComputeKernel.eraseDType, h]
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody, Kernel.eraseDType]
  exact loop_erase index k.toAlgKernel.body

open RealLoop RealLoopFormula RealLoopContext RealScores in
/-- Execute exactly the source's for-range statement to full-context row
formulas. Bounds are read from the actual entry registers once, as in the
interpreter; no successful-loop or kernel-output premise is assumed. -/
theorem statement_executes {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hflag : p.even ≠ 0 → p.context % BN = 0)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset)
    (hloreg : s.regs .int [] "lo" = some (Tile.scalar (p.lo : Int)))
    (hhireg : s.regs .int [] "hi" = some (Tile.scalar p.hi))
    (hregs : Registers p seed 0 s) :
    ∃ s', stepStmt (statement BM BN BD p.context p.window p.complement p.even p.offset p.size) s = some s' ∧
      Registers p (fullResult s p seed) (count p) s' ∧ Frame writes s s' := by
  obtain ⟨s', hs, hr, hframe⟩ := loop_full_context s p seed hN
    (full_blocks_of_source p hN hflag hlo hhi) hlo hhi hregs
  refine ⟨s', ?_, hr, hframe⟩
  simp only [statement, stepForRangeIntAux.forRangeInt_unfold, evalOp_ref, hloreg, hhireg,
    evalOp_constNat, Option.bind_some, Tile.scalar]
  exact hs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopSource
