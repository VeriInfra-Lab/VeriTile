import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Pointers
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Loop

/-! Initial carry and explicit-zero query load from the actual source. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Initialization
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev initial := RealLoopFormula.initial
abbrev writes := RealInit.writes

def code (BM BD D : Nat) : List Stmt :=
  (RealInit.code BM BD 0 0).take 3 ++
    [.assign .real [BM,BD] "q" (.load .real (.ptr (.ref .ptr [BM,BD] "Q_ptrs"))
      (.maskOther (queryMask BM BD D) (.broadcast (.const 0.0) [BM,BD]))),
     .assign .real [] "q_scale" (.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none)]

theorem source (p : Parameters) (BM BN BD : Nat) : initialization BM BN BD p = code BM BD p.headDim := by
  simp only [initialization, Verified.code, Verified.source, TypedSource.repaired,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [code,RealInit.code,queryMask]

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hp : Pointers.Registers c BM BN BD s)
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t, stepStmts (code BM BD c.p.headDim) s = some t ∧
      Iteration.Fixed s t c BM BN BD ∧ Iteration.Carries t (initial BM BD) ∧ Frame writes s t := by
  let sm := s.setReg "m_i" .real [BM] (⟨fun _ => ⊥⟩ : Tile .real [BM])
  let sl := sm.setReg "l_i" .real [BM] (finite (fun _ => 1))
  let sa := sl.setReg "acc" .real [BM,BD] (finite (fun _ => 0))
  let sq := sa.setReg "q" .real [BM,BD] (finite (qvalue s c BM BD))
  have hm : evalOp (.add .real .scalarR (.full [BM] (.const 0)) .negInf) s =
      some (⟨fun _ => ⊥⟩ : Tile .real [BM]) := by
    simp only [evalOp.eq_def,bind,Option.bind]
    rfl
  have hl : evalOp (.add .real .scalarR (.full [BM] (.const 0)) (.const 1.0)) sm =
      some (finite (fun _ : TileIndex [BM] => 1)) := by
    simp only [evalOp.eq_def,bind,Option.bind]
    congr 1
    apply Tile.ext
    intro i
    norm_num [Tile.bop_data,Tile.scalar,NumericDType.add,WithBot.realAdd,finite,-Tile.scalar_eta]
  have ha : evalOp (.full [BM,BD] (.const 0)) sl = some (finite (fun _ => 0)) := by
    simp only [evalOp.eq_def,bind,Option.bind]
    rfl
  have hq := query_load sa c BM BD (by simp [sa,sl,sm,hp.query])
    (by simpa [sa,sl,sm,RealPointers.rows] using hp.rows) (by simp [sa,sl,sm,hctx])
  rw [Preparation.q_same_memory sa s c BM BD rfl] at hq
  have hqs := scale_load sq c.p.QS (qsbase c BM+c.startM) "Q_scale_ptr"
    (by simpa [sq,sa,sl,sm,queryScalePointer] using hp.queryScale)
  change evalOp (.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) sq =
    some (Tile.scalar (some (s.readMem c.p.QS (qsbase c BM+c.startM)))) at hqs
  simp only [code,RealInit.code,List.take_succ_cons,List.take_zero,List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hm),stepStmts.cons_some (stepStmt_assign_eq_some hl),
    stepStmts.cons_some (stepStmt_assign_eq_some ha),stepStmts.cons_some (stepStmt_assign_eq_some hq),
    stepStmts.cons_some (stepStmt_assign_eq_some hqs),stepStmts.nil]
  refine ⟨_,rfl,?_,?_,rfl,rfl,rfl,?_⟩
  · constructor
    · rfl
    · simp [sq]
    · simp
    · simpa [sq,sa,sl,sm,RealPointers.rows] using hp.rows
    · simp [sq,sa,sl,sm,hp.keys]
    · simp [sq,sa,sl,sm,hctx]
    · simp [sq,sa,sl,sm,hkr]
    · simp [sq,sa,sl,sm,hvr]
  · constructor <;> simp [sq,sa,sl,sm,initial,RealLoopFormula.initial]
  · intro dt sh name hn
    simp only [writes,RealInit.writes,List.mem_cons,not_or] at hn
    simp [sq,sa,sl,sm,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Initialization
