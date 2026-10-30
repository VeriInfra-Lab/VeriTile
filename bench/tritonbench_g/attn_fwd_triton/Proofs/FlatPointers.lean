import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatCongruence
import bench.tritonbench_g.attn_fwd_triton.Proofs.Pointers
import bench.tritonbench_g.attn_fwd_triton.Proofs.Relocation

/-! Execute the actual flat translation of every role's pointer constructor.
All roles share A.flat; each original base is retained independently. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatPointers
open VeriTile.Triton Memory Relocation
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem eval_row (A : FlatAlloc) (s : BlockState) (R : RegionName) (BR BD base start strideRow strideDim : Nat)
    (baseName name : RegName) (sr : Op .nat [])
    (hflat : A.flattenOp sr = sr)
    (hb : s.regs .nat [] baseName = some (Tile.scalar base))
    (hr : s.regs .nat [BR] name = some (RealPointers.rows BR start))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : evalOp sr s = some (Tile.scalar strideRow)) :
    evalOp (A.flattenOp (Pointers.rowExpr R BR BD strideDim baseName name sr)) s =
      some (shifted A.flat (A.base R) (RealPointers.rowPointer R BR BD base start strideRow strideDim)) := by
  have h := Pointers.eval_row s R BR BD base start strideRow strideDim baseName name sr hb hr hk hsr
  unfold Pointers.rowExpr at h ⊢
  apply flatten_ptr A .scalarL s R _ _ _ h
  simp only [hflat,FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]

theorem eval_key (A : FlatAlloc) (s : BlockState) (c : Config) (BD BN : Nat)
    (hb : s.regs .nat [] "k_offset" = some (Tile.scalar (kbase c)))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow)) :
    evalOp (A.flattenOp (Pointers.keyExpr c.p BD BN)) s = some (keyPointer (config A c) BD BN 0) := by
  have h := Pointers.eval_key s c BD BN hb hn hk hsr
  rw [← shift_key A c BD BN 0]
  unfold Pointers.keyExpr at h ⊢
  exact flatten_ptr A .scalarL s c.p.K _ _ (by simp only [FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]) h

theorem executes (A : FlatAlloc) (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM))
    (hqb : s.regs .nat [] "qvk_offset" = some (Tile.scalar (qbase c)))
    (hkb : s.regs .nat [] "k_offset" = some (Tile.scalar (kbase c)))
    (hvb : s.regs .nat [] "v_offset" = some (Tile.scalar (vbase c)))
    (hob : s.regs .nat [] "o_offset" = some (Tile.scalar (obase c)))
    (hqsb : s.regs .nat [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)))
    (hksb : s.regs .nat [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)))
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t,stepStmts (A.flattenStmts (Pointers.code c.p BM BN BD)) s = some t ∧
      Pointers.Registers (config A c) BM BN BD t ∧ Frame Pointers.writes s t := by
  let rc := config A c
  let s1 := s.setReg "offs_m" .nat [BM] (RealPointers.rows BM (c.startM*BM))
  let s2 := s1.setReg "offs_n" .nat [BN] (Tile.vec (fun i => i.val))
  let s3 := s2.setReg "offs_k" .nat [BD] (Tile.vec (fun i => i.val))
  let s4 := s3.setReg "Q_ptrs" .ptr [BM,BD] (queryPointer rc BM BD)
  let s5 := s4.setReg "Q_scale_ptr" .ptr [] (queryScalePointer rc BM)
  let s6 := s5.setReg "K_ptrs" .ptr [BD,BN] (keyPointer rc BD BN 0)
  let s7 := s6.setReg "K_scale_ptr" .ptr [] (keyScalePointer rc BN 0)
  let s8 := s7.setReg "V_ptrs" .ptr [BN,BD] (valuePointer rc BN BD 0)
  have hq := eval_row A s3 c.p.Q BM BD (qbase c) (c.startM*BM) c.p.queryRow c.p.queryDim "qvk_offset" "offs_m" (.constNat c.p.queryRow)
    (by simp only [FlatAlloc.flattenOp]) (by simp [s3,s2,s1,hqb]) (by simp [s3,s2,s1]) (by simp [s3]) (by simp)
  change evalOp (A.flattenOp (Pointers.rowExpr c.p.Q BM BD c.p.queryDim "qvk_offset" "offs_m" (.constNat c.p.queryRow))) s3 =
    some (shifted A.flat (A.base c.p.Q) (queryPointer c BM BD)) at hq
  rw [shift_query] at hq
  have hqs0 : evalOp (RealPointers.queryScaleExpr c.p.QS) s4 = some (queryScalePointer c BM) := by
    simp only [RealPointers.queryScaleExpr,evalOp.eq_def]
    simp [s4,s3,s2,s1,hqsb,hm,queryScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,NumericDType.add,-Tile.scalar_eta]
  have hqs := flatten_ptr A .nil s4 c.p.QS _ _ (by simp only [FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]) hqs0
  rw [shift_queryScale] at hqs
  change evalOp (A.flattenOp (RealPointers.queryScaleExpr c.p.QS)) s4 = some (queryScalePointer rc BM) at hqs
  have hk := eval_key A s5 c BD BN (by simp [s5,s4,s3,s2,s1,hkb])
    (by simp [s5,s4,s3,s2]) (by simp [s5,s4,s3]) (by simp [s5,s4,s3,s2,s1,hkr])
  have hks0 : evalOp (RealPointers.keyScaleExpr c.p.KS) s6 = some (keyScalePointer c BN 0) := by
    simp only [RealPointers.keyScaleExpr,evalOp.eq_def]
    simp [s6,s5,s4,s3,s2,s1,hksb,keyScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.scalar,-Tile.scalar_eta]
  have hks := flatten_ptr A .nil s6 c.p.KS _ _ (by simp only [FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]) hks0
  rw [shift_keyScale] at hks
  change evalOp (A.flattenOp (RealPointers.keyScaleExpr c.p.KS)) s6 = some (keyScalePointer rc BN 0) at hks
  have hv := eval_row A s7 c.p.V BN BD (vbase c) 0 c.p.valueRow c.p.valueDim "v_offset" "offs_n" (.ref .nat [] "stride_vk64")
    (by simp only [FlatAlloc.flattenOp]) (by simp [s7,s6,s5,s4,s3,s2,s1,hvb]) (by simp [s7,s6,s5,s4,s3,s2,RealPointers.rows,Tile.vec])
    (by simp [s7,s6,s5,s4,s3]) (by simp [s7,s6,s5,s4,s3,s2,s1,hvr])
  change evalOp (A.flattenOp (Pointers.rowExpr c.p.V BN BD c.p.valueDim "v_offset" "offs_n" (.ref .nat [] "stride_vk64"))) s7 =
    some (shifted A.flat (A.base c.p.V) (valuePointer c BN BD 0)) at hv
  rw [shift_value] at hv
  have ho := eval_row A s8 c.p.Out BM BD (obase c) (c.startM*BM) c.p.outputRow c.p.outputDim "o_offset" "offs_m" (.constNat c.p.outputRow)
    (by simp only [FlatAlloc.flattenOp]) (by simp [s8,s7,s6,s5,s4,s3,s2,s1,hob]) (by simp [s8,s7,s6,s5,s4,s3,s2,s1])
    (by simp [s8,s7,s6,s5,s4,s3]) (by simp)
  change evalOp (A.flattenOp (Pointers.rowExpr c.p.Out BM BD c.p.outputDim "o_offset" "offs_m" (.constNat c.p.outputRow))) s8 =
    some (shifted A.flat (A.base c.p.Out) (outputPointer c BM BD)) at ho
  rw [shift_output] at ho
  have hmflat : A.flattenOp (RealPointers.rowExpr BM) = RealPointers.rowExpr BM := by
    simp only [RealPointers.rowExpr,FlatAlloc.flattenOp,FlatCongruence.vec0,FlatCongruence.vec1,FlatCongruence.dot2,FlatCongruence.sum1,FlatCongruence.max1]
  simp only [Pointers.code,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,hmflat]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealPointers.eval_rows s BM c.startM hm)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (A.flattenOp (.arange BN)) s1 = some (Tile.vec (fun i => i.val)) from by simp [FlatAlloc.flattenOp,evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (A.flattenOp (.arange BD)) s2 = some (Tile.vec (fun i => i.val)) from by simp [FlatAlloc.flattenOp,evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some hq),stepStmts.cons_some (stepStmt_assign_eq_some hqs),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),stepStmts.cons_some (stepStmt_assign_eq_some hks),
    stepStmts.cons_some (stepStmt_assign_eq_some hv),stepStmts.cons_some (stepStmt_assign_eq_some ho),stepStmts.nil]
  refine ⟨_,rfl,?_,rfl,rfl,rfl,?_⟩
  · constructor <;> simp [s8,s7,s6,s5,s4,s3,s2,s1,rc,config]
  · intro dt sh name hn
    simp only [Pointers.writes,List.mem_cons,not_or] at hn
    simp [s8,s7,s6,s5,s4,s3,s2,s1,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2.1,
      hn.2.2.2.2.2.1,hn.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatPointers
