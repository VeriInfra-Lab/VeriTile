import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Memory

/-! The actual independent Q/K/V/output pointer tiles. The two row-stride
registers are used exactly where they occur in the repaired source. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Pointers
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def rowExpr (R : RegionName) (BR BD sd : Nat) (baseName rows : RegName) (sr : Op .nat []) : Op .ptr [BR,BD] :=
  .ptrAdd .scalarL (.ptrBase R)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] baseName)
        (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (.ref .nat [BR] rows)) sr))
      (.mul .nat .scalarR (.expandDim ⟨0,by simp⟩ (.ref .nat [BD] "offs_k")) (.constNat sd)))

def keyExpr (p : Parameters) (BD BN : Nat) : Op .ptr [BD,BN] :=
  .ptrAdd .scalarL (.ptrBase p.K)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] "k_offset")
        (.mul .nat .scalarR (.expandDim ⟨1,by simp⟩ (.ref .nat [BD] "offs_k")) (.constNat p.keyDim)))
      (.mul .nat .scalarR (.expandDim ⟨0,by simp⟩ (.ref .nat [BN] "offs_n")) (.ref .nat [] "stride_kn64")))

def code (p : Parameters) (BM BN BD : Nat) : List Stmt :=
  [.assign .nat [BM] "offs_m" (RealPointers.rowExpr BM),
   .assign .nat [BN] "offs_n" (.arange BN),
   .assign .nat [BD] "offs_k" (.arange BD),
   .assign .ptr [BM,BD] "Q_ptrs" (rowExpr p.Q BM BD p.queryDim "qvk_offset" "offs_m" (.constNat p.queryRow)),
   .assign .ptr [] "Q_scale_ptr" (RealPointers.queryScaleExpr p.QS),
   .assign .ptr [BD,BN] "K_ptrs" (keyExpr p BD BN),
   .assign .ptr [] "K_scale_ptr" (RealPointers.keyScaleExpr p.KS),
   .assign .ptr [BN,BD] "V_ptrs" (rowExpr p.V BN BD p.valueDim "v_offset" "offs_n" (.ref .nat [] "stride_vk64")),
   .assign .ptr [BM,BD] "O_block_ptr" (rowExpr p.Out BM BD p.outputDim "o_offset" "offs_m" (.constNat p.outputRow))]

theorem source (p : Parameters) (BM BN BD : Nat) : pointers BM BN BD p = code p BM BN BD := by
  simp only [pointers, Verified.code, Verified.source, TypedSource.repaired,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [code,rowExpr,keyExpr,RealPointers.rowExpr,RealPointers.queryScaleExpr,RealPointers.keyScaleExpr]

theorem eval_row (s : BlockState) (R : RegionName) (BR BD base start strideRow strideDim : Nat)
    (baseName name : RegName) (sr : Op .nat [])
    (hb : s.regs .nat [] baseName = some (Tile.scalar base))
    (hr : s.regs .nat [BR] name = some (RealPointers.rows BR start))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : evalOp sr s = some (Tile.scalar strideRow)) :
    evalOp (rowExpr R BR BD strideDim baseName name sr) s =
      some (RealPointers.rowPointer R BR BD base start strideRow strideDim) := by
  unfold rowExpr
  rw [evalOp.eq_def]
  have hbase : evalOp (.ptrBase R) s = some (Tile.scalar (R.cast,0)) := by simp [evalOp.eq_def]
  have hexr : evalOp (.expandDim ⟨1,by simp⟩ (.ref .nat [BR] name)) s =
      some (Tile.expandDim ⟨1,by simp⟩ (RealPointers.rows BR start)) := by
    simp only [evalOp.eq_def,hr,bind,Option.bind]
    rfl
  have hexd : evalOp (.expandDim ⟨0,by simp⟩ (.ref .nat [BD] "offs_k")) s =
      some (Tile.expandDim ⟨0,by simp⟩ (Tile.vec (fun i : Fin BD => i.val))) := by
    simp only [evalOp.eq_def,hk,bind,Option.bind]
    rfl
  simp only [hbase,evalOp_add,evalOp_mul,hexr,hexd,evalOp_ref,evalOp_constNat,
    hb,hr,hk,hsr,bind,Option.bind]
  erw [hexr,hexd]
  simp only [bind,Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [RealPointers.rowPointer,RealPointers.rows,Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,Tile.expandDim_data,
    Tile.vec,TileShape.dropInsertedIndex,NumericDType.mul,NumericDType.add,-Tile.scalar_eta]

theorem eval_key (s : BlockState) (c : Config) (BD BN : Nat)
    (hb : s.regs .nat [] "k_offset" = some (Tile.scalar (kbase c)))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow)) :
    evalOp (keyExpr c.p BD BN) s = some (keyPointer c BD BN 0) := by
  simp only [keyExpr,evalOp.eq_def,hb,hn,hk,hsr,bind,Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyPointer,Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,Tile.expandDim_data,
    Tile.vec,TileShape.dropInsertedIndex,NumericDType.mul,NumericDType.add,-Tile.scalar_eta]

def writes : List RegName := ["offs_m","offs_n","offs_k","Q_ptrs","Q_scale_ptr","K_ptrs","K_scale_ptr","V_ptrs","O_block_ptr"]

structure Registers (c : Config) (BM BN BD : Nat) (s : BlockState) : Prop where
  rows : s.regs .nat [BM] "offs_m" = some (RealPointers.rows BM (c.startM*BM))
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  head : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))
  query : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD)
  queryScale : s.regs .ptr [] "Q_scale_ptr" = some (queryScalePointer c BM)
  key : s.regs .ptr [BD,BN] "K_ptrs" = some (keyPointer c BD BN 0)
  scale : s.regs .ptr [] "K_scale_ptr" = some (keyScalePointer c BN 0)
  value : s.regs .ptr [BN,BD] "V_ptrs" = some (valuePointer c BN BD 0)
  output : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD)

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM))
    (hqb : s.regs .nat [] "qvk_offset" = some (Tile.scalar (qbase c)))
    (hkb : s.regs .nat [] "k_offset" = some (Tile.scalar (kbase c)))
    (hvb : s.regs .nat [] "v_offset" = some (Tile.scalar (vbase c)))
    (hob : s.regs .nat [] "o_offset" = some (Tile.scalar (obase c)))
    (hqsb : s.regs .nat [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)))
    (hksb : s.regs .nat [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)))
    (hkr : s.regs .nat [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .nat [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t,stepStmts (code c.p BM BN BD) s = some t ∧ Registers c BM BN BD t ∧ Frame writes s t := by
  let s1 := s.setReg "offs_m" .nat [BM] (RealPointers.rows BM (c.startM*BM))
  let s2 := s1.setReg "offs_n" .nat [BN] (Tile.vec (fun i => i.val))
  let s3 := s2.setReg "offs_k" .nat [BD] (Tile.vec (fun i => i.val))
  let s4 := s3.setReg "Q_ptrs" .ptr [BM,BD] (queryPointer c BM BD)
  let s5 := s4.setReg "Q_scale_ptr" .ptr [] (queryScalePointer c BM)
  let s6 := s5.setReg "K_ptrs" .ptr [BD,BN] (keyPointer c BD BN 0)
  let s7 := s6.setReg "K_scale_ptr" .ptr [] (keyScalePointer c BN 0)
  let s8 := s7.setReg "V_ptrs" .ptr [BN,BD] (valuePointer c BN BD 0)
  have hq := eval_row s3 c.p.Q BM BD (qbase c) (c.startM*BM) c.p.queryRow c.p.queryDim "qvk_offset" "offs_m" (.constNat c.p.queryRow)
    (by simp [s3,s2,s1,hqb]) (by simp [s3,s2,s1]) (by simp [s3]) (by simp)
  change evalOp (rowExpr c.p.Q BM BD c.p.queryDim "qvk_offset" "offs_m" (.constNat c.p.queryRow)) s3 =
    some (queryPointer c BM BD) at hq
  have hqs : evalOp (RealPointers.queryScaleExpr c.p.QS) s4 = some (queryScalePointer c BM) := by
    simp only [RealPointers.queryScaleExpr,evalOp.eq_def]
    simp [s4,s3,s2,s1,hqsb,hm,queryScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,NumericDType.add,-Tile.scalar_eta]
  have hk := eval_key s5 c BD BN (by simp [s5,s4,s3,s2,s1,hkb])
    (by simp [s5,s4,s3,s2]) (by simp [s5,s4,s3]) (by simp [s5,s4,s3,s2,s1,hkr])
  have hks : evalOp (RealPointers.keyScaleExpr c.p.KS) s6 = some (keyScalePointer c BN 0) := by
    simp only [RealPointers.keyScaleExpr,evalOp.eq_def]
    simp [s6,s5,s4,s3,s2,s1,hksb,keyScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.scalar,-Tile.scalar_eta]
  have hv : evalOp (rowExpr c.p.V BN BD c.p.valueDim "v_offset" "offs_n" (.ref .nat [] "stride_vk64")) s7 =
      some (valuePointer c BN BD 0) := by
    have h := eval_row s7 c.p.V BN BD (vbase c) 0 c.p.valueRow c.p.valueDim "v_offset" "offs_n" (.ref .nat [] "stride_vk64")
      (by simp [s7,s6,s5,s4,s3,s2,s1,hvb]) (by simp [s7,s6,s5,s4,s3,s2,RealPointers.rows,Tile.vec])
      (by simp [s7,s6,s5,s4,s3]) (by simp [s7,s6,s5,s4,s3,s2,s1,hvr])
    exact h
  have ho := eval_row s8 c.p.Out BM BD (obase c) (c.startM*BM) c.p.outputRow c.p.outputDim "o_offset" "offs_m" (.constNat c.p.outputRow)
    (by simp [s8,s7,s6,s5,s4,s3,s2,s1,hob]) (by simp [s8,s7,s6,s5,s4,s3,s2,s1])
    (by simp [s8,s7,s6,s5,s4,s3]) (by simp)
  change evalOp (rowExpr c.p.Out BM BD c.p.outputDim "o_offset" "offs_m" (.constNat c.p.outputRow)) s8 =
    some (outputPointer c BM BD) at ho
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (RealPointers.eval_rows s BM c.startM hm)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.arange BN) s1 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.arange BD) s2 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some hq),stepStmts.cons_some (stepStmt_assign_eq_some hqs),
    stepStmts.cons_some (stepStmt_assign_eq_some hk),stepStmts.cons_some (stepStmt_assign_eq_some hks),
    stepStmts.cons_some (stepStmt_assign_eq_some hv),stepStmts.cons_some (stepStmt_assign_eq_some ho),stepStmts.nil]
  refine ⟨_,rfl,?_,rfl,rfl,rfl,?_⟩
  · constructor <;> simp [s8,s7,s6,s5,s4,s3,s2,s1]
  · intro dt sh name hn
    simp only [writes,List.mem_cons,not_or] at hn
    simp [s8,s7,s6,s5,s4,s3,s2,s1,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2.1,
      hn.2.2.2.2.2.1,hn.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Pointers
