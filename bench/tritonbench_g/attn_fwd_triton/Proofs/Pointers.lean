import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory

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

def rowExpr (R : RegionName) (BR BD sd : Nat) (baseName rows : RegName) (sr : Op .int []) : Op .ptr [BR,BD] :=
  .ptrAdd .scalarL (.ptrBase R) (.castIntToNat
    (.add .int (.consR (.consL .nil))
      (.add .int .scalarL (.ref .int [] baseName)
        (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [BR] rows)) sr))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [BD] "offs_k")) (wide sd))))

def keyExpr (p : Parameters) (BD BN : Nat) : Op .ptr [BD,BN] :=
  .ptrAdd .scalarL (.ptrBase p.K) (.castIntToNat
    (.add .int (.consR (.consL .nil))
      (.add .int .scalarL (.ref .int [] "k_offset")
        (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [BD] "offs_k")) (wide p.keyDim)))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [BN] "offs_n")) (.ref .int [] "stride_kn64"))))

def rowOffsets (BM : Nat) : Op .int [BM] :=
  .add .int .scalarL (.mul .int .nil (.ref .int [] "start_m") (wide BM)) (.castNatToInt (.arange BM))

def queryScaleExpr (QS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase QS) (.castIntToNat (.add .int .nil
    (.ref .int [] "q_scale_offset") (.ref .int [] "start_m")))

def keyScaleExpr (KS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase KS) (.castIntToNat (.ref .int [] "k_scale_offset"))

def code (p : Parameters) (BM BN BD : Nat) : List Stmt :=
  [.assign .int [BM] "offs_m" (rowOffsets BM),
   .assign .int [BN] "offs_n" (.castNatToInt (.arange BN)),
   .assign .int [BD] "offs_k" (.castNatToInt (.arange BD)),
   .assign .ptr [BM,BD] "Q_ptrs" (rowExpr p.Q BM BD p.queryDim "qvk_offset" "offs_m" (wide p.queryRow)),
   .assign .ptr [] "Q_scale_ptr" (queryScaleExpr p.QS),
   .assign .ptr [BD,BN] "K_ptrs" (keyExpr p BD BN),
   .assign .ptr [] "K_scale_ptr" (keyScaleExpr p.KS),
   .assign .ptr [BN,BD] "V_ptrs" (rowExpr p.V BN BD p.valueDim "v_offset" "offs_n" (.ref .int [] "stride_vk64")),
   .assign .ptr [BM,BD] "O_block_ptr" (rowExpr p.Out BM BD p.outputDim "o_offset" "offs_m" (wide p.outputRow))]

theorem source (p : Parameters) (BM BN BD : Nat) : pointers BM BN BD p = code p BM BN BD := by
  simp only [pointers, Verified.code, Verified.source, TypedSource.repaired,
    ControlView.algorithm,Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [ControlView.algorithm,code,rowExpr,keyExpr,rowOffsets,queryScaleExpr,keyScaleExpr,wide]

theorem eval_nat_offset {sh : TileShape} (e : Op .int sh) (s : BlockState) :
    evalOp (.castIntToNat e) s = (do
      let v ← evalOp e s
      some ⟨fun i => (v.data i).toNat⟩) := by
  rw [evalOp.eq_def]

theorem eval_wide (n : Nat) (s : BlockState) : evalOp (wide n) s=some (Tile.scalar (n : Int)) := by
  simp only [wide,evalOp.eq_def,bind,Option.bind]
  rfl

theorem eval_row (s : BlockState) (R : RegionName) (BR BD base start strideRow strideDim : Nat)
    (baseName name : RegName) (sr : Op .int [])
    (hb : s.regs .int [] baseName = some (Tile.scalar base))
    (hr : s.regs .int [BR] name = some (Memory.rows BR start))
    (hk : s.regs .int [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : evalOp sr s = some (Tile.scalar strideRow)) :
    evalOp (rowExpr R BR BD strideDim baseName name sr) s =
      some (RealPointers.rowPointer R BR BD base start strideRow strideDim) := by
  have hexr : evalOp (.expandDim ⟨1,by simp⟩ (.ref .int [BR] name)) s =
      some (Tile.expandDim ⟨1,by simp⟩ (Memory.rows BR start)) := by
    rw [evalOp.eq_def]
    simp only [evalOp_ref,hr,bind,Option.bind]
    rfl
  have hexd : evalOp (.expandDim ⟨0,by simp⟩ (.ref .int [BD] "offs_k")) s =
      some (Tile.expandDim ⟨0,by simp⟩ (Tile.vec (fun i : Fin BD => (i.val : Int)))) := by
    rw [evalOp.eq_def]
    simp only [evalOp_ref,hk,bind,Option.bind]
    rfl
  simp only [rowExpr,evalOp_ptrAdd,eval_nat_offset,evalOp_add,evalOp_mul,hexr,hexd,
    evalOp_ref,evalOp_ptrBase,eval_wide,hb,hr,hk,hsr,bind,Option.bind]
  erw [hexr,hexd]
  simp only [bind,Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp only [RealPointers.rowPointer,Memory.rows,Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,Tile.expandDim_data,
    Tile.vec,TileShape.dropInsertedIndex,NumericDType.mul,NumericDType.add]
  change (R.cast,0 + Int.toNat ((base : Int)+((start+r.val : Nat) : Int)*(strideRow : Int)+
    (d.val : Int)*(strideDim : Int))) = _
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  all_goals rfl

theorem eval_rows (s : BlockState) (BM start : Nat)
    (h : s.regs .int [] "start_m"=some (Tile.scalar (start : Int))) :
    evalOp (rowOffsets BM) s=some (Memory.rows BM (start*BM)) := by
  simp only [rowOffsets,wide,evalOp.eq_def,h,bind,Option.bind]
  congr 1

theorem eval_key (s : BlockState) (c : Config) (BD BN : Nat)
    (hb : s.regs .int [] "k_offset" = some (Tile.scalar (kbase c)))
    (hn : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .int [BD] "offs_k" = some (Tile.vec (fun i => i.val)))
    (hsr : s.regs .int [] "stride_kn64" = some (Tile.scalar c.p.keyRow)) :
    evalOp (keyExpr c.p BD BN) s = some (keyPointer c BD BN 0) := by
  simp only [keyExpr,wide,evalOp.eq_def,hb,hn,hk,hsr,bind,Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp only [keyPointer,Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,Tile.expandDim_data,
    Tile.vec,TileShape.dropInsertedIndex,NumericDType.mul,NumericDType.add]
  change (c.p.K.cast,0 + Int.toNat ((kbase c : Int)+(d.val : Int)*(c.p.keyDim : Int)+
    (j.val : Int)*(c.p.keyRow : Int))) = _
  simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Nat.zero_add]
  all_goals simp

def writes : List RegName := ["offs_m","offs_n","offs_k","Q_ptrs","Q_scale_ptr","K_ptrs","K_scale_ptr","V_ptrs","O_block_ptr"]

structure Registers (c : Config) (BM BN BD : Nat) (s : BlockState) : Prop where
  rows : s.regs .int [BM] "offs_m" = some (Memory.rows BM (c.startM*BM))
  keys : s.regs .int [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  head : s.regs .int [BD] "offs_k" = some (Tile.vec (fun i => i.val))
  query : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD)
  queryScale : s.regs .ptr [] "Q_scale_ptr" = some (queryScalePointer c BM)
  key : s.regs .ptr [BD,BN] "K_ptrs" = some (keyPointer c BD BN 0)
  scale : s.regs .ptr [] "K_scale_ptr" = some (keyScalePointer c BN 0)
  value : s.regs .ptr [BN,BD] "V_ptrs" = some (valuePointer c BN BD 0)
  output : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c BM BD)

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hm : s.regs .int [] "start_m" = some (Tile.scalar c.startM))
    (hqb : s.regs .int [] "qvk_offset" = some (Tile.scalar (qbase c)))
    (hkb : s.regs .int [] "k_offset" = some (Tile.scalar (kbase c)))
    (hvb : s.regs .int [] "v_offset" = some (Tile.scalar (vbase c)))
    (hob : s.regs .int [] "o_offset" = some (Tile.scalar (obase c)))
    (hqsb : s.regs .int [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)))
    (hksb : s.regs .int [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)))
    (hkr : s.regs .int [] "stride_kn64" = some (Tile.scalar c.p.keyRow))
    (hvr : s.regs .int [] "stride_vk64" = some (Tile.scalar c.p.valueRow)) :
    ∃ t,stepStmts (code c.p BM BN BD) s = some t ∧ Registers c BM BN BD t ∧ Frame writes s t := by
  let s1 := s.setReg "offs_m" .int [BM] (Memory.rows BM (c.startM*BM))
  let s2 := s1.setReg "offs_n" .int [BN] (Tile.vec (fun i => i.val))
  let s3 := s2.setReg "offs_k" .int [BD] (Tile.vec (fun i => i.val))
  let s4 := s3.setReg "Q_ptrs" .ptr [BM,BD] (queryPointer c BM BD)
  let s5 := s4.setReg "Q_scale_ptr" .ptr [] (queryScalePointer c BM)
  let s6 := s5.setReg "K_ptrs" .ptr [BD,BN] (keyPointer c BD BN 0)
  let s7 := s6.setReg "K_scale_ptr" .ptr [] (keyScalePointer c BN 0)
  let s8 := s7.setReg "V_ptrs" .ptr [BN,BD] (valuePointer c BN BD 0)
  have hq := eval_row s3 c.p.Q BM BD (qbase c) (c.startM*BM) c.p.queryRow c.p.queryDim "qvk_offset" "offs_m" (wide c.p.queryRow)
    (by simp [s3,s2,s1,hqb]) (by simp [s3,s2,s1]) (by simp [s3]) (by simp [wide,evalOp.eq_def])
  change evalOp (rowExpr c.p.Q BM BD c.p.queryDim "qvk_offset" "offs_m" (wide c.p.queryRow)) s3 =
    some (queryPointer c BM BD) at hq
  have hqs : evalOp (queryScaleExpr c.p.QS) s4 = some (queryScalePointer c BM) := by
    simp only [queryScaleExpr,evalOp.eq_def]
    simp [s4,s3,s2,s1,hqsb,hm,queryScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.bop_data,Tile.scalar,NumericDType.add,-Tile.scalar_eta]
    all_goals simp only [←Int.natCast_add,Int.toNat_natCast]
  have hk := eval_key s5 c BD BN (by simp [s5,s4,s3,s2,s1,hkb])
    (by simp [s5,s4,s3,s2]) (by simp [s5,s4,s3]) (by simp [s5,s4,s3,s2,s1,hkr])
  have hks : evalOp (keyScaleExpr c.p.KS) s6 = some (keyScalePointer c BN 0) := by
    simp only [keyScaleExpr,evalOp.eq_def]
    simp [s6,s5,s4,s3,s2,s1,hksb,keyScalePointer]
    apply Tile.ext
    rintro ⟨⟩
    simp [Tile.ptrAdd_data,Tile.scalar,-Tile.scalar_eta]
    all_goals simp only [Int.toNat_natCast]
  have hv : evalOp (rowExpr c.p.V BN BD c.p.valueDim "v_offset" "offs_n" (.ref .int [] "stride_vk64")) s7 =
      some (valuePointer c BN BD 0) := by
    have h := eval_row s7 c.p.V BN BD (vbase c) 0 c.p.valueRow c.p.valueDim "v_offset" "offs_n" (.ref .int [] "stride_vk64")
      (by simp [s7,s6,s5,s4,s3,s2,s1,hvb]) (by simp [s7,s6,s5,s4,s3,s2,Memory.rows,Tile.vec])
      (by simp [s7,s6,s5,s4,s3]) (by simp [s7,s6,s5,s4,s3,s2,s1,hvr])
    exact h
  have ho := eval_row s8 c.p.Out BM BD (obase c) (c.startM*BM) c.p.outputRow c.p.outputDim "o_offset" "offs_m" (wide c.p.outputRow)
    (by simp [s8,s7,s6,s5,s4,s3,s2,s1,hob]) (by simp [s8,s7,s6,s5,s4,s3,s2,s1])
    (by simp [s8,s7,s6,s5,s4,s3]) (by simp [wide,evalOp.eq_def])
  change evalOp (rowExpr c.p.Out BM BD c.p.outputDim "o_offset" "offs_m" (wide c.p.outputRow)) s8 =
    some (outputPointer c BM BD) at ho
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (eval_rows s BM c.startM hm)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.castNatToInt (.arange BN)) s1 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def,Tile.vec])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.castNatToInt (.arange BD)) s2 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def,Tile.vec])),
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
