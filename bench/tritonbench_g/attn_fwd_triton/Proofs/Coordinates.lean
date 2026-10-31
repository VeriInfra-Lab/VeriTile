import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory

/-! Actual leading metadata assignments. Q/K/V/Out bases are computed from
independent batch/head strides, and cdiv reads the widened context register. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Coordinates
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def ceilExpr (B : Nat) : Op .int [] :=
  .tritonDiv .int .nil (.add .int .nil (.ref .int [] "n_ctx")
    (.sub .int .nil (wide B) (.constInt 1))) (wide B)

def code (p : Parameters) (BM BN : Nat) : List Stmt :=
  [.assign .int [] "n_ctx" (wide p.context),
   .assign .int [] "stride_kn64" (wide p.keyRow),
   .assign .int [] "stride_vk64" (wide p.valueRow),
   .assign .int [] "start_m" (.castNatToInt (.programId 0)),
   .assign .int [] "off_hz" (.castNatToInt (.programId 1)),
   .assign .int [] "off_z" (.tritonDiv .int .nil (.ref .int [] "off_hz") (wide p.heads)),
   .assign .int [] "off_h" (.tritonMod .int .nil (.ref .int [] "off_hz") (wide p.heads)),
   .assign .int [] "qvk_offset" (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.queryBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.queryHead))),
   .assign .int [] "k_offset" (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.keyBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.keyHead))),
   .assign .int [] "v_offset" (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.valueBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.valueHead))),
   .assign .int [] "o_offset" (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.outputBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.outputHead))),
   .assign .int [] "q_scale_offset" (.mul .int .nil (.ref .int [] "off_hz") (ceilExpr BM)),
   .assign .int [] "k_scale_offset" (.mul .int .nil (.ref .int [] "off_hz") (ceilExpr BN))]

def configuration (s : BlockState) (p : Parameters) : Config :=
  {p := p,z := s.pids 1/p.heads,h := s.pids 1%p.heads,startM := s.pids 0}

structure Values where
  (context keyStride valueStride row headBatch batch head queryBase keyBase valueBase outputBase queryScaleBase keyScaleBase : Nat)

def values (s : BlockState) (p : Parameters) (BM BN : Nat) : Values where
  context := p.context
  keyStride := p.keyRow
  valueStride := p.valueRow
  row := s.pids 0
  headBatch := s.pids 1
  batch := s.pids 1/p.heads
  head := s.pids 1%p.heads
  queryBase := (s.pids 1/p.heads)*p.queryBatch+(s.pids 1%p.heads)*p.queryHead
  keyBase := (s.pids 1/p.heads)*p.keyBatch+(s.pids 1%p.heads)*p.keyHead
  valueBase := (s.pids 1/p.heads)*p.valueBatch+(s.pids 1%p.heads)*p.valueHead
  outputBase := (s.pids 1/p.heads)*p.outputBatch+(s.pids 1%p.heads)*p.outputHead
  queryScaleBase := s.pids 1*((p.context+BM-1)/BM)
  keyScaleBase := s.pids 1*((p.context+BN-1)/BN)

structure Registers (v : Values) (s : BlockState) : Prop where
  context : s.regs .int [] "n_ctx" = some (Tile.scalar v.context)
  keyStride : s.regs .int [] "stride_kn64" = some (Tile.scalar v.keyStride)
  valueStride : s.regs .int [] "stride_vk64" = some (Tile.scalar v.valueStride)
  row : s.regs .int [] "start_m" = some (Tile.scalar v.row)
  headBatch : s.regs .int [] "off_hz" = some (Tile.scalar v.headBatch)
  batch : s.regs .int [] "off_z" = some (Tile.scalar v.batch)
  head : s.regs .int [] "off_h" = some (Tile.scalar v.head)
  queryBase : s.regs .int [] "qvk_offset" = some (Tile.scalar v.queryBase)
  keyBase : s.regs .int [] "k_offset" = some (Tile.scalar v.keyBase)
  valueBase : s.regs .int [] "v_offset" = some (Tile.scalar v.valueBase)
  outputBase : s.regs .int [] "o_offset" = some (Tile.scalar v.outputBase)
  queryScaleBase : s.regs .int [] "q_scale_offset" = some (Tile.scalar v.queryScaleBase)
  keyScaleBase : s.regs .int [] "k_scale_offset" = some (Tile.scalar v.keyScaleBase)

def writes : List RegName := ["n_ctx","stride_kn64","stride_vk64","start_m","off_hz","off_z","off_h","qvk_offset","k_offset","v_offset","o_offset","q_scale_offset","k_scale_offset"]

theorem source (p : Parameters) (BM BN BD : Nat) :
    coordinates BM BN BD p = code p BM BN := by
  simp only [coordinates, Verified.code, Verified.source, TypedSource.repaired,
    Bool.false_eq_true, ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp [ControlView.algorithm,code,ceilExpr,wide]

theorem executes (s : BlockState) (p : Parameters) (BM BN : Nat) :
    ∃ t, stepStmts (code p BM BN) s = some t ∧ Registers (values s p BM BN) t ∧ Frame writes s t := by
  let v := values s p BM BN
  let s0 := s.setReg "n_ctx" .int [] (Tile.scalar v.context)
  let s1 := s0.setReg "stride_kn64" .int [] (Tile.scalar v.keyStride)
  let s2 := s1.setReg "stride_vk64" .int [] (Tile.scalar v.valueStride)
  let s3 := s2.setReg "start_m" .int [] (Tile.scalar v.row)
  let s4 := s3.setReg "off_hz" .int [] (Tile.scalar v.headBatch)
  let s5 := s4.setReg "off_z" .int [] (Tile.scalar v.batch)
  let s6 := s5.setReg "off_h" .int [] (Tile.scalar v.head)
  let s7 := s6.setReg "qvk_offset" .int [] (Tile.scalar v.queryBase)
  let s8 := s7.setReg "k_offset" .int [] (Tile.scalar v.keyBase)
  let s9 := s8.setReg "v_offset" .int [] (Tile.scalar v.valueBase)
  let s10 := s9.setReg "o_offset" .int [] (Tile.scalar v.outputBase)
  let s11 := s10.setReg "q_scale_offset" .int [] (Tile.scalar v.queryScaleBase)
  let s12 := s11.setReg "k_scale_offset" .int [] (Tile.scalar v.keyScaleBase)
  have h0 : evalOp (wide p.context) s = some (Tile.scalar v.context) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,v,values]
  have h1 : evalOp (wide p.keyRow) s0 = some (Tile.scalar v.keyStride) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s0, v,values]
  have h2 : evalOp (wide p.valueRow) s1 = some (Tile.scalar v.valueStride) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s1, s0, v,values]
  have h3 : evalOp (.castNatToInt (.programId 0)) s2 = some (Tile.scalar v.row) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s2, s1, s0, v,values]
  have h4 : evalOp (.castNatToInt (.programId 1)) s3 = some (Tile.scalar v.headBatch) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s3, s2, s1, s0, v,values]
  have h5 : evalOp (.tritonDiv .int .nil (.ref .int [] "off_hz") (wide p.heads)) s4 = some (Tile.scalar v.batch) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals rfl
  have h6 : evalOp (.tritonMod .int .nil (.ref .int [] "off_hz") (wide p.heads)) s5 = some (Tile.scalar v.head) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals try rw [←Int.emod_def]
    all_goals rfl
  have h7 : evalOp (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.queryBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.queryHead))) s6 = some (Tile.scalar v.queryBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals rfl
  have h8 : evalOp (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.keyBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.keyHead))) s7 = some (Tile.scalar v.keyBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s7, s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals rfl
  have h9 : evalOp (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.valueBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.valueHead))) s8 = some (Tile.scalar v.valueBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s8, s7, s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals rfl
  have h10 : evalOp (.add .int .nil (.mul .int .nil (.ref .int [] "off_z") (wide p.outputBatch)) (.mul .int .nil (.ref .int [] "off_h") (wide p.outputHead))) s9 = some (Tile.scalar v.outputBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals rfl
  have h11 : evalOp (.mul .int .nil (.ref .int [] "off_hz") (ceilExpr BM)) s10 = some (Tile.scalar v.queryScaleBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals
      have weighted := congrArg (fun z : Int => (s.pids 1 : Int)*z) (ControlView.signed_cdiv p.context BM)
      simpa [mul_ite,Int.natCast_ediv] using weighted
  have h12 : evalOp (.mul .int .nil (.ref .int [] "off_hz") (ceilExpr BN)) s11 = some (Tile.scalar v.keyScaleBase) := by
    simp [evalOp.eq_def,ceilExpr,wide,NumericDType.add,NumericDType.sub,NumericDType.mul,IntegralDType.floorDiv,IntegralDType.mod,ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,Int.natCast_emod,s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, v,values]
    all_goals apply Tile.ext
    all_goals intro i; cases i
    all_goals simp [Tile.select_data,Tile.cop_data,Tile.bop_data,Tile.scalar,NumericDType.add,
      NumericDType.sub,NumericDType.mul,ComparableDType.lt,IntegralDType.floorDiv,
      ControlView.signed_div,ControlView.signed_mod,ControlView.signed_cdiv,Int.natCast_ediv,
      Int.natCast_emod,-Tile.scalar_eta]
    all_goals try simp only [←Int.natCast_add,←Int.natCast_mul]
    all_goals
      have weighted := congrArg (fun z : Int => (s.pids 1 : Int)*z) (ControlView.signed_cdiv p.context BN)
      simpa [mul_ite,Int.natCast_ediv] using weighted
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h0),
    stepStmts.cons_some (stepStmt_assign_eq_some h1),
    stepStmts.cons_some (stepStmt_assign_eq_some h2),
    stepStmts.cons_some (stepStmt_assign_eq_some h3),
    stepStmts.cons_some (stepStmt_assign_eq_some h4),
    stepStmts.cons_some (stepStmt_assign_eq_some h5),
    stepStmts.cons_some (stepStmt_assign_eq_some h6),
    stepStmts.cons_some (stepStmt_assign_eq_some h7),
    stepStmts.cons_some (stepStmt_assign_eq_some h8),
    stepStmts.cons_some (stepStmt_assign_eq_some h9),
    stepStmts.cons_some (stepStmt_assign_eq_some h10),
    stepStmts.cons_some (stepStmt_assign_eq_some h11),
    stepStmts.cons_some (stepStmt_assign_eq_some h12),stepStmts.nil]
  refine ⟨_,rfl,?_,rfl,rfl,rfl,?_⟩
  · constructor <;> simp [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,v]
  · intro dt sh name hn
    simp only [writes,List.mem_cons,not_or] at hn
    simp [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,hn.1,hn.2.1,hn.2.2.1,hn.2.2.2.1,hn.2.2.2.2.1,hn.2.2.2.2.2.1,hn.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.2.2.2.1,hn.2.2.2.2.2.2.2.2.2.2.2.2.1]

/-- The decoded head coordinate reconstructs the packed scale row, including
zero-head metadata (empty tensors do not launch). -/
theorem packed_head (s : BlockState) (p : Parameters) :
    (configuration s p).z*p.heads+(configuration s p).h = s.pids 1 := by
  simpa only [configuration,Nat.mul_comm] using Nat.div_add_mod (s.pids 1) p.heads

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Coordinates
