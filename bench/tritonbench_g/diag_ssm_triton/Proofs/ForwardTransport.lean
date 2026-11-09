import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Semantics.Integer
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_RealReference
import bench.tritonbench_g.diag_ssm_triton.Proofs.Mathematical_Erasure
import VeriTile.Triton.Memory.FlattenTolerant

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardTransport
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev source (p : ForwardLayout) (length : Nat) :=
  (Erasure.forward Bool.false p.s p.x p.l p.y length p.batch p.dim p.B).toAlgKernel

def before (p : ForwardLayout) (length : Nat) : List Stmt :=
  (source p length).body.take 8

def body (p : ForwardLayout) : List Stmt :=
  match (source p 0).body.drop 8 with
  | [.forRangeInt _ _ _ _ loop] => loop
  | _ => []

def columns (p : ForwardLayout) (u : BlockState) (j : Fin p.B) : Nat :=
  u.pids 0*p.B+j.val

def mask (p : ForwardLayout) (u : BlockState) : Tile .bool [p.B] :=
  Tile.vec (fun j=>decide (columns p u j<p.batch*p.dim))

/-- Only live recurrence registers are related. Source-only cast registers
remain in the actual execution. All observable state is preserved exactly. -/
structure Related (p : ForwardLayout) (u a b : BlockState) : Prop where
  observations : a.ObsAgree b
  sourceColumns : a.regs .int [p.B] "col_offsets"=
    some (Tile.vec fun j=>((columns p u j : Nat) : Int))
  modelColumns : b.regs .nat [p.B] "col_offsets"=
    some (Tile.vec fun j=>columns p u j)
  sourceMask : a.regs .bool [p.B] "mask"=some (mask p u)
  modelMask : b.regs .bool [p.B] "mask"=some (mask p u)
  carried : ∃ v, a.regs .real [p.B] "s"=some v ∧ b.regs .real [p.B] "s"=some v
  transition : ∃ v, a.regs .real [p.B] "Lambda"=some v ∧ b.regs .real [p.B] "Lambda"=some v

/-- The witness is obtained by running the complete actual signed prefix. -/
def setup (p : ForwardLayout) (length : Nat) (u : BlockState) :
    {a // stepStmts (before p length) u=some a} := by
  refine ⟨?_,?_⟩
  swap
  unfold before source Erasure.forward TypedSource.forward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  simp only [stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  try rfl


def modelSetup (p : ForwardLayout) (u : BlockState) :
    {b // stepStmts (RealReference.diagSsmForwardPreLoop p.s p.l p.batch p.dim p.B) u=some b} := by
  refine ⟨?_,?_⟩
  swap
  simp only [RealReference.diagSsmForwardPreLoop,stepStmts,stepStmt,evalOp.eq_def,
    Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.add,IntegralDType.mod,ComparableDType.lt,Region.cast,
    bind,pure,Option.bind,Option.map,Option.pure_def,
    ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  try rfl

theorem setup_related (p : ForwardLayout) (length : Nat) (u : BlockState) :
    Related p u (setup p length u).val (modelSetup p u).val := by
  constructor
  · exact ⟨rfl,rfl,rfl,rfl⟩
  all_goals
    simp only [setup,modelSetup,columns,mask,BlockState.ObsAgree,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
      BlockState.setReg_mem,BlockState.setReg_pids,BlockState.setReg_numPids,
      BlockState.setReg_undef,Tile.vec,Tile.scalar,Tile.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,
      Int.ofNat_eq_natCast,←Int.natCast_add,←Int.natCast_mul,←Int.ofNat_tmod,
      Int.toNat_natCast,Int.ofNat_lt,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,
      Bool.false_eq_true,Bool.true_eq_false]
  all_goals first | rfl | exact ⟨_,rfl,rfl⟩



def sourceNext (p : ForwardLayout) (base : BlockState) (n : Nat) (a : BlockState)
    (carried transition : Tile .real [p.B])
    (hc : a.regs .int [p.B] "col_offsets"=
      some (Tile.vec fun j=>((columns p base j : Nat) : Int)))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "s"=some carried)
    (hl : a.regs .real [p.B] "Lambda"=some transition) :
    {next // stepStmts (body p) (a.setReg "t" .int [] (Tile.scalar (n : Int)))=some next} := by
  refine ⟨?_,?_⟩
  swap
  unfold body source Erasure.forward TypedSource.forward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  simp only [stepStmts,stepStmt,↓eval_truncMod,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right,
    NumericDType.mul,NumericDType.add,NumericDType.sub,IntegralDType.mod,IntegralDType.floorDiv,
    ComparableDType.lt,Region.cast,bind,pure,Option.bind,Option.map,Option.pure_def,
    hc,hm,hs,hl,columns,mask,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  try rfl



def modelNext (p : ForwardLayout) (base : BlockState) (n : Nat) (a : BlockState)
    (carried transition : Tile .real [p.B])
    (hc : a.regs .nat [p.B] "col_offsets"=some (Tile.vec fun j=>columns p base j))
    (hm : a.regs .bool [p.B] "mask"=some (mask p base))
    (hs : a.regs .real [p.B] "s"=some carried)
    (hl : a.regs .real [p.B] "Lambda"=some transition) :
    {next // stepStmts (RealReference.diagSsmForwardLoopBody p.x p.y p.batch p.dim p.B)
      (a.setReg "t" .nat [] (Tile.scalar n))=some next} := by
  refine ⟨?_,?_⟩
  swap
  simp only [RealReference.diagSsmForwardLoopBody,
    stepStmts,stepStmt,evalOp.eq_def,Tile.scalar,Tile.vec,Tile.bop,Tile.cop,
    Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.mul,NumericDType.add,IntegralDType.mod,ComparableDType.lt,Region.cast,
    bind,pure,Option.bind,Option.map,Option.pure_def,
    hc,hm,hs,hl,columns,mask,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
    BlockState.setReg_pids,BlockState.setReg_readMemValue,BlockState.setReg_readMem,
    BlockState.readMemValue_real,
    String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  try rfl

theorem write_observations {a b : BlockState} (obs : a.ObsAgree b)
    (r : RegionName) (offset : Nat) (value : TileCarrier .real) :
    (a.writeMemTyped .real r offset value).ObsAgree (b.writeMemTyped .real r offset value) := by
  refine ⟨?_,obs.pids,obs.numPids,obs.undef⟩
  funext q i
  simp only [BlockState.writeMemTyped,BlockState.writeMemAs,obs.mem]

theorem store_observations {I : Type} (r : RegionName) (offset : I → Nat)
    (value : I → TileCarrier .real) (active : I → Bool) (indices : List I)
    (a b : BlockState) (obs : a.ObsAgree b) :
    (indices.foldl (fun s i=>if active i then s.writeMemTyped .real r (offset i) (value i) else s) a).ObsAgree
    (indices.foldl (fun s i=>if active i then s.writeMemTyped .real r (offset i) (value i) else s) b) := by
  induction indices generalizing a b with
  | nil => exact obs
  | cons i rest ih =>
    rw [List.foldl_cons,List.foldl_cons]
    cases active i <;> simp only [Bool.false_eq_true,if_false,if_true]
    · exact ih a b obs
    · exact ih _ _ (write_observations obs r (offset i) (value i))



theorem next_related (p : ForwardLayout) (base : BlockState) (n : Nat) (a b : BlockState)
    (carried transition : Tile .real [p.B]) (related : Related p base a b)
    (ha : a.regs .real [p.B] "s"=some carried)
    (hb : b.regs .real [p.B] "s"=some carried)
    (hla : a.regs .real [p.B] "Lambda"=some transition)
    (hlb : b.regs .real [p.B] "Lambda"=some transition) :
    Related p base
      (sourceNext p base n a carried transition related.sourceColumns related.sourceMask ha hla).val
      (modelNext p base n b carried transition related.modelColumns related.modelMask hb hlb).val := by
  constructor
  all_goals
    simp only [sourceNext,modelNext,columns,mask,
      ↓BlockState.foldl_writeMemTyped_masked_regs,
      ↓BlockState.setReg_same,↓BlockState.setReg_ne_name,
      Tile.scalar,Tile.vec,Tile.bop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,
      Int.ofNat_eq_natCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast,
      Nat.mul_assoc,related.observations.readMem_eq,
      ha,hb,hla,hlb,related.sourceColumns,related.modelColumns,related.sourceMask,related.modelMask,
      String.reduceEq,ne_eq,not_false_eq_true,if_true,if_false,Bool.false_eq_true,Bool.true_eq_false]
  all_goals first
    | exact store_observations _ _ _ _ _ _ _ related.observations
    | rfl
    | exact ⟨_,rfl,rfl⟩



theorem loop_related (p : ForwardLayout) (base : BlockState) (stop start : Nat)
    (bound : start≤stop) (a b : BlockState) (related : Related p base a b) :
    ∃ x y,
      stepForRangeIntAux "t" (start : Int) (stop : Int) 1 (body p) a=some x ∧
      stepForLoopAux "t" start stop
        (RealReference.diagSsmForwardLoopBody p.x p.y p.batch p.dim p.B) b=some y ∧
      Related p base x y := by
  by_cases active : start<stop
  · have activeInt : (start : Int)<stop := by exact_mod_cast active
    rw [stepForRangeIntAux.step_lt (by decide) activeInt,stepForLoopAux.step_lt active]
    obtain ⟨carried,ha,hb⟩:=related.carried
    obtain ⟨transition,hla,hlb⟩:=related.transition
    let nextA:=sourceNext p base start a carried transition related.sourceColumns related.sourceMask ha hla
    let nextB:=modelNext p base start b carried transition related.modelColumns related.modelMask hb hlb
    have nextRelated : Related p base nextA.val nextB.val :=
      next_related p base start a b carried transition related ha hb hla hlb
    obtain ⟨x,y,hx,hy,hr⟩:=loop_related p base stop (start+1) (by omega) nextA.val nextB.val nextRelated
    refine ⟨x,y,?_,?_,hr⟩
    · rw [nextA.property]
      simpa only [Option.bind_some,Nat.cast_add,Nat.cast_one] using hx
    · rw [nextB.property]
      simpa only [Option.bind_some] using hy
  · have done : stop≤start := by omega
    have doneInt : (stop : Int)≤start := by exact_mod_cast done
    rw [stepForRangeIntAux.step_ge (by decide) doneInt,stepForLoopAux.step_ge done]
    exact ⟨a,b,rfl,rfl,related⟩
termination_by stop-start
decreasing_by omega



theorem source_code (p : ForwardLayout) (length : Nat) :
    (source p length).body = before p length ++
      [.forRangeInt "t" (.constInt 0) (.castNatToInt (.constNat length)) (.constNat 1) (body p)] := by
  unfold before body source Erasure.forward TypedSource.forward
  simp only [Bool.false_eq_true,if_false]
  unfold VeriTile.Bench.Repairs.DiagSsmTriton.forwardKernel
  simp only [
    ComputeKernel.eraseDType,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody,ComputeKernel.surfaceBody,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList.eq_def,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,List.take,List.drop,List.map_cons,List.map_nil,ComputeStmt.listToAlgorithm?_map_alg]
  try rfl

/-- Both actual runners succeed and agree on every observable component.
No equality of their different internal index registers is asserted. -/
theorem runs_related (p : ForwardLayout) (length : Nat) (u : BlockState) :
    ∃ a b, exec (source p length) u=some a ∧
      exec (RealReference.diag_ssm_forward_kernel p.s p.x p.l p.y length p.batch p.dim p.B).toAlgKernel u=some b ∧
      a.ObsAgree b := by
  let a0:=setup p length u
  let b0:=modelSetup p u
  have initial : Related p u a0.val b0.val := setup_related p length u
  obtain ⟨a,b,ha,hb,related⟩:=loop_related p u length 0 (Nat.zero_le _) a0.val b0.val initial
  simp only [Nat.cast_zero] at ha
  refine ⟨a,b,?_,?_,related.observations⟩
  · unfold exec
    rw [source_code,ProgramRewrite.append,a0.property]
    simp only [stepStmts,stepStmt,evalOp.eq_def,Tile.scalar,pure,bind,Option.bind_some]
    simp only [Int.ofNat_eq_natCast]
    rw [ha]
  · unfold exec
    rw [RealReference.diag_ssm_forward_kernel_toAlg_body]
    simp only [RealReference.diagSsmForwardProjectedBody,ProgramRewrite.append,b0.property,
      stepStmts,stepStmt,Option.bind_some,hb]


#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardTransport
