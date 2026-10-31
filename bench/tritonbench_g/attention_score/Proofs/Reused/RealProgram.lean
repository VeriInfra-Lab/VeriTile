import bench.tritonbench_g.attention_score.Proofs.Reused.RealBody

/-! Full real-erased source decomposition. The prelude keeps grouped-head
addressing, K load checks and scale initialization. The postlude retains the
source's unit inner output stride and NK mask. Both versions share these
fragments and differ only in the loop's scalar multiplication order. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealProgram
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Prelude projected from the sole repaired source. The irrelevant output,
window and query-tail arguments do not occur before its query loop. -/
def rawBefore (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) (evenN : Bool) :
    List Stmt :=
  (TypedSource.original Q K M "unused"
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk 0 0 0
    0 H H_KV N_CTX ROUND_CTX NKV_CTX BN BD BN 0 0 sm_scale
    Bool.false Bool.false Bool.false evenN rfl).toAlgKernel.body.take 16

/-- The output fragment is also extracted from the repaired source. -/
def rawAfter (Out : RegionName) (BN NKV_CTX stride_oz stride_oh : Nat) : List Stmt :=
  (TypedSource.original "q" "k" "m" Out
    0 0 0 0 0 0 0 0 stride_oz stride_oh 0 0 0 0 0 0 NKV_CTX BN 0 BN
    0 0 0 Bool.false Bool.false Bool.false Bool.false rfl).toAlgKernel.body.drop 17

def prelude (Q K M : RegionName) (scale : ℝ)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (evenN : Bool) : List Stmt :=
  Stmt.eraseDTypeList (rawBefore Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN)

def postlude (Out : RegionName) (B NK soz soh : Nat) : List Stmt :=
  Stmt.eraseDTypeList (rawAfter Out B NK soz soh)

private theorem erased_append (a b : List Stmt) :
    Stmt.eraseDTypeList (a++b) = Stmt.eraseDTypeList a ++ Stmt.eraseDTypeList b := by
  induction a with
  | nil => simp [Stmt.eraseDTypeList]
  | cons st rest ih => simp [Stmt.eraseDTypeList,ih]

def program (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) : List Stmt :=
  prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN ++
    [.forRangeInt "start_m" (.castNatToInt (.ref .nat [] "lo")) (.ref .int [] "hi") (.constNat B)
      (RealBody.body B D N offset size window complement evenM evenN candidate)] ++
    postlude Out B NK soz soh

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_source :
    (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body =
      program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
        offset size scale window complement evenM evenN Bool.false := by
  subst BM
  let actual := TypedSource.original Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BN BD BN offset size scale window complement evenM evenN rfl
  have split : actual.toAlgKernel.body =
      rawBefore Q K M scale H HK N RN NK BN BD sqz sqh sqm sqk skz skh skn skk evenN ++
      [.forRangeInt "start_m" (.castNatToInt (.ref .nat [] "lo")) (.ref .int [] "hi") (.constNat BN)
        ((RealArithmetic.loopBody? actual.toAlgKernel.body).getD [])] ++
      rawAfter Out BN NK soz soh := by
    simp only [actual,rawBefore,rawAfter,TypedSource.original,TypedSource.original,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
      List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero,
      RealArithmetic.loopBody?,Option.getD]
    rfl
  have loop := RealBody.original_loop Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BN BD BN offset size scale window complement evenM evenN rfl
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_,rfl⟩),
    RealArithmetic.loopBody_erased] at loop
  obtain ⟨raw,found,erased⟩ := Option.map_eq_some_iff.mp loop
  have gate : Stmt.eraseDTypeList ((RealArithmetic.loopBody? actual.toAlgKernel.body).getD []) =
      RealBody.body BN BD N offset size window complement evenM evenN Bool.false := by
    simpa only [actual,found,Option.getD_some] using erased
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_,rfl⟩)]
  change Stmt.eraseDTypeList actual.toAlgKernel.body = _
  rw [split,erased_append,erased_append]
  simp only [Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def]
  rw [gate]
  rfl

theorem candidate_source :
    (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body =
      program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
        offset size scale window complement evenM evenN Bool.true := by
  subst BM
  let actual := TypedSource.scalarLeft Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BN BD BN offset size scale window complement evenM evenN rfl
  have split : actual.toAlgKernel.body =
      rawBefore Q K M scale H HK N RN NK BN BD sqz sqh sqm sqk skz skh skn skk evenN ++
      [.forRangeInt "start_m" (.castNatToInt (.ref .nat [] "lo")) (.ref .int [] "hi") (.constNat BN)
        ((RealArithmetic.loopBody? actual.toAlgKernel.body).getD [])] ++
      rawAfter Out BN NK soz soh := by
    simp only [actual,rawBefore,rawAfter,TypedSource.scalarLeft,TypedSource.original,
      ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
      List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero,
      RealArithmetic.loopBody?,Option.getD]
    rfl
  have loop := RealBody.candidate_loop Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BN BD BN offset size scale window complement evenM evenN rfl
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_,rfl⟩),
    RealArithmetic.loopBody_erased] at loop
  obtain ⟨raw,found,erased⟩ := Option.map_eq_some_iff.mp loop
  have gate : Stmt.eraseDTypeList ((RealArithmetic.loopBody? actual.toAlgKernel.body).getD []) =
      RealBody.body BN BD N offset size window complement evenM evenN Bool.true := by
    simpa only [actual,found,Option.getD_some] using erased
  rw [SourceAlgorithm.erased_body _ (SourceAlgorithm.projection _ ⟨_,rfl⟩)]
  change Stmt.eraseDTypeList actual.toAlgKernel.body = _
  rw [split,erased_append,erased_append]
  simp only [Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def]
  rw [gate]
  rfl

end

/-- Every prelude statement writes registers only, including the checked and
unchecked K-load alternatives. -/
def writes : List RegName :=
  ["start_n","off_hz","off_z","off_h","off_hkv","q_offset","k_offset",
    "m_ptrs","o","Q_block_ptr","K_block_ptr","k","lo","hi","qk_scale"]

theorem prelude_code (Q K M : RegionName) (scale : ℝ)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (evenN : Bool) :
    RealFrame.Code writes (prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) := by
  simp only [prelude,rawBefore,TypedSource.original,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Except.bind,bind,List.take_succ_cons,List.take_zero,Stmt.eraseDTypeList,Stmt.eraseDType.eq_def]
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.nil
    | simp [writes]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealProgram
