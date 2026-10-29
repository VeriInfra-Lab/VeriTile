import VeriTile.Triton.Float.IntegerErasure
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealBody

/-! Full real-erased source decomposition. The prelude keeps grouped-head
addressing, K load checks and scale initialization. The postlude retains the
source's unit inner output stride and NK mask. Both versions share these
fragments and differ only in the loop's scalar multiplication order. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealProgram
open VeriTile.Triton
set_option maxRecDepth 64000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def prelude (Q K M : RegionName) (sm_scale : ℝ)
    (H H_KV N_CTX ROUND_CTX NKV_CTX BN BD
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk : Nat) (evenN : Bool) :
    List Stmt :=
  [Stmt.assign TileDType.int [] "start_n" (Op.castNatToInt (Op.programId 0)),
    Stmt.assign TileDType.int [] "off_hz" (Op.castNatToInt (Op.programId 1)),
    Stmt.assign TileDType.int [] "off_z"
      (Op.truncDiv Broadcast.nil (Op.ref TileDType.int [] "off_hz") (Op.constInt H)),
    Stmt.assign TileDType.int [] "off_h"
      (Op.truncMod Broadcast.nil (Op.ref TileDType.int [] "off_hz") (Op.constInt H)),
    Stmt.assign TileDType.int [] "off_hkv"
      (Op.truncDiv Broadcast.nil (Op.ref TileDType.int [] "off_h")
        (Op.truncDiv Broadcast.nil (Op.constInt H) (Op.constInt H_KV))),
    Stmt.assign TileDType.int [] "q_offset"
      (Op.add NumericDType.int Broadcast.nil
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_z") (Op.constInt stride_qz))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_h") (Op.constInt stride_qh))),
    Stmt.assign TileDType.int [] "k_offset"
      (Op.add NumericDType.int Broadcast.nil
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_z") (Op.constInt stride_kz))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_hkv") (Op.constInt stride_kh))),
    Stmt.assign TileDType.ptr [BN] "m_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase M)
        (Op.castIntToNat (Op.add NumericDType.int Broadcast.scalarL
          (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_hz") (Op.castNatToInt (Op.constNat ROUND_CTX)))
          (Op.castNatToInt (Op.arange BN))))),
    Stmt.assign TileDType.real [BN] "o" (Op.full [BN] (Op.const 0)),
    Stmt.assign TileDType.blockPtr [BN, BD] "Q_block_ptr"
      (Op.makeBlockPtrDyn Q (Op.castIntToNat (Op.ref TileDType.int [] "q_offset")) [N_CTX, BD] [BN, BD]
        [stride_qm, stride_qk] [0, 0]),
    Stmt.assign TileDType.blockPtr [BD, BN] "K_block_ptr"
      (Op.makeBlockPtrDynOffsets K (Op.castIntToNat (Op.ref TileDType.int [] "k_offset")) [BD, NKV_CTX] [BD, BN]
        [stride_kk, stride_kn]
        [Op.constNat 0,
          Op.castIntToNat (Op.intToSigned 32 (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "start_n") (Op.castNatToInt (Op.constNat BN))))]),
    RealBody.loadKey BN BD evenN,
    Stmt.assign TileDType.int [] "lo" (Op.full [] (Op.constInt 0)),
    Stmt.assign TileDType.int [] "hi" (Op.castNatToInt (Op.constNat N_CTX)),
    Stmt.assign TileDType.real [] "qk_scale" (Op.const sm_scale),
    Stmt.assign TileDType.real [] "qk_scale"
      (Op.mul NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "qk_scale")
        (Op.const 1.4426950408889634))]

def postlude (Out : RegionName) (BN NKV_CTX stride_oz stride_oh : Nat) : List Stmt :=
  [Stmt.assign TileDType.int [] "o_offset"
      (Op.add NumericDType.int Broadcast.nil
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_z") (Op.constInt stride_oz))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "off_h") (Op.constInt stride_oh))),
    Stmt.assign TileDType.int [BN] "o_range"
      (Op.add NumericDType.int Broadcast.scalarR (Op.castNatToInt (Op.arange BN))
        (Op.mul NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "start_n") (Op.constInt BN))),
    Stmt.assign TileDType.ptr [BN] "o_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.castIntToNat (Op.add NumericDType.int Broadcast.scalarL (Op.ref TileDType.int [] "o_offset")
          (Op.ref TileDType.int [BN] "o_range")))),
    Stmt.store TileDType.real [BN] (MemAccess.ptr (Op.ref TileDType.ptr [BN] "o_ptrs"))
      (Op.ref TileDType.real [BN] "o")
      (MaskOpt.mask
        (Op.lt ComparableDType.int Broadcast.scalarR (Op.ref TileDType.int [BN] "o_range")
          (Op.castNatToInt (Op.constNat NKV_CTX))))]

def program (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool) : List Stmt :=
  prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN ++
    [.forRangeInt "start_m" (.ref .int [] "lo") (.ref .int [] "hi") (.constNat B)
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
  simp only [TypedSource.original, Whole002.TypedSource.widened,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind]
  rw [ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [Kernel.eraseDType, Stmt.eraseDTypeList,
    program, prelude, postlude, RealBody.body, RealBody.loadPrefix,
    RealBody.suffix, RealBody.loadQuery, RealBody.loadKey, RealArithmetic.body,
    Bool.false_eq_true, ↓reduceIte, List.cons_append, List.nil_append]
  repeat' first | apply congrArg₂ List.cons | rfl
  all_goals as_aux_lemma =>
    simp only [Stmt.eraseDType, Stmt.eraseDTypeList, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, ComputeDType.eraseDType, List.map_cons, List.map_nil,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType,
      Op.eraseDType_truncDiv, Op.eraseDType_truncMod, Op.eraseDType_intToSigned]
    try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, ComputeDType.eraseDType,
      FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, List.map_cons, List.map_nil,
      Op.eraseDType_castFloat, Op.eraseDType_ref, Op.eraseDType_truncDiv,
      Op.eraseDType_truncMod, Op.eraseDType_intToSigned]
    try rfl

theorem candidate_source :
    (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).eraseDType.toAlgKernel.body =
      program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
        offset size scale window complement evenM evenN Bool.true := by
  subst BM
  simp only [TypedSource.scalarLeft, Whole002.TypedSource.widenedScalarLeft,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind]
  rw [ComputeKernel.toAlgKernel_fromKernelBody]
  simp only [Kernel.eraseDType, Stmt.eraseDTypeList,
    program, prelude, postlude, RealBody.body, RealBody.loadPrefix,
    RealBody.suffix, RealBody.loadQuery, RealBody.loadKey, RealArithmetic.body,
    Bool.false_eq_true, ↓reduceIte, List.cons_append, List.nil_append]
  repeat' first | apply congrArg₂ List.cons | rfl
  all_goals as_aux_lemma =>
    simp only [Stmt.eraseDType, Stmt.eraseDTypeList, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, ComputeDType.eraseDType, List.map_cons, List.map_nil,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType,
      Op.eraseDType_truncDiv, Op.eraseDType_truncMod, Op.eraseDType_intToSigned]
    try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, ComputeDType.eraseDType,
      FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, List.map_cons, List.map_nil,
      Op.eraseDType_castFloat, Op.eraseDType_ref, Op.eraseDType_truncDiv,
      Op.eraseDType_truncMod, Op.eraseDType_intToSigned]
    try rfl


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
  unfold prelude RealBody.loadKey
  repeat' first
    | apply RealFrame.Code.assign
    | apply RealFrame.Code.ifThenElse
    | apply RealFrame.Code.nil
    | simp [writes]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealProgram
