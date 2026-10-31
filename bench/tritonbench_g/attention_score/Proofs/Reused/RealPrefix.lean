import bench.tritonbench_g.attention_score.Proofs.Reused.RealProgram
import VeriTile.Triton.Semantics.UncheckedBlockLoad
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 400000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 16000

def beforeKey (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) : List Stmt :=
  let keyIndex : Op .nat [] := .castIntToNat
    (Op.intToSigned 32 (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat B))))
  [.assign .int [] "start_n" (.castNatToInt (.programId 0)),
   .assign .int [] "off_hz" (.castNatToInt (.programId 1)),
   .assign .int [] "off_z" (.tritonDiv .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat H))),
   .assign .int [] "off_h" (.tritonMod .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat H))),
   .assign .int [] "off_hkv" (.tritonDiv .int .nil (.ref .int [] "off_h")
     (.castNatToInt (.tritonDiv .nat .nil (.constNat H) (.constNat HK)))),
   .assign .int [] "q_offset" (.add .int .nil
     (.mul .int .nil (.ref .int [] "off_z") (.castNatToInt (.constNat sqz)))
     (.mul .int .nil (.ref .int [] "off_h") (.castNatToInt (.constNat sqh)))),
   .assign .int [] "k_offset" (.add .int .nil
     (.mul .int .nil (.ref .int [] "off_z") (.castNatToInt (.constNat skz)))
     (.mul .int .nil (.ref .int [] "off_hkv") (.castNatToInt (.constNat skh)))),
   .assign .ptr [B] "m_ptrs" (.ptrAdd .scalarL (.ptrBase M)
     (.castIntToNat (.add .int .scalarL
       (.mul .int .nil (.ref .int [] "off_hz") (.castNatToInt (.constNat RN)))
       (.castNatToInt (.arange B))))),
   .assign .real [B] "o" (.full [B] (.const 0)),
   .assign .blockPtr [B,D] "Q_block_ptr"
     (.makeBlockPtrDyn Q (.castIntToNat (.ref .int [] "q_offset")) [N,D] [B,D] [sqm,sqk] [0,0]),
   .assign .blockPtr [D,B] "K_block_ptr"
     (.makeBlockPtrDynOffsets K (.castIntToNat (.ref .int [] "k_offset")) [D,NK] [D,B] [skk,skn]
       [.constNat 0,keyIndex])]

theorem beforeKey_shape (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (scale : ℝ) (evenN : Bool) :
    (RealProgram.prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN).take 11 =
      beforeKey Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk := by
  simp only [RealProgram.prelude,RealProgram.rawBefore,TypedSource.original,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    List.take_succ_cons,List.take_zero,Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,IntegralDType.eraseDType,
    ComparableDType.eraseDType,eq_mpr_eq_cast,eq_mp_eq_cast,cast_eq,Region.cast]
  simp only [SourceAlgorithm.er_add_int,SourceAlgorithm.er_add_nat,SourceAlgorithm.er_mul_int,SourceAlgorithm.er_mul_nat,SourceAlgorithm.er_sub_int,SourceAlgorithm.er_floorDiv_int,SourceAlgorithm.er_floorDiv_nat,SourceAlgorithm.er_mod_int,SourceAlgorithm.er_mod_nat,SourceAlgorithm.er_lt_int,SourceAlgorithm.er_lt_nat,SourceAlgorithm.er_full_int,SourceAlgorithm.er_full_nat,SourceAlgorithm.er_full_real,SourceAlgorithm.er_const_int,SourceAlgorithm.er_const_nat,SourceAlgorithm.er_const_real,SourceAlgorithm.er_nat_to_int,SourceAlgorithm.er_int_to_nat,SourceAlgorithm.er_ref_int,SourceAlgorithm.er_ref_nat,SourceAlgorithm.er_program,SourceAlgorithm.er_arange,SourceAlgorithm.er_ptr_base,SourceAlgorithm.er_ptr_add,SourceAlgorithm.er_block,SourceAlgorithm.er_block_offsets,SourceAlgorithm.er_where_int,SourceAlgorithm.erase_signed32,Op.tritonDiv,Op.truncDiv,Op.tritonMod,Op.truncMod,List.map_cons,List.map_nil]
  simp only [beforeKey,Region.cast]
  congr 9 <;> try rfl
  simp only [Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,VeriTile.Triton.eraseDType]
def afterKey (B D N : Nat) (scale : ℝ) (evenN : Bool) : List Stmt :=
  [RealBody.loadKey B D evenN,
   .assign .nat [] "lo" (.constNat 0),
   .assign .int [] "hi" (.castNatToInt (.constNat N)),
   .assign .real [] "qk_scale" (.const scale),
   .assign .real [] "qk_scale" (.mul .real .nil (.ref .real [] "qk_scale") (.const 1.4426950408889634))]

theorem prelude_shape (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (scale : ℝ) (evenN : Bool) :
    RealProgram.prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN =
      beforeKey Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk ++ afterKey B D N scale evenN := by
  have split := (List.take_append_drop 11 (RealProgram.prelude Q K M scale H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk evenN)).symm
  rw [beforeKey_shape] at split
  rw [split]
  congr 1
  simp only [RealProgram.prelude,RealProgram.rawBefore,TypedSource.original,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Except.bind,bind,
    List.take_succ_cons,List.take_zero,Stmt.eraseDTypeList,List.drop_succ_cons,List.drop_zero]
  simp only [afterKey,RealBody.loadKey,Stmt.eraseDTypeList,Stmt.eraseDType.eq_def,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,ComputeDType.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Region.cast,eq_mpr_eq_cast,eq_mp_eq_cast,cast_eq]

theorem key_index_eval (s : BlockState) (start B : Nat)
    (hs : s.regs .int [] "start_n"=some (Tile.scalar (start:Int)))
    (keyRange : start*B<2147483648) :
    evalOp (.castIntToNat (Op.intToSigned 32 (.mul .int .nil (.ref .int [] "start_n") (.castNatToInt (.constNat B))))) s =
      some (Tile.scalar (start*B)) := by
  rw [evalOp,SourceAlgorithm.eval_signed32]
  simp only [evalOp.eq_def,hs,Option.bind,Tile.bop,Tile.scalar,NumericDType.mul,Broadcast.leftIndex,Broadcast.rightIndex]
  change some (⟨fun _ => (BitVec.ofInt 32 ((start:Int)*(B:Int))).toInt.toNat⟩ : Tile .nat []) = _
  rw [BitVec.toInt_ofInt_eq_self (by decide) (by norm_num;omega) (by norm_num;exact_mod_cast keyRange)]
  congr 1
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealPrefix
