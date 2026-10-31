import bench.tritonbench_g.attention_score.Sources.Parameters
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.SignedCast

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.SourceAlgorithm
open VeriTile.Triton
set_option maxRecDepth 16000

theorem erased_body (source : ComputeKernel)
    (projected : source.toAlgorithm? = .ok source.toAlgKernel) :
    source.eraseDType.toAlgKernel.body = Stmt.eraseDTypeList source.toAlgKernel.body := by
  simp only [ComputeKernel.eraseDType,projected]
  simp [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?_fromKernelBody,Kernel.eraseDType]

theorem projection (source : ComputeKernel) (h : ∃ alg, source.toAlgorithm? = .ok alg) :
    source.toAlgorithm? = .ok source.toAlgKernel := by
  obtain ⟨alg,ha⟩ := h
  simp [ComputeKernel.toAlgKernel,ha]

theorem narrow_value (x : Int) :
    (if 2*((x%4294967296).toNat%4294967296)<4294967296 then
      max (x%4294967296) 0%4294967296 else
      max (x%4294967296) 0%4294967296-4294967296) = (BitVec.ofInt 32 x).toInt := by
  have h := signedBits_int_mod 32 x
  simpa only [IntegerExpr.signedBits,Int.natCast_emod,Int.toNat_of_nonneg
    (Int.emod_nonneg x (by decide : (4294967296:Int)≠0)),Int.toNat_eq_max] using h

theorem narrow_nat_exact (n : Nat) (hn : n<2147483648) :
    (if 2*((((n:Int)%4294967296).toNat)%4294967296)<4294967296 then
      max ((n:Int)%4294967296) 0%4294967296 else
      max ((n:Int)%4294967296) 0%4294967296-4294967296).toNat = n := by
  rw [narrow_value]
  rw [BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)]
  rfl

theorem er_mul_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.mul .int bc x y).eraseDType = Op.mul .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_sub_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.sub .int bc x y).eraseDType = Op.sub .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_mod_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.mod .int bc x y).eraseDType = Op.mod .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
 theorem er_full_int (shape : TileShape) (x : Op .int []) :
    (Op.full shape x).eraseDType = Op.full shape x.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_const_int (n : Int) :
    (Op.constInt n).eraseDType = Op.constInt n := by
  rw [Op.eraseDType.eq_def]
theorem er_mul_nat (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    (Op.mul .nat bc x y).eraseDType = Op.mul .nat bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_mod_nat (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    (Op.mod .nat bc x y).eraseDType = Op.mod .nat bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
 theorem er_full_nat (shape : TileShape) (x : Op .nat []) :
    (Op.full shape x).eraseDType = Op.full shape x.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_const_nat (n : Nat) :
    (Op.constNat n).eraseDType = Op.constNat n := by
  rw [Op.eraseDType.eq_def]
theorem er_nat_to_int (x : Op .nat shape) :
    (Op.castNatToInt x).eraseDType = Op.castNatToInt x.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_int_to_nat (x : Op .int shape) :
    (Op.castIntToNat x).eraseDType = Op.castIntToNat x.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_lt_nat (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    (Op.lt .nat bc x y).eraseDType = Op.lt .nat bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
 theorem er_where_int (c : Op .bool shape) (x y : Op .int shape) :
    (Op.where c x y).eraseDType = Op.where c.eraseDType x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem erase_signed (width : Nat) (x : Op .int shape) :
    (Op.intToSigned width x).eraseDType = Op.intToSigned width x.eraseDType := by
  simp only [Op.intToSigned,Op.unsignedToInt,Op.intToUnsigned,Op.natToUnsigned,
    er_where_int,er_lt_nat,er_mul_nat,er_full_nat,er_full_int,er_const_nat,er_const_int,
    er_sub_int,er_mod_nat,er_mod_int,er_nat_to_int,er_int_to_nat]
  rfl

theorem erase_signed32 (x : Op .int shape) :
    (Op.intToSigned 32 x).eraseDType = Op.intToSigned 32 x.eraseDType :=
  erase_signed 32 x

theorem er_add_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.add .int bc x y).eraseDType = Op.add .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_floorDiv_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.floorDiv .int bc x y).eraseDType = Op.floorDiv .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_lt_int (bc : Broadcast a b out) (x : Op .int a) (y : Op .int b) :
    (Op.lt .int bc x y).eraseDType = Op.lt .int bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_ref_int (name : RegName) :
    (Op.ref .int shape name).eraseDType = Op.ref .int shape name := by
  rw [Op.eraseDType.eq_def]
  cases shape <;> rfl
theorem er_add_nat (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    (Op.add .nat bc x y).eraseDType = Op.add .nat bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_floorDiv_nat (bc : Broadcast a b out) (x : Op .nat a) (y : Op .nat b) :
    (Op.floorDiv .nat bc x y).eraseDType = Op.floorDiv .nat bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
  rfl
theorem er_ref_nat (name : RegName) :
    (Op.ref .nat shape name).eraseDType = Op.ref .nat shape name := by
  rw [Op.eraseDType.eq_def]
  cases shape <;> rfl
theorem er_program (coord : Nat) : (Op.programId coord).eraseDType = Op.programId coord := by
  rw [Op.eraseDType.eq_def]
 theorem er_arange (n : Nat) : (Op.arange n).eraseDType = Op.arange n := by
  rw [Op.eraseDType.eq_def]
 theorem er_full_real (shape : TileShape) (x : Op .real []) :
    (Op.full shape x).eraseDType = Op.full shape x.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_const_real (x : ℝ) : (Op.const x).eraseDType = Op.const x := by
  rw [Op.eraseDType.eq_def]
 theorem er_ptr_base (r : Region d) :
    (Op.ptrBase r).eraseDType = Op.ptrBase (r.cast : Region (VeriTile.Triton.eraseDType d)) := by
  rw [Op.eraseDType.eq_def]
 theorem er_ptr_add (bc : Broadcast a b out) (x : Op .ptr a) (y : Op .nat b) :
    (Op.ptrAdd bc x y).eraseDType = Op.ptrAdd bc x.eraseDType y.eraseDType := by
  rw [Op.eraseDType.eq_def]
 theorem er_block (r : RegionName) (base : Op .nat []) (parent block strides offsets : List Nat) :
    (Op.makeBlockPtrDyn r base parent block strides offsets).eraseDType =
      Op.makeBlockPtrDyn r base.eraseDType parent block strides offsets := by
  rw [Op.eraseDType.eq_def]
 theorem er_block_offsets (r : RegionName) (base : Op .nat []) (parent block strides : List Nat)
    (offsets : List (Op .nat [])) :
    (Op.makeBlockPtrDynOffsets r base parent block strides offsets).eraseDType =
      Op.makeBlockPtrDynOffsets r base.eraseDType parent block strides (offsets.map Op.eraseDType) := by
  rw [Op.eraseDType.eq_def]

theorem eval_signed32 (x : Op .int shape) (s : BlockState) :
    evalOp (Op.intToSigned 32 x) s = (do
      let t ← evalOp x s
      pure ⟨fun i => (BitVec.ofInt 32 (t.data i)).toInt⟩) :=
  eval_intToSigned 32 x s

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)

theorem original_projection :
    (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).toAlgorithm? =
    .ok (TypedSource.original Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).toAlgKernel := by
  rfl

theorem candidate_projection :
    (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).toAlgorithm? =
    .ok (TypedSource.scalarLeft Q K M Out
      sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
      offset size scale window complement evenM evenN hBlock).toAlgKernel := by
  rfl
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.SourceAlgorithm
