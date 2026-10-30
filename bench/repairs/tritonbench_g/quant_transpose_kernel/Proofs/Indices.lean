import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteNativeExamples
import bench.repairs.tritonbench_g.quant_transpose_kernel.Contract

namespace VeriTile.Bench.Repairs.QuantTransposeKernel
open QuantTransposeFinite
set_option maxHeartbeats 4000000

def Small (x : NativeInteger) (n : Nat) : Prop := x.value = (n : Int) ∧ x.kind ≠ .i64

theorem wrap_small (k : IntegerKind) (hk : k ≠ .i64) (n : Nat) (hn : n < 2^31) :
    Small (integerWrap k n) n := by
  cases k with
  | constant => exact ⟨rfl, by simp [integerWrap]⟩
  | i64 => contradiction
  | i32 =>
    refine ⟨?_, by simp [integerWrap]⟩
    exact BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem metadata_small (n : Nat) (hn : n < 2^31) : Small (metadataInteger n) n := by
  by_cases h : n = 1
  · subst n; exact ⟨rfl, by decide⟩
  · simp only [metadataInteger, h, if_false, hn, if_true]
    exact wrap_small .i32 (by decide) n hn

theorem literal_small (n : Nat) : Small (integerLiteral n) n := ⟨rfl, by simp [integerLiteral]⟩

theorem promote_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n) :
    promoteInteger a.kind b.kind ≠ .i64 := by
  rcases a with ⟨ak,av⟩; rcases b with ⟨bk,bv⟩
  cases ak <;> cases bk <;> simp_all [Small, promoteInteger]

theorem add_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (h : m+n < 2^31) : Small (integerAdd a b) (m+n) := by
  change Small (integerWrap (promoteInteger a.kind b.kind) (a.value+b.value)) (m+n)
  rw [ha.1, hb.1, ← Nat.cast_add]
  exact wrap_small _ (promote_small a b m n ha hb) _ h

theorem mul_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (h : m*n < 2^31) : Small (integerMul a b) (m*n) := by
  change Small (integerWrap (promoteInteger a.kind b.kind) (a.value*b.value)) (m*n)
  rw [ha.1, hb.1, ← Nat.cast_mul]
  exact wrap_small _ (promote_small a b m n ha hb) _ h

theorem sub_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (hle : n ≤ m) (h : m-n < 2^31) : Small (integerSub a b) (m-n) := by
  change Small (integerWrap (promoteInteger a.kind b.kind) (a.value-b.value)) (m-n)
  rw [ha.1, hb.1, ← Nat.cast_sub hle]
  exact wrap_small _ (promote_small a b m n ha hb) _ h

theorem min_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (h : min m n < 2^31) : Small (integerMin a b) (min m n) := by
  unfold integerMin integerBinary
  rw [ha.1, hb.1, ← Nat.cast_min]
  exact wrap_small _ (promote_small a b m n ha hb) _ h

theorem div_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (hn : 0 < n) (h : m/n < 2^31) :
    ∃ c, integerDiv a b = some c ∧ Small c (m/n) := by
  refine ⟨integerWrap (promoteInteger a.kind b.kind) (m/n), ?_,
    wrap_small _ (promote_small a b m n ha hb) _ h⟩
  have hd : ¬divisionUndefined a b := by simp [divisionUndefined, ha.1, hb.1]; omega
  simp only [integerDiv, hd, if_false, ha.1, hb.1]
  split_ifs <;> congr 2 <;> simp [truncDiv, Int.natCast_ediv, Int.fdiv_eq_ediv]

theorem rem_small (a b : NativeInteger) (m n : Nat) (ha : Small a m) (hb : Small b n)
    (hn : 0 < n) (h : m%n < 2^31) :
    ∃ c, integerRem a b = some c ∧ Small c (m%n) := by
  refine ⟨integerWrap (promoteInteger a.kind b.kind) (m%n), ?_,
    wrap_small _ (promote_small a b m n ha hb) _ h⟩
  have hd : ¬divisionUndefined a b := by simp [divisionUndefined, ha.1, hb.1]; omega
  simp only [integerRem, hd, if_false, ha.1, hb.1]
  split_ifs <;> congr 2 <;> simp [truncDiv, Int.natCast_emod, Int.fmod_eq_emod]
  simp only [show ¬((m : Int) < 0) from by omega, show ¬((n : Int) < 0) from by omega, iff_self, if_true]
  have hmod := Int.emod_add_ediv_mul (m : Int) (n : Int)
  nlinarith




theorem tile_small (a : Input) (h : IndexDomain a) (pid : Nat)
    (hp : pid < gridM a.M*gridN a.N) :
    ∃ t, nativeTile a.M a.N pid = some t ∧
      Small t.row (tileRow (gridM a.M) (gridN a.N) pid) ∧
      Small t.col (tileCol (gridM a.M) (gridN a.N) pid) := by
  let gm := gridM a.M
  let gn := gridN a.N
  have hgm : 0 < gm := by have hp := h.rows; dsimp [gm,gridM]; omega
  have hgn : 0 < gn := by have hp := h.columns; dsimp [gn,gridN]; omega
  have hmup : a.M+128 < 2^31 := h.rowsUpper
  have hnup : a.N+128 < 2^31 := h.columnsUpper
  have hgmu : gm < 2^31 := by dsimp [gm,gridM]; omega
  have hgnu : gn < 2^31 := by dsimp [gn,gridN]; omega
  have hpid : pid < 2^31 := lt_trans hp h.grid
  have hw : 0 < 8*gn := by omega
  have hstart : pid/(8*gn)*8 < gm := by
    have hd := Nat.div_add_mod' pid (8*gn)
    change pid < gm*gn at hp
    by_contra hn
    have hc : gm ≤ pid/(8*gn)*8 := by omega
    have hm := Nat.mul_le_mul_right gn hc
    have hh : gm*gn ≤ pid := calc
      gm*gn ≤ pid/(8*gn)*8*gn := hm
      _ = pid/(8*gn)*(8*gn) := by ring
      _ ≤ pid := Nat.div_mul_le_self _ _
    omega
  have hs : 0 < min (gm-pid/(8*gn)*8) 8 := by omega
  have bounds := tile_bounds gm gn pid hgn hp
  have hm := metadata_small a.M (by omega)
  have hn := metadata_small a.N (by omega)
  have hma := add_small _ _ _ _ hm (literal_small 128) hmup
  have hna := add_small _ _ _ _ hn (literal_small 128) hnup
  have hms := sub_small _ _ _ _ hma (literal_small 1) (by omega) (by omega)
  have hns := sub_small _ _ _ _ hna (literal_small 1) (by omega) (by omega)
  have hM : a.M+128-1 = a.M+127 := by omega
  have hN : a.N+128-1 = a.N+127 := by omega
  rw [hM] at hms
  rw [hN] at hns
  obtain ⟨mg,hmg,smg⟩ := div_small _ _ _ _ hms (literal_small 128) (by decide) hgmu
  obtain ⟨ng,hng,sng⟩ := div_small _ _ _ _ hns (literal_small 128) (by decide) hgnu
  change Small mg gm at smg
  change Small ng gn at sng
  have sw := mul_small _ _ _ _ (literal_small 8) sng h.width
  have sp := wrap_small .i32 (by decide) pid hpid
  obtain ⟨gr,hgr,sgr⟩ := div_small _ _ _ _ sp sw hw (lt_of_le_of_lt (Nat.div_le_self pid _) hpid)
  have ss := mul_small _ _ _ _ sgr (literal_small 8) (lt_trans hstart hgmu)
  have sd := sub_small _ _ _ _ smg ss (by omega) (by omega)
  have sz := min_small _ _ _ _ sd (literal_small 8) (by omega)
  obtain ⟨rw,hrw,srw⟩ := rem_small _ _ _ _ sp sz hs (lt_of_le_of_lt (Nat.mod_le pid _) hpid)
  obtain ⟨cw,hcw,scw⟩ := rem_small _ _ _ _ sp sw hw (lt_of_le_of_lt (Nat.mod_le pid _) hpid)
  obtain ⟨co,hco,sco⟩ := div_small _ _ _ _ scw sz hs (by
    have hc := bounds.2.2
    change pid%(8*gn)/min (gm-pid/(8*gn)*8) 8 < gn at hc
    omega)
  have sr := add_small _ _ _ _ ss srw (by
    have hr := bounds.2.1
    change pid/(8*gn)*8+pid%min (gm-pid/(8*gn)*8) 8 < gm at hr
    omega)
  refine ⟨⟨integerAdd (integerMul gr (integerLiteral 8)) rw, co⟩, ?_, sr, sco⟩
  norm_num only [Nat.cast_ofNat] at hmg hng hgr hrw hcw hco
  simp only [nativeTile, hmg, hng, hgr, hrw, hcw, hco, bind, Bind.bind, pure, Pure.pure, Option.bind_some]



theorem lane_small (a : Input) (h : IndexDomain a) (pid lane : Nat)
    (hp : pid < gridM a.M*gridN a.N) (hl : lane < 128*128) :
    ∃ native, nativeLane a.M a.N a.sam a.san pid lane = some native ∧
      native.mask = decide (row a.M a.N pid lane < a.M ∧ col a.M a.N pid lane < a.N) ∧
      Small native.inputOffset (row a.M a.N pid lane*a.sam+col a.M a.N pid lane*a.san) ∧
      Small native.outputOffset (row a.M a.N pid lane+col a.M a.N pid lane*a.M) := by
  obtain ⟨t,ht,sr,sc⟩ := tile_small a h pid hp
  have hgn : 0 < gridN a.N := by have hn := h.columns; unfold gridN; omega
  have bounds := tile_bounds (gridM a.M) (gridN a.N) pid hgn hp
  have hli : lane/128 < 128 := by omega
  have hlj : lane%128 < 128 := Nat.mod_lt _ (by decide)
  have hr : row a.M a.N pid lane < gridM a.M*128 := by unfold row; omega
  have hc : col a.M a.N pid lane < gridN a.N*128 := by unfold col; omega
  have hru : gridM a.M*128 < 2^31 := by have hu := h.rowsUpper; unfold gridM; omega
  have hcu : gridN a.N*128 < 2^31 := by have hu := h.columnsUpper; unfold gridN; omega
  have srm := mul_small _ _ _ _ sr (literal_small 128) (by omega)
  have scn := mul_small _ _ _ _ sc (literal_small 128) (by omega)
  have sli := wrap_small .i32 (by decide) (lane/128) (by omega)
  have slj := wrap_small .i32 (by decide) (lane%128) (by omega)
  have rm := add_small _ _ _ _ srm sli (lt_trans hr hru)
  have rn := add_small _ _ _ _ scn slj (lt_trans hc hcu)
  change Small _ (row a.M a.N pid lane) at rm
  change Small _ (col a.M a.N pid lane) at rn
  have hi1 := Nat.mul_le_mul_right a.sam (Nat.le_of_lt hr)
  have hi2 := Nat.mul_le_mul_right a.san (Nat.le_of_lt hc)
  have hin := h.inputOffset
  have hin1 : row a.M a.N pid lane*a.sam < 2^31 := by omega
  have hin2 : col a.M a.N pid lane*a.san < 2^31 := by omega
  have hins : row a.M a.N pid lane*a.sam+col a.M a.N pid lane*a.san < 2^31 := by omega
  have si1 := mul_small _ _ _ _ rm (metadata_small a.sam h.strideM) hin1
  have si2 := mul_small _ _ _ _ rn (metadata_small a.san h.strideN) hin2
  have sin := add_small _ _ _ _ si1 si2 hins
  have ho1 := Nat.mul_le_mul_right a.M (Nat.le_of_lt hc)
  have hout := h.outputOffset
  have hout2 : col a.M a.N pid lane*a.M < 2^31 := by omega
  have houts : row a.M a.N pid lane+col a.M a.N pid lane*a.M < 2^31 := by omega
  have so1 := mul_small _ _ _ _ rm (metadata_small 1 (by decide)) (by simpa using lt_trans hr hru)
  have so2 := mul_small _ _ _ _ rn (metadata_small a.M (by have hh := h.rowsUpper; omega)) hout2
  have sout := add_small _ _ _ _ so1 so2 (by simpa using houts)
  let R := integerAdd (integerMul t.row (integerLiteral 128)) (integerWrap .i32 (lane/128))
  let C := integerAdd (integerMul t.col (integerLiteral 128)) (integerWrap .i32 (lane%128))
  refine ⟨⟨R,C,integerLess R (metadataInteger a.M) && integerLess C (metadataInteger a.N),
    integerAdd (integerMul R (metadataInteger a.sam)) (integerMul C (metadataInteger a.san)),
    integerAdd (integerMul R (metadataInteger 1)) (integerMul C (metadataInteger a.M))⟩,
    ?_, ?_, sin, ?_⟩
  · simp [nativeLane, ht, R, C]
  · change (integerLess R (metadataInteger a.M) && integerLess C (metadataInteger a.N)) = _
    have hm := metadata_small a.M (by have hu := h.rowsUpper; omega)
    have hn := metadata_small a.N (by have hu := h.columnsUpper; omega)
    change Small R _ at rm
    change Small C _ at rn
    simp [integerLess, rm.1, rn.1, hm.1, hn.1]
  · simpa only [Nat.mul_one] using sout

theorem pointer_exact (base width n : Nat) (v : NativeInteger) (hv : Small v n)
    (hb : base+n*width < 2^64) :
    (nativePointer base width v).toNat = base+n*width := by
  unfold nativePointer
  rw [hv.1]
  simp only [BitVec.ofInt_natCast]
  rw [← BitVec.ofNat_mul, ← BitVec.ofNat_add]
  exact Nat.mod_eq_of_lt hb

/-- The extra condition of the accepted native-to-logical theorem is now a
consequence of checked finite-index envelopes and ordinary allocation bounds.
It is not an assumed numerical leaf or a premise on the desired result. -/
theorem arithmetic_refines (a : Input) (h : IndexDomain a) (d : Storage)
    (alloc : Allocations) (m : Memory) (x : String)
    (logical : List (Word d)) (xs : Nat → Word d) (p : PhysicalLayout)
    (D : Domain d a alloc m x logical xs) (P : PhysicalBinding d a alloc m x p) :
    ArithmeticRefines a d p := by
  intro pid lane hp hl
  obtain ⟨v,hv,hm,hi,ho⟩ := lane_small a h pid lane hp hl
  refine ⟨v,hv,hm,?_⟩
  intro activeMask
  have activeLogical : row a.M a.N pid lane<a.M ∧ col a.M a.N pid lane<a.N := by
    rw [hm] at activeMask
    exact of_decide_eq_true activeMask
  have activeSource : active (parameters a alloc x) pid lane := activeLogical
  have ptr := quant_transpose_physical_lane_accesses d a alloc m x logical xs p D P pid lane 0
    activeSource (bytes_positive d)
  constructor
  · exact pointer_exact _ _ _ _ hi (by simpa [laneInput, parameters] using ptr.2.1)
  · simpa only [Nat.mul_one] using
      pointer_exact p.output.base 1 _ _ ho (by simpa [offset, parameters, Nat.mul_one, Nat.add_comm] using ptr.2.2.2.1)

#axiomsClean arithmetic_refines
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.QuantTransposeKernel
