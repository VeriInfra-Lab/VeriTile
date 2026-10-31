import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.StatementAudit

/-!
# Complete execution of the two int_scaled_matmul source programs

Source-audit target: matmul_kernel_with_block_pointers

Kernel 1 uses typed integer block pointers and `tl.load(...,
boundary_check=(0, 1))` with no padding option. These loads retain unspecified
padding. For an active output lane its exact integer result is

`sum_{k<K} A[row,k]*B[k,col] + sum_{K<=k<F} U_A(row,k)*U_B(k,col)`,

where `F = ceil(K/BLOCK_K)*BLOCK_K` and each U is the integer projection
(`Int.floor`) of the initial state's unspecified-value oracle at that actual
block-pointer address. The former zero-padding-only GEMM statement was false
for ragged K and arbitrary oracles; the current statement accounts for the
source's tail without restricting the oracle or requiring divisible K.
The loop invariant preserves both the input memory and the oracle.

Kernel 2 retains its explicit `other=0.0` masked loads and its existing scaled
GEMM theorem. Both EVEN_K arms remain modeled. Its real-valued store models
`(integer GEMM : Real)*s1[row]`; hardware float-to-int32 container truncation
is outside this model.

Translation-surface blocker: integer widths are erased; typed base regions
preserve the integer load channel. Tuple shapes/advance deltas use brackets,
and block-pointer order metadata is erased. Kernel 1 keeps the actual grouped
program-ID mapping, constexpr GROUP_M rebind, positive-step K loop, all pointer
advances, and boundary-checked stores. Kernel 2 retains its disclosed ascending
counter representation of the descending remaining-K loop, explicit numKBlocks
binder, and existing hCeil/hEven premises. Its broadcast shape attributes are
spelled with known literal dimensions; the cache hint is preserved then erased.
The passed scale strides are dead in the Python body and remain unused binders.

Correctness-surface blocker: the two public theorems give successful complete
execution and active output values via inline existentials. They do not claim
a memory-frame theorem, host-launch validity, finite-allocation safety,
fixed-width overflow behavior, or native GPU refinement.
-/

namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel 1: `matmul_kernel_with_block_pointers` — surface -/





/-! ## Kernel 2: `scaled_matmul_kernel_with_block_pointers` — surface -/

/-! ## Shared eval recipes (private copies — bench ports never import each other)

Scalar / tile `evalOp` reductions in the `int8_matmul_kernel` style, plus the
`MemCell`-level scatter machinery for the two output stores (`.int` for
kernel 1, `.real` for kernel 2). -/

section SharedRecipes

private theorem is_constInt_eval (n : Int) (t : BlockState) :
    evalOp (Op.constInt n) t = some (Tile.scalar n) := by
  simp [evalOp]

private theorem is_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

private theorem is_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

private theorem is_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

private theorem is_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

private theorem is_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

private theorem is_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

private theorem is_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem is_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem is_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem is_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

private theorem is_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_remap_eval {dtype : TileDType} {sh : TileShape}
    (outShape : TileShape) (map : TileIndex outShape → TileIndex sh)
    (x : Op dtype sh) (t : BlockState) (v : Tile dtype sh)
    (hv : evalOp x t = some v) :
    evalOp (Op.remap outShape map x) t = some (Tile.remap map v) := by
  simp only [evalOp, hv]
  rfl

private theorem is_intToReal_eval {sh : TileShape} (x : Op .int sh)
    (t : BlockState) (v : Tile .int sh) (hv : evalOp x t = some v) :
    evalOp (Op.intToReal x) t = some (Tile.intToReal v) := by
  simp only [evalOp, hv]
  rfl

/-- `tl.dot` on the `.int` channel at rank 2. `erw`, not `rw`: the operand
shapes are `[] ++ [M, K]`, which does not unfold at reducible transparency. -/
private theorem is_dotInt_eval {M K N : Nat} (x : Op .int [M, K])
    (y : Op .int [K, N]) (t : BlockState)
    (vx : Tile .int [M, K]) (vy : Tile .int [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dotInt (batch := []) x y) t = some (Tile.dotInt [] vx vy) := by
  erw [evalOp_dotInt, hx, hy]
  rfl

/-- A masked-with-`other` `.ptr` load, fully general. -/
private theorem is_load_ptr_maskOther {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (maskOp : Op .bool sh) (otherOp : Op dtype sh)
    (t : BlockState) (pt : Tile .ptr sh) (masks : Tile .bool sh)
    (others : Tile dtype sh)
    (hp : t.regs .ptr sh nm = some pt)
    (hm : evalOp maskOp t = some masks)
    (ho : evalOp otherOp t = some others) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm))
        (MaskOpt.maskOther maskOp otherOp)) t
      = some (⟨fun i => if masks.data i
            then t.readMemValue dtype (pt.data i).1 (pt.data i).2
            else others.data i⟩ : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho]
  rfl

/-- An unmasked `.ptr` load: every lane reads memory. -/
private theorem is_load_ptr_plain {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩
          : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

private theorem is_broadcastInt_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.constInt (0 : Int)).broadcast sh) t
      = some (⟨fun _ => (0 : ℤ)⟩ : Tile .int sh) := by
  simp only [evalOp]
  rfl

/-! ### `nat` scalar shapes -/

private theorem is_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem is_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem is_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem is_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [is_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [is_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [is_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [is_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [is_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem is_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem is_min_as_where (u v : Nat) :
    (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level. -/
private theorem is_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- `tl.cdiv` (`Op.div` over `(X + BX - 1)`). -/
private theorem is_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  is_divScalarNat_eval _ _ t (X + BX - 1) BX
    (is_subScalarNat_eval _ _ t (X + BX) 1
      (is_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

/-- `(X + BX - 1) // BX` (kernel 2's inline `grid_*` spelling — `Op.floorDiv`). -/
private theorem is_gridDiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  is_floorDivScalar_eval _ _ t (X + BX - 1) BX
    (is_subScalarNat_eval _ _ t (X + BX) 1
      (is_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

/-- `Stmt.ifThenElse` on a closed `Bool` guard picks the arm. -/
private theorem is_ifThenElse_step (cond : Bool) (tb eb : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThenElse (Op.constBool cond) tb eb) t
      = if cond then stepStmts tb t else stepStmts eb t := by
  unfold stepStmt
  have hc : evalOp (Op.constBool cond) t = some (Tile.scalar cond) := by
    simp [evalOp]
  rw [hc]
  cases cond <;> rfl

/-! ### `MemCell`-level scatter machinery -/

private theorem is_writeMemTyped_mem_other {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) (ρ : RegionName) (o : Nat)
    (h : ¬(rg = ρ ∧ a = o)) :
    (s.writeMemTyped dtype rg a v).mem ρ o = s.mem ρ o := by
  cases dtype <;>
    · show (if ρ = rg ∧ o = a then _ else s.mem ρ o) = s.mem ρ o
      rw [if_neg (fun hc => h ⟨hc.1.symm, hc.2.symm⟩)]

private theorem is_writeMemTyped_int_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : Int) :
    (s.writeMemTyped .int rg a v).mem rg a = MemCell.of .int v := by
  show (if rg = rg ∧ a = a then MemCell.of .int v else s.mem rg a)
    = MemCell.of .int v
  rw [if_pos ⟨rfl, rfl⟩]

private theorem is_writeMemTyped_real_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier .real) :
    (s.writeMemTyped .real rg a v).mem rg a
      = MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue v)) := by
  show (if rg = rg ∧ a = a then
      MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue v))
    else s.mem rg a) = _
  rw [if_pos ⟨rfl, rfl⟩]

/-- Cell-level frame for a masked scatter foldl: every cell missed by all
active writes is unchanged. -/
private theorem is_foldl_write_mem_preserve {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (P : α → Prop) [DecidablePred P] (ρ : RegionName) (o : Nat) (l : List α) :
    ∀ s : BlockState, (∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) →
      ((l.foldl (fun acc k =>
          if P k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k)
          else acc) s)).mem ρ o = s.mem ρ o := by
  induction l with
  | nil => intro s _; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, P k → ¬(region = ρ ∧ offsetFn k = o) :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      by_cases hP : P hd
      · rw [if_pos hP, ih _ htl,
          is_writeMemTyped_mem_other _ _ _ _ _ _ (h hd List.mem_cons_self hP)]
      · rw [if_neg hP]
        exact ih _ htl

/-- `.int` scatter readback with mask-restricted injectivity. -/
private theorem is_scatter_int_mem {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop)
    [DecidablePred P]
    (h_inj : ∀ k₁ k₂, P k₁ → P k₂ → offsetFn k₁ = offsetFn k₂ → k₁ = k₂)
    (i : TileIndex shape) (hPi : P i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k)
         else acc) s).mem region (offsetFn i)
      = MemCell.of .int (valueFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l2 : ∀ k ∈ l₂, P k → ¬(region = region ∧ offsetFn k = offsetFn i) := by
    intro k hk hPk hc
    have hki : k = i := h_inj k i hPk hPi hc.2
    subst hki
    exact hi_notin_l2 hk
  rw [is_foldl_write_mem_preserve (dtype := .int) region offsetFn valueFn P
    region (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact is_writeMemTyped_int_mem_self _ _ _ _

/-- `.real` scatter readback with mask-restricted injectivity (kernel 2's
flat store: the address map is injective only on the masked lanes). -/
private theorem is_scatter_real_mem {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier .real) (P : TileIndex shape → Prop)
    [DecidablePred P]
    (h_inj : ∀ k₁ k₂, P k₁ → P k₂ → offsetFn k₁ = offsetFn k₂ → k₁ = k₂)
    (i : TileIndex shape) (hPi : P i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k)
         else acc) s).mem region (offsetFn i)
      = MemCell.of .real
          (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i))) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l2 : ∀ k ∈ l₂, P k → ¬(region = region ∧ offsetFn k = offsetFn i) := by
    intro k hk hPk hc
    have hki : k = i := h_inj k i hPk hPi hc.2
    subst hki
    exact hi_notin_l2 hk
  rw [is_foldl_write_mem_preserve (dtype := .real) region offsetFn valueFn P
    region (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact is_writeMemTyped_real_mem_self _ _ _ _

end SharedRecipes

/-! ## Kernel 1 — the group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it — including the
constexpr-name rebind `GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)` — so
the spec is stated in the kernel's own coordinates. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_M)`. -/
def mmNumPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_N)`. -/
def mmNumPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_in_group = GROUP_M * num_pid_n`. -/
def mmNumPidInGroup (N BN GM : Nat) : Nat := GM * mmNumPidN N BN

/-- `group_id = pid // num_pid_in_group`. -/
def mmGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / mmNumPidInGroup N BN GM

/-- `first_pid_m = group_id * GROUP_M`. -/
def mmFirstPidM (s : BlockState) (N BN GM : Nat) : Nat :=
  mmGroupId s N BN GM * GM

/-- The rebound register `GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)`. -/
def mmGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (mmNumPidM M BM - mmFirstPidM s N BN GM) GM

/-- `pid_m = first_pid_m + (pid % GROUP_M)` — `GROUP_M` here is the rebound
register (the group size), not the parameter. -/
def mmPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  mmFirstPidM s N BN GM + s.pids 0 % mmGroupSize s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // GROUP_M` (the rebound register). -/
def mmPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % mmNumPidInGroup N BN GM / mmGroupSize s M N BM BN GM

/-! ## Kernel 1 — element accessors and the GEMM spec -/

/-- `A[row, kk]` — a signed `.int`-channel read of the launch state (the
host's `torch.int8` input; widths erased, kernel-1 disclosure (1)). -/
def mmAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)

/-- `B[kk, col]` — same on `B`. -/
def mmBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)

/-- The actual omitted-padding integer value, independently determined by
  the launch state's unspecified-value oracle at the computed address. -/
noncomputable def mmPad (s : BlockState) (r : RegionName) (addr : Nat) : Int :=
  Int.floor (s.undef r addr)

/-- Number of K lanes swept by the positive-step source loop. -/
def mmSpan (K BK : Nat) : Nat := ((K+BK-1)/BK)*BK

/-- One K-lane's ℤ contribution to output cell `(row, col)`, **as the
boundary-checked loads produce it**: each failed boundary check takes its
actual unspecified-value oracle factor. On active output cells only the
K guard can fail. -/
noncomputable def mmG (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn : Nat) (row col kk : Nat) : ℤ :=
  (if row < M ∧ kk < K then mmAElem s a_ptr sam sak row kk
      else mmPad s (Region.cast a_ptr) (row*sam+kk*sak))
    * (if kk < K ∧ col < N then mmBElem s b_ptr sbk sbn kk col
      else mmPad s (Region.cast b_ptr) (kk*sbk+col*sbn))

/-- The running accumulator after the loop counter has reached `c`: the sum
of all `c` swept K-lanes, including any unspecified padded tail. -/
noncomputable def mmAccVal (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn : Nat) (row col c : Nat) : ℤ :=
  ∑ kk ∈ Finset.range c, mmG s a_ptr b_ptr M N K sam sak sbk sbn row col kk

/-- The ordinary, unpadded GEMM term. The actual source additionally
contributes `mmPaddingTail` when K is ragged. -/
@[standalone_spec] def mmSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, mmAElem s a_ptr sam sak row kk.val
    * mmBElem s b_ptr sbk sbn kk.val col

/-- The padded K-tail contribution. It is a formula over the initial oracle,
not an assumption about the final output. -/
noncomputable def mmPaddingTail (s : BlockState) (a_ptr b_ptr : Region .int)
    (K F sam sak sbk sbn row col : Nat) : Int :=
  ∑ e ∈ Finset.range (F-K),
    mmPad s (Region.cast a_ptr) (row*sam+(K+e)*sak) *
      mmPad s (Region.cast b_ptr) ((K+e)*sbk+col*sbn)

/-- The actual source result is GEMM plus the unspecified padded tail. -/
@[standalone_spec] noncomputable def mmPaddedSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K F sam sak sbk sbn row col : Nat) : Int :=
  mmSpec s a_ptr b_ptr K sam sak sbk sbn row col +
    mmPaddingTail s a_ptr b_ptr K F sam sak sbk sbn row col

theorem mmAccVal_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col : Nat) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col 0 = 0 := by
  simp [mmAccVal]

/-- One loop iteration (counter `c`, one `BK`-wide slab of K-lanes) extends
the running sum. -/
theorem mmAccVal_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col c BK : Nat) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col c
      + ∑ e : Fin BK, mmG s a_ptr b_ptr M N K sam sak sbk sbn row col (c + e.val)
      = mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col (c + BK) := by
  unfold mmAccVal
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun e =>
    mmG s a_ptr b_ptr M N K sam sak sbk sbn row col (c + e)) BK]

/-- Split the actual accumulated result into the independent GEMM sum and
the unspecified padding contribution on `[K,F)`. -/
theorem mmAccVal_final (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col F : Nat)
    (hrow : row < M) (hcol : col < N) (hKF : K ≤ F) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col F
      = mmPaddedSpec s a_ptr b_ptr K F sam sak sbk sbn row col := by
  unfold mmAccVal mmPaddedSpec mmSpec mmPaddingTail
  rw [show F=K+(F-K) from by omega, Finset.sum_range_add]
  simp only [Nat.add_sub_cancel_left]
  congr 1
  · rw [Fin.sum_univ_eq_sum_range (fun kk =>
      mmAElem s a_ptr sam sak row kk * mmBElem s b_ptr sbk sbn kk col) K]
    apply Finset.sum_congr rfl
    intro kk hk
    simp [mmG,hrow,hcol,Finset.mem_range.mp hk]
  · apply Finset.sum_congr rfl
    intro e he
    simp [mmG,show ¬K+e<K from by omega]

/-- The `C` store address for output lane `idx` — the `c_block_ptr` address
(base offset `0`, offsets `(pid_m·BM, pid_n·BN)`). -/
def mmCAddr (scm scn BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  (pm * BM + idx.1.val) * scm + (pn * BN + idx.2.1.val) * scn

/-- `hInj` discharge for a row-major `C`: distinct output lanes get distinct
addresses when the column stride is positive and one block row fits inside
the row stride. -/
theorem mmCAddr_injective_rowMajor (scm scn BM BN pm pn : Nat)
    (hcn : 0 < scn) (hfit : BN * scn ≤ scm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [mmCAddr] at hij
  have hexp : ∀ r c : Nat, (pm * BM + r) * scm + (pn * BN + c) * scn
      = pm * BM * scm + r * scm + (pn * BN * scn + c * scn) := by
    intro r c
    ring
  rw [hexp, hexp] at hij
  have key : r₁.val * scm + c₁.val * scn = r₂.val * scm + c₂.val * scn := by
    omega
  have hrow : ∀ c : Fin BN, c.val * scn < scm := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hcn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * scm + scm ≤ r₂.val * scm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * scm + scm ≤ r₁.val * scm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * scn = c₂.val * scn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hcn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ## Kernel 1 — compiled body decomposition

Checked against the macro output by `rfl`. Lowerings worth naming:
`tl.cdiv` expands to `Op.div .nat` (while the `//` operator is
`Op.floorDiv`); `min(a, b)` is `Op.where (Op.lt …) a b`; the rebound
`GROUP_M` is an ordinary `.nat` register named `"GROUP_M"`; the two
`tl.make_block_ptr` with runtime offsets lower to
`Op.makeBlockPtrDynOffsets` with per-axis offset *ops*; the
`boundary_check=(0, 1)` loads are `.int`-typed `MemAccess.blockPtr … [0, 1]`
accesses with `MaskOpt.none` and `PaddingOption.unspecified`; `tl.advance` deltas are `Int`-cast; the store is the `.int`-typed
boundary-checked block-ptr store. -/









/-! ## Kernel 1 — block-pointer tiles and eval recipes -/

/-- The constant block-pointer tile every lane of a `tl.make_block_ptr`
result carries (base offset `0`). -/
def mmBPTile (R : RegionName) (pR pC sR sC BR BC o0 o1 : Nat) :
    Tile .blockPtr [BR, BC] :=
  ⟨fun _ => { region := R, baseOffset := 0, parentShape := [pR, pC],
              blockShape := [BR, BC], strides := [sR, sC],
              offsets := [o0, o1] }⟩

/-- `tl.make_block_ptr` with two dynamic offset ops. -/
private theorem mm_makeBP_eval {BR BC : Nat} (R : RegionName) (t : BlockState)
    (pR pC sR sC : Nat) (off0 off1 : Op .nat []) (v0 v1 : Nat)
    (h0 : evalOp off0 t = some (Tile.scalar v0))
    (h1 : evalOp off1 t = some (Tile.scalar v1)) :
    evalOp (Op.makeBlockPtrDynOffsets R (Op.constNat 0) [pR, pC]
        ([BR, BC] : TileShape) [sR, sC] [off0, off1]) t
      = some (mmBPTile R pR pC sR sC BR BC v0 v1) := by
  rw [makeBlockPtr2_eval]
  simp only [evalOp_constNat, h0, h1, List.mapM_cons, List.mapM_nil,
    Option.bind_some, Option.pure_def, Option.bind_eq_bind]
  rfl

/-- A boundary-checked `.int` block-ptr load: in-bounds lanes read the
`.int` channel at `(o0+i)·sR + (o1+j)·sC`; out-of-bounds lanes take
the oracle at that address, without reading memory. -/
private theorem mm_load_int_eval {BR BC : Nat} (nm : RegName) (t : BlockState)
    (R : RegionName) (pR pC sR sC o0 o1 : Nat)
    (hbp : t.regs .blockPtr [BR, BC] nm = some (mmBPTile R pR pC sR sC BR BC o0 o1)) :
    evalOp (Op.load .int (MemAccess.blockPtr (Op.ref .blockPtr [BR, BC] nm) [0, 1])
        MaskOpt.none .unspecified) t
      = some (⟨fun idx => if o0 + idx.1.val < pR ∧ o1 + idx.2.1.val < pC
            then t.readMemValue .int R
              ((o0 + idx.1.val) * sR + (o1 + idx.2.1.val) * sC)
            else mmPad t R ((o0+idx.1.val)*sR+(o1+idx.2.1.val)*sC)⟩ : Tile .int [BR, BC]) := by
  simp only [evalOp, evalOp_ref, hbp, Option.bind_some, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [mmBPTile, TileShape.blockPtr_inBounds_2d_offsets_index,
    TileShape.blockPtr_address_2d_offsets_index, Nat.zero_add,
    decide_eq_true_eq]
  rfl

/-- One `tl.advance` of a block-pointer register by `Nat`-cast deltas. -/
private theorem mm_advance_eval {BR BC : Nat} (nm : RegName) (t : BlockState)
    (R : RegionName) (pR pC sR sC o0 o1 d0 d1 : Nat)
    (hbp : t.regs .blockPtr [BR, BC] nm = some (mmBPTile R pR pC sR sC BR BC o0 o1)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BR, BC] nm)
        [((d0 : Nat) : Int), ((d1 : Nat) : Int)]) t
      = some (mmBPTile R pR pC sR sC BR BC (o0 + d0) (o1 + d1)) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp_ref, hbp, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [mmBPTile, BlockPtr.advance_2d_offsets]

/-! ## Kernel 1 — loaded tiles and the accumulator tile -/

/-- The loaded `a` tile, including the source's unspecified boundary lanes. -/
noncomputable def mmATile (s : BlockState) (a_ptr : Region .int)
    (M K sam sak BM BK pm c : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => if pm * BM + idx.1.val < M ∧ c + idx.2.1.val < K
      then mmAElem s a_ptr sam sak (pm * BM + idx.1.val) (c + idx.2.1.val)
      else mmPad s (Region.cast a_ptr)
        ((pm*BM+idx.1.val)*sam+(c+idx.2.1.val)*sak)⟩

/-- The loaded `b` tile at loop counter `c`. -/
noncomputable def mmBTile (s : BlockState) (b_ptr : Region .int)
    (K N sbk sbn BK BN pn c : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => if c + idx.1.val < K ∧ pn * BN + idx.2.1.val < N
      then mmBElem s b_ptr sbk sbn (c + idx.1.val) (pn * BN + idx.2.1.val)
      else mmPad s (Region.cast b_ptr)
        ((c+idx.1.val)*sbk+(pn*BN+idx.2.1.val)*sbn)⟩

/-- `accumulator` at loop counter `c`. -/
noncomputable def mmAccTile (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN pm pn c : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn
      (pm * BM + idx.1.val) (pn * BN + idx.2.1.val) c⟩

theorem mmAccTile_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN pm pn : Nat) :
    mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [mmAccTile, mmAccVal_zero]

/-- **The `Op.dotInt` accumulator step.** `accumulator += tl.dot(a, b)`
extends the running sum by one `BK`-slab: the per-lane product of the two
actual padded loads is exactly the guarded summand `mmG`. -/
theorem mmAccTile_dotInt_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn c : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn c)
        (Tile.dotInt [] (mmATile s a_ptr M K sam sak BM BK pm c)
          (mmBTile s b_ptr K N sbk sbn BK BN pn c))
      = mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn (c + BK) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    mmAccTile, NumericDType.int_add]
  erw [Tile.dotInt_nil_data]
  rw [← mmAccVal_step s a_ptr b_ptr M N K sam sak sbk sbn
    (pm * BM + r.val) (pn * BN + cc.val) c BK]
  -- the per-lane product of the two actual padded loads is *definitionally*
  -- the guarded summand `mmG`, so `congr 1` closes the sum.
  congr 1

/-! ## Kernel 1 — the loop invariant and walks -/

/-- The state carried across loop iterations (`c` is the raw counter — a
multiple of `BK`). -/
noncomputable def mmInv (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM : Nat) (c : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem
  ∧ s.undef = s0.undef
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (mmPidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (mmPidN s0 M N BM BN GM))
  ∧ s.regs .blockPtr [BM, BK] "a_block_ptr"
      = some (mmBPTile (Region.cast a_ptr) M K sam sak BM BK (mmPidM s0 M N BM BN GM * BM) c)
  ∧ s.regs .blockPtr [BK, BN] "b_block_ptr"
      = some (mmBPTile (Region.cast b_ptr) K N sbk sbn BK BN c (mmPidN s0 M N BM BN GM * BN))
  ∧ s.regs .int [BM, BN] "accumulator"
      = some (mmAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN
          (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) c)

/-- The loop combinator writes the counter register `"k"` (unused by the
body); `mmInv` constrains no register named `"k"`. -/
theorem mmInv_setReg_k (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM c v : Nat) (s : BlockState)
    (h : mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c s) :
    mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c
      (s.setReg "k" .nat [] (Tile.scalar v)) := by
  obtain ⟨hmem, hundef, hpids, hpm, hpn, hA, hB, hacc⟩ := h
  exact ⟨hmem, hundef, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hA, by simpa using hB, by simpa using hacc⟩





/-! ## Kernel 1 — the prologue walk -/

private theorem mm_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (mmNumPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (mmNumPidInGroup N BN GM)) := by
  rw [is_mulScalarNat_eval _ _ t GM (mmNumPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem mm_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmNumPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (mmGroupId s N BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0) (mmNumPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem mm_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (mmGroupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (mmFirstPidM s N BN GM)) := by
  rw [is_mulRef_eval t "group_id" (mmGroupId s N BN GM) GM hgid]
  rfl

/-- The `GROUP_M` register rebind: `min(num_pid_m - first_pid_m, GROUP_M)`. -/
private theorem mm_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (mmNumPidM M BM)))
    (hfp : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmFirstPidM s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (mmGroupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (mmNumPidM M BM - mmFirstPidM s N BN GM)) :=
    is_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (by rw [evalOp_ref]; exact hfp)
  rw [is_whereScalarNat_eval _ _ _ t
    (decide (mmNumPidM M BM - mmFirstPidM s N BN GM < GM))
    (mmNumPidM M BM - mmFirstPidM s N BN GM) GM
    (is_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, is_min_as_where, mmGroupSize]

private theorem mm_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfp : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmFirstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "GROUP_M"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "GROUP_M"))) t
      = some (Tile.scalar (mmPidM s M N BM BN GM)) := by
  rw [is_addScalarNat_eval _ _ t (mmFirstPidM s N BN GM)
    (s.pids 0 % mmGroupSize s M N BM BN GM)
    (by rw [evalOp_ref]; exact hfp)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem mm_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmNumPidInGroup N BN GM)))
    (hgs : t.regs .nat [] "GROUP_M"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "GROUP_M")) t
      = some (Tile.scalar (mmPidN s M N BM BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0 % mmNumPidInGroup N BN GM)
    (mmGroupSize s M N BM BN GM)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl



/-! ## Kernel 1 — the store tail -/

/-- The post-store state: the boundary-checked block-ptr store is a
bounds-guarded `.int` scatter over the output tile. -/
noncomputable def mmStoreState (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (f : TileIndex [BM, BN] → ℤ)
    (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .int c_ptr (mmCAddr scm scn BM BN pm pn i) (f i)
      else acc) t

/-- The boundary-checked `.int` block-ptr store steps to the named scatter
state. -/
private theorem mm_store_eq (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .int [BM, BN]) (f : TileIndex [BM, BN] → ℤ)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .blockPtr [BM, BN] "c_block_ptr"
      = some (mmBPTile c_ptr M N scm scn BM BN (pm * BM) (pn * BN)))
    (hv : t.regs .int [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .int [BM, BN]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BN] "c_block_ptr") [0, 1])
        (Op.ref .int [BM, BN] "c") MaskOpt.none) t
      = some (mmStoreState c_ptr M N scm scn BM BN pm pn f t) := by
  unfold stepStmt mmStoreState
  simp only [evalOp_ref, hv, hcp, Option.bind_some, Option.map_some, bind,
    Option.bind_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  simp only [mmBPTile, Bool.true_and,
    TileShape.blockPtr_inBounds_2d_offsets_index,
    TileShape.blockPtr_address_2d_offsets_index, Nat.zero_add]
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos (by simpa using hb), if_pos hb, hfv]
    rfl
  · rw [if_neg (by simpa using hb), if_neg hb]

/-- `MemCell`-level readback of the bounds-guarded `.int` scatter. -/
private theorem mm_store_props (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℤ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (mmStoreState c_ptr M N scm scn BM BN pm pn f t).mem c_ptr
          (mmCAddr scm scn BM BN pm pn i)
        = MemCell.of .int (f i) := by
  classical
  intro i hi
  unfold mmStoreState
  exact is_scatter_int_mem t
    (fun j : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] =>
      pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    (fun k₁ k₂ _ _ h => hInj h) i hi



/-! ## Kernel 1 — main theorem -/



/-! ## Kernel 2 — the group-swizzled block coordinates

The `triton.ops.matmul` swizzle (the `int8_dequant_matmul` twin): `grid_*`
are computed with the inline `(X + B - 1) // B` spelling (`Op.floorDiv`,
not `tl.cdiv`'s `Op.div`). -/

/-- `grid_m = (M + BLOCK_M - 1) // BLOCK_M`. -/
def smGridM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `grid_n = (N + BLOCK_N - 1) // BLOCK_N`. -/
def smGridN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `width = GROUP_M * grid_n`. -/
def smWidth (N BN GM : Nat) : Nat := GM * smGridN N BN

/-- `group_id = pid // width`. -/
def smGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / smWidth N BN GM

/-- `group_size = min(grid_m - group_id * GROUP_M, GROUP_M)`. -/
def smGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (smGridM M BM - smGroupId s N BN GM * GM) GM

/-- `pid_m = group_id * GROUP_M + (pid % group_size)`. -/
def smPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  smGroupId s N BN GM * GM + s.pids 0 % smGroupSize s M N BM BN GM

/-- `pid_n = (pid % width) // group_size`. -/
def smPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % smWidth N BN GM / smGroupSize s M N BM BN GM

/-! ## Kernel 2 — element accessors and the scaled-GEMM spec -/

/-- `A[row, kk]` — the `.int` channel of the launch state (`torch.int8`,
widths erased). -/
def smAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)

/-- `B[kk, col]`. -/
def smBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)

/-- `s1[row]` — the per-row fp32 scale, read at the **strideless** address
`row` (the `tl.broadcast_to(idx_m, …)` load; `stride_s1m`/`stride_s1n` are
dead parameters). -/
noncomputable def smS1 (s : BlockState) (s1_ptr : RegionName) (row : Nat) : ℝ :=
  s.readMem s1_ptr row

/-- One K-lane's ℤ contribution: in the masked (`EVEN_K = false`) arm both
loads are masked on the *same* `rk < k` remaining-count condition, so the
lane product is the single-guard `if kk < K` form; in the unmasked arm the
guard is always true (`hEven`). -/
def smG (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col kk : Nat) : ℤ :=
  if kk < K then smAElem s a_ptr sam sak row kk * smBElem s b_ptr sbk sbn kk col
  else 0

/-- The running accumulator after `c` swept K-lanes. -/
def smAccVal (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col c : Nat) : ℤ :=
  ∑ kk ∈ Finset.range c, smG s a_ptr b_ptr K sam sak sbk sbn row col kk

/-- **The kernel-2 integer accumulator value**: the genuine ℤ GEMM. -/
@[standalone_spec] def smSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, smAElem s a_ptr sam sak row kk.val
    * smBElem s b_ptr sbk sbn kk.val col

theorem smAccVal_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col : Nat) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col 0 = 0 := by
  simp [smAccVal]

theorem smAccVal_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col c BK : Nat) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col c
      + ∑ e : Fin BK, smG s a_ptr b_ptr K sam sak sbk sbn row col (c + e.val)
      = smAccVal s a_ptr b_ptr K sam sak sbk sbn row col (c + BK) := by
  unfold smAccVal
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun e =>
    smG s a_ptr b_ptr K sam sak sbk sbn row col (c + e)) BK]

/-- At any final counter `F ≥ K` the guarded running sum is the genuine
GEMM (lanes `kk ≥ K` contribute `0`). -/
theorem smAccVal_final (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col F : Nat) (hKF : K ≤ F) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col F
      = smSpec s a_ptr b_ptr K sam sak sbk sbn row col := by
  unfold smAccVal smSpec
  rw [Fin.sum_univ_eq_sum_range (fun kk =>
    smAElem s a_ptr sam sak row kk * smBElem s b_ptr sbk sbn kk col) K]
  have hsub : Finset.range K ⊆ Finset.range F := fun x hx =>
    Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hKF)
  rw [← Finset.sum_subset hsub
    (fun kk _ hkk => by
      unfold smG
      rw [if_neg (fun hc => hkk (Finset.mem_range.mpr hc))])]
  refine Finset.sum_congr rfl fun kk hkk => ?_
  unfold smG
  rw [if_pos (Finset.mem_range.mp hkk)]

/-- The flat inductor store address `col + N·row` (absolute, unwrapped
coordinates). -/
def smCAddr (N BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  pn * BN + idx.2.1.val + N * (pm * BM + idx.1.val)

/-- **Mask-restricted injectivity of the flat address** — proven outright
(no headline hypothesis): on lanes the store mask lets through, `col < N`,
so `col + N·row` is a base-`N` encoding. -/
theorem smCAddr_inj_active (M N BM BN pm pn : Nat) :
    ∀ i₁ i₂ : TileIndex [BM, BN],
      (pm * BM + i₁.1.val < M ∧ pn * BN + i₁.2.1.val < N) →
      (pm * BM + i₂.1.val < M ∧ pn * BN + i₂.2.1.val < N) →
      smCAddr N BM BN pm pn i₁ = smCAddr N BM BN pm pn i₂ → i₁ = i₂ := by
  rintro ⟨r₁, c₁, u₁⟩ ⟨r₂, c₂, u₂⟩ ⟨_, hc₁⟩ ⟨_, hc₂⟩ heq
  simp only [smCAddr] at heq hc₁ hc₂
  have hr : pm * BM + r₁.val = pm * BM + r₂.val := by
    rcases Nat.lt_trichotomy (pm * BM + r₁.val) (pm * BM + r₂.val) with h | h | h
    · have : N * (pm * BM + r₁.val) + N ≤ N * (pm * BM + r₂.val) := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left N h
      omega
    · exact h
    · have : N * (pm * BM + r₂.val) + N ≤ N * (pm * BM + r₁.val) := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left N h
      omega
  have hr' : r₁.val = r₂.val := by omega
  have hc' : c₁.val = c₂.val := by
    rw [hr] at heq
    omega
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr', Fin.ext hc', trivial⟩

/-! ## Kernel 2 — compiled body decomposition -/

/-! ## Kernel 2 — index vectors, pointer tiles and value tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)`. -/
def smOffs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The wrapped vector `(base + e) % Mm` — `ram` / `rbn`. -/
def smWrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `A` lane `(r, e)` at iteration `j`. -/
def smAAddr (M sam sak BM BK pm j : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * sam + idx.2.1.val * sak + j * (BK * sak)

/-- `B` lane `(e, c)` at iteration `j`. -/
def smBAddr (N sbk sbn BK BN pn j : Nat) (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val * sbk + (pn * BN + idx.2.1.val) % N * sbn + j * (BK * sbk)

noncomputable def smAPtrs (a_ptr : Region .int)
    (M sam sak BM BK pm j : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (Region.cast a_ptr, smAAddr M sam sak BM BK pm j idx)⟩

noncomputable def smBPtrs (b_ptr : Region .int)
    (N sbk sbn BK BN pn j : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast b_ptr, smBAddr N sbk sbn BK BN pn j idx)⟩

theorem smAPtrs_succ (a_ptr : Region .int) (M sam sak BM BK pm j : Nat) :
    Tile.ptrAdd Broadcast.scalarR (smAPtrs a_ptr M sam sak BM BK pm j)
        (Tile.scalar (BK * sak))
      = smAPtrs a_ptr M sam sak BM BK pm (j + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, smAPtrs, smAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

theorem smBPtrs_succ (b_ptr : Region .int) (N sbk sbn BK BN pn j : Nat) :
    Tile.ptrAdd Broadcast.scalarR (smBPtrs b_ptr N sbk sbn BK BN pn j)
        (Tile.scalar (BK * sbk))
      = smBPtrs b_ptr N sbk sbn BK BN pn (j + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, smBPtrs, smBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

theorem smAAddr_eq (M sam sak BM BK pm j : Nat) (idx : TileIndex [BM, BK]) :
    smAAddr M sam sak BM BK pm j idx
      = (pm * BM + idx.1.val) % M * sam + (j * BK + idx.2.1.val) * sak := by
  simp only [smAAddr]
  ring

theorem smBAddr_eq (N sbk sbn BK BN pn j : Nat) (idx : TileIndex [BK, BN]) :
    smBAddr N sbk sbn BK BN pn j idx
      = (j * BK + idx.1.val) * sbk + (pn * BN + idx.2.1.val) % N * sbn := by
  simp only [smBAddr]
  ring

/-- The loaded `a` tile at iteration `j` — the single-guard form both arms
land on. -/
def smATile (s : BlockState) (a_ptr : Region .int)
    (M K sam sak BM BK pm j : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => if j * BK + idx.2.1.val < K
      then smAElem s a_ptr sam sak ((pm * BM + idx.1.val) % M)
        (j * BK + idx.2.1.val)
      else 0⟩

/-- The loaded `b` tile at iteration `j`. -/
def smBTile (s : BlockState) (b_ptr : Region .int)
    (N K sbk sbn BK BN pn j : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => if j * BK + idx.1.val < K
      then smBElem s b_ptr sbk sbn (j * BK + idx.1.val)
        ((pn * BN + idx.2.1.val) % N)
      else 0⟩

/-- `acc` at iteration `j`, at the wrapped lane coordinates. -/
def smAccTile (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn j : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => smAccVal s a_ptr b_ptr K sam sak sbk sbn
      ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) (j * BK)⟩

theorem smAccTile_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn : Nat) :
    smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [smAccTile, smAccVal_zero]

/-- **The `Op.dotInt` accumulator step**: the per-lane product of the two
single-guard loads is the guarded summand `smG` (`(if c then x else 0) ·
(if c then y else 0) = if c then x·y else 0`). -/
theorem smAccTile_dotInt_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn j : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn j)
        (Tile.dotInt [] (smATile s a_ptr M K sam sak BM BK pm j)
          (smBTile s b_ptr N K sbk sbn BK BN pn j))
      = smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn (j + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    smAccTile, NumericDType.int_add]
  erw [Tile.dotInt_nil_data]
  rw [show (j + 1) * BK = j * BK + BK from by ring]
  rw [← smAccVal_step s a_ptr b_ptr K sam sak sbk sbn
    ((pm * BM + r.val) % M) ((pn * BN + cc.val) % N) (j * BK) BK]
  congr 1
  refine Finset.sum_congr rfl fun e _ => ?_
  simp only [smATile, smBTile, smG]
  by_cases h : j * BK + e.val < K
  · rw [if_pos h, if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg h, mul_zero]

/-! ## Kernel 2 — per-statement eval recipes -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem sm_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (smOffs (base * c) BD) := by
  rw [is_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (is_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- Bare `tl.arange(0, BK)` — `rk`. -/
private theorem sm_arange_eval (t : BlockState) (BD : Nat) :
    evalOp (Op.arange BD) t = some (smOffs 0 BD) := by
  rw [evalOp_arange]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smOffs, Tile.vec]

/-- `ram = rm % M` (behind the value-erased contiguity hints). -/
private theorem sm_wrap_eval (nm : RegName) (t : BlockState) (BD base Mm : Nat)
    (hr : t.regs .nat [BD] nm = some (smOffs base BD)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BD] nm)
        (Op.constNat Mm)) t
      = some (smWrapOffs base BD Mm) := by
  rw [is_mod_eval Broadcast.scalarR _ _ t (smOffs base BD) (Tile.scalar Mm)
    (by rw [evalOp_ref]; exact hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smWrapOffs, smOffs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `A = a_ptr + (ram[:, None] * sam + rk[None, :] * sak)` — iteration 0. -/
private theorem sm_aPtrsInit_eval (a_ptr : Region .int) (t : BlockState)
    (M sam sak BM BK pm : Nat)
    (hram : t.regs .nat [BM] "ram" = some (smWrapOffs (pm * BM) BM M))
    (hrk : t.regs .nat [BK] "rk" = some (smOffs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "ram"))
            (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sak)))) t
      = some (smAPtrs a_ptr M sam sak BM BK pm 0) := by
  rw [is_ptrAddBase_eval _ _ t _ _
    (is_addTile_eval NumericDType.nat _ _ _ t _ _
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sam)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hram))
        (evalOp_constNat _ _))
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sak)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smAPtrs, smAAddr, smWrapOffs, smOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `B = b_ptr + (rk[:, None] * sbk + rbn[None, :] * sbn)` — iteration 0. -/
private theorem sm_bPtrsInit_eval (b_ptr : Region .int) (t : BlockState)
    (N sbk sbn BK BN pn : Nat)
    (hrk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hrbn : t.regs .nat [BN] "rbn" = some (smWrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rbn"))
            (Op.constNat sbn)))) t
      = some (smBPtrs b_ptr N sbk sbn BK BN pn 0) := by
  rw [is_ptrAddBase_eval _ _ t _ _
    (is_addTile_eval NumericDType.nat _ _ _ t _ _
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sbk)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _))
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sbn)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smBPtrs, smBAddr, smWrapOffs, smOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- The masked arm's `a`-load mask: `rk[None, :] < k` at the remaining
count `kv`. -/
private theorem sm_aMask_eval (t : BlockState) (BM BK kv : Nat)
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar kv)) :
    evalOp (Op.remap [BM, BK]
        (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
          (Op.ref .nat [] "k"))) t
      = some (⟨fun idx => decide (idx.2.1.val < kv)⟩ : Tile .bool [BM, BK]) := by
  rw [is_remap_eval _ _ _ t _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (by rw [evalOp_ref]; exact hkk))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, smOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked arm's `b`-load mask: `rk[:, None] < k`. -/
private theorem sm_bMask_eval (t : BlockState) (BK BN kv : Nat)
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar kv)) :
    evalOp (Op.remap [BK, BN]
        (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
          (Op.ref .nat [] "k"))) t
      = some (⟨fun idx => decide (idx.1.val < kv)⟩ : Tile .bool [BK, BN]) := by
  rw [is_remap_eval _ _ _ t _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (by rw [evalOp_ref]; exact hkk))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, smOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The **unmasked** (`EVEN_K = true`) `a` load lands on the single-guard
tile: under `hcov` every lane is in K-range. -/
private theorem sm_aLoad_plain_eq (s0 : BlockState) (a_ptr : Region .int)
    (t : BlockState) (M K sam sak BM BK pm j : Nat)
    (hcov : (j + 1) * BK ≤ K)
    (hmem : t.mem = s0.mem)
    (hA : t.regs .ptr [BM, BK] "A" = some (smAPtrs a_ptr M sam sak BM BK pm j)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
        MaskOpt.none) t
      = some (smATile s0 a_ptr M K sam sak BM BK pm j) := by
  rw [is_load_ptr_plain "A" t _ hA]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [smATile, smAPtrs]
  rw [if_pos (show j * BK + e.val < K from by
    have : j * BK + e.val < (j + 1) * BK := by
      have := e.isLt
      calc j * BK + e.val < j * BK + BK := by omega
        _ = (j + 1) * BK := by ring
    omega)]
  rw [smAAddr_eq]
  simp only [smAElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-- The **masked** (`EVEN_K = false`) `a` load lands on the same
single-guard tile: `e < K - j·BK ↔ j·BK + e < K` (ℕ subtraction), and
masked-off lanes read the faithful `other = 0`. -/
private theorem sm_aLoad_masked_eq (s0 : BlockState) (a_ptr : Region .int)
    (t : BlockState) (M K sam sak BM BK pm j : Nat)
    (hmem : t.mem = s0.mem)
    (hA : t.regs .ptr [BM, BK] "A" = some (smAPtrs a_ptr M sam sak BM BK pm j))
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar (K - j * BK))) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
        (MaskOpt.maskOther
          (Op.remap [BM, BK]
            (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
              (Op.ref .nat [] "k")))
          ((Op.constInt 0).broadcast [BM, BK]))) t
      = some (smATile s0 a_ptr M K sam sak BM BK pm j) := by
  rw [is_load_ptr_maskOther "A" _ _ t _ _ _ hA
    (sm_aMask_eval t BM BK (K - j * BK) hk hkk)
    (is_broadcastInt_eval [BM, BK] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [smATile, smAPtrs, decide_eq_true_eq]
  by_cases h : j * BK + e.val < K
  · rw [if_pos (show e.val < K - j * BK from by omega), if_pos h, smAAddr_eq]
    simp only [smAElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  · rw [if_neg (show ¬ e.val < K - j * BK from by omega), if_neg h]

/-- Unmasked `b` load. -/
private theorem sm_bLoad_plain_eq (s0 : BlockState) (b_ptr : Region .int)
    (t : BlockState) (N K sbk sbn BK BN pn j : Nat)
    (hcov : (j + 1) * BK ≤ K)
    (hmem : t.mem = s0.mem)
    (hB : t.regs .ptr [BK, BN] "B" = some (smBPtrs b_ptr N sbk sbn BK BN pn j)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
        MaskOpt.none) t
      = some (smBTile s0 b_ptr N K sbk sbn BK BN pn j) := by
  rw [is_load_ptr_plain "B" t _ hB]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [smBTile, smBPtrs]
  rw [if_pos (show j * BK + e.val < K from by
    have : j * BK + e.val < (j + 1) * BK := by
      have := e.isLt
      calc j * BK + e.val < j * BK + BK := by omega
        _ = (j + 1) * BK := by ring
    omega)]
  rw [smBAddr_eq]
  simp only [smBElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-- Masked `b` load. -/
private theorem sm_bLoad_masked_eq (s0 : BlockState) (b_ptr : Region .int)
    (t : BlockState) (N K sbk sbn BK BN pn j : Nat)
    (hmem : t.mem = s0.mem)
    (hB : t.regs .ptr [BK, BN] "B" = some (smBPtrs b_ptr N sbk sbn BK BN pn j))
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar (K - j * BK))) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
        (MaskOpt.maskOther
          (Op.remap [BK, BN]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
              (Op.ref .nat [] "k")))
          ((Op.constInt 0).broadcast [BK, BN]))) t
      = some (smBTile s0 b_ptr N K sbk sbn BK BN pn j) := by
  rw [is_load_ptr_maskOther "B" _ _ t _ _ _ hB
    (sm_bMask_eval t BK BN (K - j * BK) hk hkk)
    (is_broadcastInt_eval [BK, BN] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [smBTile, smBPtrs, decide_eq_true_eq]
  by_cases h : j * BK + e.val < K
  · rw [if_pos (show e.val < K - j * BK from by omega), if_pos h, smBAddr_eq]
    simp only [smBElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  · rw [if_neg (show ¬ e.val < K - j * BK from by omega), if_neg h]

/-! ## Kernel 2 — the loop invariant and walks -/

/-- The state carried across iterations (`j` is the ascending counter of
the changed variable). -/
noncomputable def smInv (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks : Nat)
    (j : Nat) (s : BlockState) : Prop :=
  j ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (smPidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (smPidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "rk" = some (smOffs 0 BK)
  ∧ s.regs .ptr [BM, BK] "A"
      = some (smAPtrs a_ptr M sam sak BM BK (smPidM s0 M N BM BN GM) j)
  ∧ s.regs .ptr [BK, BN] "B"
      = some (smBPtrs b_ptr N sbk sbn BK BN (smPidN s0 M N BM BN GM) j)
  ∧ s.regs .int [BM, BN] "acc"
      = some (smAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK
          (smPidM s0 M N BM BN GM) (smPidN s0 M N BM BN GM) j)

/-- The loop combinator writes `"j"`; `smInv` constrains no register named
`"j"`. -/
theorem smInv_setReg_j (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks j v : Nat) (s : BlockState)
    (h : smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks j s) :
    smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks j
      (s.setReg "j" .nat [] (Tile.scalar v)) := by
  obtain ⟨hle, hmem, hpids, hpm, hpn, hrk, hA, hB, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hrk, by simpa using hA, by simpa using hB,
    by simpa using hacc⟩

/-! ## Kernel 2 — the prologue walk -/

private theorem sm_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "grid_n" = some (Tile.scalar (smGridN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "grid_n")) t
      = some (Tile.scalar (smWidth N BN GM)) := by
  rw [is_mulScalarNat_eval _ _ t GM (smGridN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem sm_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (smWidth N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "width")) t
      = some (Tile.scalar (smGroupId s N BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0) (smWidth N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem sm_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "grid_m" = some (Tile.scalar (smGridM M BM)))
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (smGroupId s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
              (Op.constNat GM)))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GM)))
        (Op.constNat GM)) t
      = some (Tile.scalar (smGroupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))) t
      = some (Tile.scalar (smGridM M BM - smGroupId s N BN GM * GM)) :=
    is_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (is_mulRef_eval t "group_id" (smGroupId s N BN GM) GM hgid)
  rw [is_whereScalarNat_eval _ _ _ t
    (decide (smGridM M BM - smGroupId s N BN GM * GM < GM))
    (smGridM M BM - smGroupId s N BN GM * GM) GM
    (is_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, is_min_as_where, smGroupSize]

private theorem sm_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (smGroupId s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (smGroupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size"))) t
      = some (Tile.scalar (smPidM s M N BM BN GM)) := by
  rw [is_addScalarNat_eval _ _ t (smGroupId s N BN GM * GM)
    (s.pids 0 % smGroupSize s M N BM BN GM)
    (is_mulRef_eval t "group_id" (smGroupId s N BN GM) GM hgid)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem sm_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (smWidth N BN GM)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (smGroupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")) t
      = some (Tile.scalar (smPidN s M N BM BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0 % smWidth N BN GM)
    (smGroupSize s M N BM BN GM)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-! ## Kernel 2 — the inductor suffix -/

/-- `idx_m = rm[:, None]`. -/
def smIdxMTile (BM pm : Nat) : Tile .nat [BM, 1] :=
  ⟨fun idx => pm * BM + idx.1.val⟩

/-- `idx_n = rn[None, :]`. -/
def smIdxNTile (BN pn : Nat) : Tile .nat [1, BN] :=
  ⟨fun idx => pn * BN + idx.2.1.val⟩

/-- `mask = (idx_m < M) & (idx_n < N)` — absolute (unwrapped) coordinates. -/
def smMaskTile (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M)
      && decide (pn * BN + idx.2.1.val < N)⟩

/-- `xindex = idx_n + (N * idx_m)` — the flat store index. -/
def smXIndexTile (N BM BN pm pn : Nat) : Tile .nat [BM, BN] :=
  ⟨fun idx => pn * BN + idx.2.1.val + N * (pm * BM + idx.1.val)⟩

private theorem sm_idxM_eval (t : BlockState) (BM pm : Nat)
    (hrm : t.regs .nat [BM] "rm" = some (smOffs (pm * BM) BM)) :
    evalOp (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "rm")) t
      = some (smIdxMTile BM pm) := by
  rw [is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrm)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smIdxMTile, smOffs, Tile.expandDim_data, TileShape.dropInsertedIndex]

private theorem sm_idxN_eval (t : BlockState) (BN pn : Nat)
    (hrn : t.regs .nat [BN] "rn" = some (smOffs (pn * BN) BN)) :
    evalOp (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rn")) t
      = some (smIdxNTile BN pn) := by
  rw [is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrn)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smIdxNTile, smOffs, Tile.expandDim_data, TileShape.dropInsertedIndex]

private theorem sm_mask_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hin : t.regs .nat [1, BN] "idx_n" = some (smIdxNTile BN pn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BM, 1] "idx_m") (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [1, BN] "idx_n") (Op.constNat N))) t
      = some (smMaskTile M N BM BN pm pn) := by
  rw [is_boolAnd_eval (Broadcast.consR (Broadcast.consL Broadcast.nil)) _ _ t _ _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar M) (by rw [evalOp_ref]; exact him) (evalOp_constNat _ _))
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar N) (by rw [evalOp_ref]; exact hin) (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smMaskTile, smIdxMTile, smIdxNTile, Tile.bop_data, Tile.cop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]

private theorem sm_xindex_eval (t : BlockState) (N BM BN pm pn : Nat)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hin : t.regs .nat [1, BN] "idx_n" = some (smIdxNTile BN pn)) :
    evalOp (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.ref .nat [1, BN] "idx_n")
        (Op.mul .nat Broadcast.scalarL (Op.constNat N)
          (Op.ref .nat [BM, 1] "idx_m"))) t
      = some (smXIndexTile N BM BN pm pn) := by
  rw [is_addTile_eval NumericDType.nat _ _ _ t _ _
    (by rw [evalOp_ref]; exact hin)
    (is_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
      (Tile.scalar N) _ (evalOp_constNat _ _)
      (by rw [evalOp_ref]; exact him))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smXIndexTile, smIdxMTile, smIdxNTile, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
    NumericDType.mul]

/-- The masked `.real` region load without `other`: active lanes read
memory, masked-off lanes read the (never-stored) `undef` value. -/
private theorem sm_load_region_masked_real {sh : TileShape}
    (rg : RegionName) (offOp : Op .nat sh) (maskOp : Op .bool sh)
    (t : BlockState) (offs : Tile .nat sh) (masks : Tile .bool sh)
    (ho : evalOp offOp t = some offs) (hm : evalOp maskOp t = some masks) :
    evalOp (Op.load .real (MemAccess.region rg offOp) (MaskOpt.mask maskOp)) t
      = some (⟨fun i => if masks.data i
            then t.readMemValue .real rg (offs.data i)
            else some (t.undef rg (offs.data i))⟩
          : Tile .real sh) := by
  simp only [evalOp, ho, hm]
  rfl

/-- The strideless `tl.broadcast_to(idx_m, …)` scale load: lane `(r, c)`
reads `s1[pid_m·BM + r]` when the mask lets it through. -/
private theorem sm_s1Load_eq (s0 : BlockState) (s1_ptr : RegionName)
    (t : BlockState) (M N BM BN pm pn : Nat)
    (hmem : t.mem = s0.mem)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hmk : t.regs .bool [BM, BN] "mask" = some (smMaskTile M N BM BN pm pn)) :
    evalOp (Op.load .real
        (MemAccess.region s1_ptr
          (Op.remap [BM, BN]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .nat [BM, 1] "idx_m")))
        (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))) t
      = some (⟨fun idx =>
            if pm * BM + idx.1.val < M ∧ pn * BN + idx.2.1.val < N
            then some (smS1 s0 s1_ptr (pm * BM + idx.1.val))
            else some (t.undef s1_ptr (pm * BM + idx.1.val))⟩
          : Tile .real [BM, BN]) := by
  have hread : ∀ o, t.readMem s1_ptr o = s0.readMem s1_ptr o := by
    intro o
    unfold BlockState.readMem
    rw [hmem]
  rw [sm_load_region_masked_real s1_ptr _ _ t
    (Tile.remap ((Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex)
      (smIdxMTile BM pm))
    (smMaskTile M N BM BN pm pn)
    (is_remap_eval _ _ _ t _ (by rw [evalOp_ref]; exact him)) (by
      rw [evalOp_ref]; exact hmk)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [smMaskTile, smIdxMTile, Tile.remap, Broadcast.leftIndex,
    Bool.and_eq_true, decide_eq_true_eq, BlockState.readMemValue_real, smS1]
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos hb, if_pos hb, hread]
  · rw [if_neg hb, if_neg hb]

/-! ## Kernel 2 — the masked flat store -/

/-- The post-store state: a mask-guarded `.real` scatter at the flat
addresses. -/
noncomputable def smStoreState (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (v : TileIndex [BM, BN] → TileCarrier .real)
    (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .real c_ptr (smCAddr N BM BN pm pn i) (v i)
      else acc) t

/-- The masked `.real` store of `acc * tmp0` steps to the named scatter
state (the value is consulted only on active lanes). -/
private theorem sm_store_eq (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (t : BlockState)
    (valueOp : Op .real [BM, BN]) (vt : Tile .real [BM, BN])
    (v : TileIndex [BM, BN] → TileCarrier .real)
    (hval : evalOp valueOp t = some vt)
    (hfv : ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) → vt.data i = v i)
    (hx : t.regs .nat [BM, BN] "xindex" = some (smXIndexTile N BM BN pm pn))
    (hmk : t.regs .bool [BM, BN] "mask" = some (smMaskTile M N BM BN pm pn)) :
    stepStmt (Stmt.store .real [BM, BN]
        (MemAccess.region c_ptr (Op.ref .nat [BM, BN] "xindex"))
        valueOp (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))) t
      = some (smStoreState c_ptr M N BM BN pm pn v t) := by
  unfold stepStmt smStoreState
  simp only [evalOp_ref, hval, hx, hmk, Option.bind_some, Option.map_some, bind]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos (show (smMaskTile M N BM BN pm pn).data (r, cc, u) = Bool.true from by
      simp only [smMaskTile, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hb.1, hb.2⟩)]
    rw [if_pos hb, hfv _ hb]
    rfl
  · rw [if_neg (show ¬((smMaskTile M N BM BN pm pn).data (r, cc, u) = Bool.true) from by
      simp only [smMaskTile, Bool.and_eq_true, decide_eq_true_eq]
      exact fun hc => hb hc)]
    rw [if_neg hb]

/-- `MemCell`-level readback of the masked `.real` scatter on every active
lane — the flat-address injectivity is the *proven*
`smCAddr_inj_active`. -/
private theorem sm_store_props (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (t : BlockState)
    (v : TileIndex [BM, BN] → TileCarrier .real) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (smStoreState c_ptr M N BM BN pm pn v t).mem c_ptr
          (smCAddr N BM BN pm pn i)
        = MemCell.of .real
            (FloatDType.real.ofReal (FloatDType.real.storeValue (v i))) := by
  classical
  intro i hi
  unfold smStoreState
  exact is_scatter_real_mem t
    (fun j : TileIndex [BM, BN] => smCAddr N BM BN pm pn j) v
    (fun j : TileIndex [BM, BN] =>
      pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    (fun k₁ k₂ h₁ h₂ h => smCAddr_inj_active M N BM BN pm pn k₁ k₂ h₁ h₂ h) i hi

/-! ## Kernel 2 — main theorem -/

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.IntScaledMatmul

#guard_msgs (drop info) in
#auditModuleAxioms
