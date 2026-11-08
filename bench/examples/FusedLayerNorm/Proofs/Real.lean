import bench.examples.FusedLayerNorm.Kernels
import bench.examples.Welford.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for FusedLayerNorm.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.FusedLayerNormCorrect
open VeriTile.Bench.Examples.FusedLayerNorm.Kernels
open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open WelfordCorrect (stepStmt_assign_inv welfordLoop_traceSafe welfordBody_pid_preserved)
open scoped VeriTile.Triton.KernelIO₃

set_option maxRecDepth 6000

section SourceDecomposition

/- Only the current shared source is executable. Mathematical views below are
its dtype erasure; these local reductions expose that view to execution proofs. -/
attribute [local simp] ComputeKernel.eraseDType ComputeKernel.toAlgorithm?
  ComputeKernel.toAlgKernel ComputeKernel.fromKernelBody ComputeStmt.listToAlgorithm?
  ComputeStmt.toAlgorithm? ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm?
  Kernel.eraseDType Stmt.eraseDTypeList Stmt.eraseDType Op.eraseDType.eq_def
  MemAccess.eraseDType.eq_def MaskOpt.eraseDType.eq_def
  VeriTile.Triton.eraseDType NumericDType.eraseDType ComparableDType.eraseDType
  IntegralDType.eraseDType ComputeDType.eraseDType FloatDType.toTileDType
attribute [local simp] twoPassLayerNormKernel fusedLayerNormKernel

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-- Executable real counterpart of the two-pass implementation. -/
def twoPassMathKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel :=
  (twoPassLayerNormKernel xReg γReg βReg yReg N rowStride ε).eraseDType

def fusedMathKernel
    (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) :
    ComputeKernel :=
  (fusedLayerNormKernel xReg γReg βReg yReg N rowStride ε).eraseDType

/-- Proof decomposition of the entire erased source. The equalities below
connect both lists to the current pasted Python bodies, including count casts. -/
def affineTail (x g b y : RegionName) (N stride : Nat) (eps : ℝ) : List Stmt :=
 [.assign .real [N] "γ" (.load .real (.region g (.arange N)) .none),
  .assign .real [N] "β" (.load .real (.region b (.arange N)) .none),
  .assign .real [] "σ_inv" (.div .real .nil (.const 1)
    (.sqrt (.add .real .nil (.ref .real [] "v") (.const eps)))),
  .assign .real [N] "y" (.add .real (.consSame .nil)
    (.mul .real (.consSame .nil)
      (.mul .real .scalarR (.sub .real .scalarR (.ref .real [N] "x") (.ref .real [] "μ"))
        (.ref .real [] "σ_inv")) (.ref .real [N] "γ")) (.ref .real [N] "β")),
  .store .real [N] (.region y (.ref .nat [N] "offs")) (.ref .real [N] "y") .none]

def twoPassStatements (x g b y : RegionName) (N stride : Nat) (eps : ℝ) : List Stmt :=
  (WelfordCorrect.twoPassStatements x y y N stride).take 9 ++ affineTail x g b y N stride eps

def fusedStatements (x g b y : RegionName) (N stride : Nat) (eps : ℝ) : List Stmt :=
  (WelfordCorrect.onlineStatements x y y N stride).take 4 ++
  [.assign .real [] "μ" (.ref .real [] "M"),
   .assign .real [] "count" (.natToReal (.full []
     (.mod .nat (IntegerExpr.sameShape []) (.constNat N) (.full [] (.constNat (2^64)))))),
   .assign .real [] "v" (.div .real .nil (.ref .real [] "S") (.ref .real [] "count")),
   .assign .real [] "σ_inv" (.div .real .nil (.const 1)
     (.sqrt (.add .real .nil (.ref .real [] "v") (.const eps)))),
   .assign .nat [N] "offs" (.add .nat .scalarL
     (.mul .nat .nil (.ref .nat [] "pid") (.constNat stride)) (.arange N)),
   .assign .real [N] "x" (.load .real (.region x (.ref .nat [N] "offs")) .none)] ++
  (affineTail x g b y N stride eps).take 2 ++ (affineTail x g b y N stride eps).drop 3

private theorem erased_algorithm (source : ComputeKernel) (alg : Kernel)
    (h : source.toAlgorithm? = .ok alg) : source.eraseDType.toAlgKernel = alg.eraseDType := by
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgKernel_fromKernelBody]

@[simp] theorem twoPass_algorithm (x g b y : RegionName) (N stride : Nat) (eps : ℝ) :
    (twoPassMathKernel x g b y N stride eps).toAlgKernel =
      {inputs := [x,g,b], outputs := [y], body := twoPassStatements x g b y N stride eps} := by
  have h : (twoPassLayerNormKernel x g b y N stride eps).toAlgorithm? =
      .ok (twoPassLayerNormKernel x g b y N stride eps).toAlgKernel := rfl
  rw [show (twoPassMathKernel x g b y N stride eps).toAlgKernel =
      (twoPassLayerNormKernel x g b y N stride eps).toAlgKernel.eraseDType from erased_algorithm _ _ h]
  simp [twoPassStatements,affineTail,WelfordCorrect.twoPassStatements]

@[simp] theorem fused_algorithm (x g b y : RegionName) (N stride : Nat) (eps : ℝ) :
    (fusedMathKernel x g b y N stride eps).toAlgKernel =
      {inputs := [x,g,b], outputs := [y], body := fusedStatements x g b y N stride eps} := by
  have h : (fusedLayerNormKernel x g b y N stride eps).toAlgorithm? =
      .ok (fusedLayerNormKernel x g b y N stride eps).toAlgKernel := rfl
  rw [show (fusedMathKernel x g b y N stride eps).toAlgKernel =
      (fusedLayerNormKernel x g b y N stride eps).toAlgKernel.eraseDType from erased_algorithm _ _ h]
  simp [fusedStatements,affineTail,WelfordCorrect.onlineStatements,onlineWelfordLoopBody]

end SourceDecomposition

private theorem project_erasure (ck : ComputeKernel) (alg : Kernel)
    (h : ck.toAlgorithm? = .ok alg) :
    ck.eraseDType.toAlgorithm? = .ok ck.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

theorem twoPass_projection (x g b y : RegionName) (N stride : Nat) (eps : ℝ) :
    (twoPassLayerNormKernel x g b y N stride eps).eraseDType.toAlgorithm? =
      .ok (twoPassMathKernel x g b y N stride eps).toAlgKernel :=
  project_erasure (twoPassLayerNormKernel x g b y N stride eps) _ rfl

theorem fused_projection (x g b y : RegionName) (N stride : Nat) (eps : ℝ) :
    (fusedLayerNormKernel x g b y N stride eps).eraseDType.toAlgorithm? =
      .ok (fusedMathKernel x g b y N stride eps).toAlgKernel :=
  project_erasure (fusedLayerNormKernel x g b y N stride eps) _ rfl

theorem twoPass_region_run (N stride : Nat) (hWord : N < 2^64) (ε : ℝ) (s : BlockState)
    (xs gs bs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs)
    (hg : ∀ i : Fin N, s.readMem "gamma" i.val = gs i)
    (hb : ∀ i : Fin N, s.readMem "beta" i.val = bs i) :
    ∃ t, exec (twoPassMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel s = some t ∧
      (∀ i : Fin N, t.readMem "y" (s.pid * stride + i.val) =
        (xs i - twoPassMean xs) / Real.sqrt (twoPassS xs / N + ε) * gs i + bs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin N, o ≠ s.pid * stride + i.val) →
        t.mem r o = s.mem r o) := by
  rw [twoPass_algorithm]
  have hmod : N % 18446744073709551616 = N := Nat.mod_eq_of_lt hWord
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  unfold InputRowLoadedAt at hx
  simp only [BlockState.pid_eq] at hx ⊢
  simp [twoPass_algorithm, twoPassStatements, affineTail, WelfordCorrect.twoPassStatements, exec, stepStmts, stepStmt, evalOp.eq_def,
    hmod, Tile.bop, Tile.uop, Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hx, hg, hb, twoPassMean, twoPassS, pow_two, div_eq_mul_inv]
    exact Or.inl rfl
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans rfl
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans rfl

theorem fused_region_run (N stride : Nat) (hWord : N < 2^64) (ε : ℝ) (s : BlockState)
    (xs gs bs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs)
    (hg : ∀ i : Fin N, s.readMem "gamma" i.val = gs i)
    (hb : ∀ i : Fin N, s.readMem "beta" i.val = bs i) :
    ∃ t, exec (fusedMathKernel "x" "gamma" "beta" "y" N stride ε).toAlgKernel s = some t ∧
      (∀ i : Fin N, t.readMem "y" (s.pid * stride + i.val) =
        (xs i - twoPassMean xs) / Real.sqrt (twoPassS xs / N + ε) * gs i + bs i) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin N, o ≠ s.pid * stride + i.val) →
        t.mem r o = s.mem r o) := by
  rw [fused_algorithm]
  have hmod : N % 18446744073709551616 = N := Nat.mod_eq_of_lt hWord
  obtain ⟨t, he, hM, hS, hp, _, hmem⟩ := WelfordCorrect.welford_loop_run N stride "x" s xs hx
  have hxt : ∀ i : Fin N, t.readMem "x" (s.pid * stride + i.val) = xs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hx i)
  have hgt : ∀ i : Fin N, t.readMem "gamma" i.val = gs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hg i)
  have hbt : ∀ i : Fin N, t.readMem "beta" i.val = bs i := by
    intro i
    exact (BlockState.readMem_congr (hmem _ _)).trans (hb i)
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * stride + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp only [BlockState.pid_eq] at hp hxt ⊢
  simp [fused_algorithm, fusedStatements, affineTail, WelfordCorrect.onlineStatements, exec, stepStmts, stepStmt, evalOp.eq_def,
    hmod, Tile.bop, Tile.uop, Tile.natToReal]
  have hzero : (0.0 : ℝ) = 0 := by norm_num
  simp only [hzero]
  simp only [WelfordCorrect.loopInitial] at he
  rw [he]
  simp [hM, hS, hp, NumericDType.add, NumericDType.mul,
    NumericDType.sub, NumericDType.div]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [BlockState.scatter_readback_nd _ _ _ hinj (i, PUnit.unit)]
    simp [hxt, hgt, hbt, div_eq_mul_inv]
  · rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans (hmem r o)
    · by_cases hr : r = "y"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun k _ => Ne.symm (ho k.1)) _).trans (hmem _ o)
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans (hmem r o)

section Safety

variable (xReg γReg βReg yReg : RegionName) (N rowStride : Nat) (ε : ℝ) (s : BlockState)

private theorem offs_activeAddressSafe (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (hread : t.regs .nat [N] "offs"
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * rowStride + N ≤ bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.ref .nat [N] "offs")) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.ref .nat [N] "offs") t
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg

private theorem arange_activeAddressSafe (bounds : RegionBounds) (t : BlockState)
    (active : TileIndex [N] → Prop)
    (reg : RegionName) (hreg : N ≤ bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.arange N)) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.arange N) t = some ⟨fun i => i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.vec]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le i.1.isLt hreg

set_option maxHeartbeats 1600000 in
theorem twoPass_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafe  bounds
      ((twoPassMathKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  rw [twoPass_algorithm]
  simp only [twoPassStatements, affineTail, WelfordCorrect.twoPassStatements, List.take, List.cons_append, List.nil_append]
  try simp only [twoPassMathKernel, fusedMathKernel, twoPassLayerNormKernel, fusedLayerNormKernel, ComputeKernel.seq, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType, ComputeDType.eraseDType, Except.bind, bind]
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp  (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N)) (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro sc hsc
  obtain ⟨vc, hvc, rfl⟩ := stepStmt_assign_inv hsc
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmt_assign_inv hs8
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ γReg hγb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmt_assign_inv hs9
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ βReg hβb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmt_assign_inv hs10
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmt_assign_inv hs11
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmt_assign_inv hs12
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  refine ⟨trivial, by simp [Op.SafeAt.eq_def], trivial,
    offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]

theorem twoPass_flattenOk :
    ((twoPassMathKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [twoPass_algorithm]
  simp [twoPassStatements, affineTail, WelfordCorrect.twoPassStatements, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

private theorem welfordLoop_pid_preserved (xReg : RegionName) (rowStride : Nat)
    :
    ∀ (fuel k n : Nat) (t t' : BlockState), n - k ≤ fuel →
      stepForLoopAux  "i" k n (onlineWelfordLoopBody xReg rowStride) t
        = some t' →
      t'.regs .nat [] "pid" = t.regs .nat [] "pid"
  | fuel, k, n, t, t', hf, h => by
      rw [stepForLoopAux] at h
      split at h
      next hlt =>
        cases fuel with
        | zero => omega
        | succ fuel =>
            cases hb : stepStmts  (onlineWelfordLoopBody xReg rowStride)
                (t.setReg "i" .nat [] (Tile.scalar k)) with
            | none =>
                rw [hb] at h
                exact absurd h (by simp)
            | some s1 =>
                rw [hb] at h
                replace h : stepForLoopAux  "i" (k + 1) n
                    (onlineWelfordLoopBody xReg rowStride) s1 = some t' := h
                rw [welfordLoop_pid_preserved xReg rowStride  fuel (k + 1) n
                    s1 t' (by omega) h,
                  welfordBody_pid_preserved xReg rowStride hb]
                simp [BlockState.setReg]
      next =>
        obtain rfl := Option.some_inj.mp h
        rfl

set_option maxHeartbeats 1600000 in
theorem fused_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + N ≤ bounds xReg)
    (hγb : N ≤ bounds γReg) (hβb : N ≤ bounds βReg)
    (hyb : s.pids 0 * rowStride + N ≤ bounds yReg) :
    Kernel.TraceSafe  bounds
      ((fusedMathKernel xReg γReg βReg yReg N rowStride ε).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  rw [fused_algorithm]
  simp only [fusedStatements, affineTail, WelfordCorrect.onlineStatements, List.take, List.drop, List.cons_append, List.nil_append]
  try simp only [twoPassMathKernel, fusedMathKernel, twoPassLayerNormKernel, fusedLayerNormKernel, ComputeKernel.seq, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType, ComputeDType.eraseDType, Except.bind, bind]
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe]
    exact welfordLoop_traceSafe xReg N rowStride  bounds (s.pids 0) hxb
      N 0 _ (by omega) (by simp [BlockState.setReg])
  intro s4 hs4
  have hpid4 : s4.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
    simp only [stepStmt] at hs4
    rw [welfordLoop_pid_preserved xReg rowStride  N 0 N _ s4 (by omega) hs4]
    simp [BlockState.setReg]
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro sc hsc
  obtain ⟨vc, hvc, rfl⟩ := stepStmt_assign_inv hsc
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmt_assign_inv hs8
  rw [show evalOp  (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange N))
      ((((s4.setReg "μ" .real [] v5).setReg "count" .real [] vc).setReg "v" .real [] v6).setReg
        "σ_inv" .real [] v7)
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg, hpid4]] at hv8
  obtain rfl := Option.some_inj.mp hv8
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s9 hs9
  obtain ⟨v9, hv9, rfl⟩ := stepStmt_assign_inv hs9
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ γReg hγb⟩
  intro s10 hs10
  obtain ⟨v10, hv10, rfl⟩ := stepStmt_assign_inv hs10
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, arange_activeAddressSafe N  bounds _ _ βReg hβb⟩
  intro s11 hs11
  obtain ⟨v11, hv11, rfl⟩ := stepStmt_assign_inv hs11
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s12 hs12
  obtain ⟨v12, hv12, rfl⟩ := stepStmt_assign_inv hs12
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  refine ⟨trivial, by simp [Op.SafeAt.eq_def], trivial,
    offs_activeAddressSafe N rowStride  bounds (s.pids 0) _ _ ?_
      (Region.cast yReg) (by simpa [Region.cast_self] using hyb)⟩
  simp [BlockState.setReg]

theorem fused_flattenOk :
    ((fusedMathKernel xReg γReg βReg yReg N rowStride ε
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [fused_algorithm]
  simp [fusedStatements, affineTail, WelfordCorrect.onlineStatements, onlineWelfordLoopBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

end Safety

end VeriTile.Bench.Examples.FusedLayerNormCorrect
