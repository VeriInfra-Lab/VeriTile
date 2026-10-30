import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.CopySource
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling.CopyExecution
open VeriTile.Triton FP.Structural CopySource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def safe (width i : Nat) : Nat := if i < width then i else 0

def input (key : Bool) : RegionName := if key then "key" else "value"
def output (key : Bool) : RegionName := if key then "keyCache" else "valueCache"
def width (p : Parameters) (key : Bool) : Nat := if key then p.dk else p.dv
def capacity (p : Parameters) (key : Bool) : Nat := if key then p.ck else p.cv
def lanes (p : Parameters) (key : Bool) : Nat := if key then p.bk else p.bv

def value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src : Nat) (key : Bool) (i : TileIndex [lanes p key]) : α :=
  M.fp32Load (if i.1.val < width p key then
    (s.mem (input key) (src * width p key + safe (capacity p key) i.1.val)).read .real
    else M.literal (some .fp32) .real 0.0)

def prepared {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  let z := s.setReg "job" .int [] (fun _ => s.pids 0)
  let z := z.setReg "src" .int [] (fun _ => src)
  let z := z.setReg "dst" .int [] (fun _ => dst)
  let z := z.setReg "ik" .int [p.bk] (fun i => i.1.val)
  let z := z.setReg "iv" .int [p.bv] (fun i => i.1.val)
  let z := z.setReg "safe_ik" .int [p.bk] (fun i => safe p.ck i.1.val)
  let z := z.setReg "safe_iv" .int [p.bv] (fun i => safe p.cv i.1.val)
  let z := z.setReg "k" .real [p.bk] (value M p s src Bool.true)
  z.setReg "v" .real [p.bv] (value M p s src Bool.false)

def scatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (key : Bool) (t : State α) : State α :=
  (TileShape.allIndices [lanes p key]).foldl (fun z i =>
    if i.1.val < width p key then
      z.write (output key) (dst * width p key + safe (capacity p key) i.1.val)
        (.mk .real (value M p s src key i)) else z) t

def result {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) : State α :=
  scatter M p s src dst Bool.false
    (scatter M p s src dst Bool.true (prepared M p s src dst))

theorem index_toNat (a b n i : Nat) :
    Int.toNat ((a : Int) * b + if i < n then (i : Int) else 0) =
      a * b + if i < n then i else 0 := by
  by_cases h : i < n <;>
    simp only [h, ↓reduceIte, Int.add_zero, Nat.add_zero,
      ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

/-- Actual successful execution of each policy-0 source version. Schedule
reads are explicit input-only premises; no success or output is assumed. -/
theorem execute {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int)) (v : Bool) :
    exec M (kernel v p) s = some (result M p s src dst) := by
  cases v <;>
    simp [exec, kernel, originalKernel, commutedKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, store, numeric, numericLt, bop,
      ComputeDType.eraseDType, Region.cast, Option.bind, hs, hd,
      State.masked_scatter_eq_filter, prepared, result, scatter, value,
      lanes, width, capacity, input, output, safe,
      Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
  all_goals simp [index_toNat, value, lanes, width, capacity, input, output, safe]
  all_goals congr 20

def rowCopy {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (key : Bool) (t : State α) : State α :=
  (TileShape.allIndices [lanes p key]).foldl (fun z i =>
    if i.1.val < width p key then
      z.write (output key) (dst * width p key + i.1.val)
        (.mk .real (M.fp32Load
          ((s.mem (input key) (src * width p key + i.1.val)).read .real)))
    else z) t

theorem scatter_eq {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (key : Bool) (t : State α)
    (shape : capacity p key = width p key) :
    scatter M p s src dst key t = rowCopy M p s src dst key t := by
  unfold scatter rowCopy
  congr 1
  funext z i
  by_cases hi : i.1.val < width p key <;> simp [hi, value, safe, shape]

theorem rowCopy_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (key : Bool) (t : State α)
    (r : RegionName) (o : Nat)
    (h : r ≠ output key ∨ o < dst * width p key ∨ dst * width p key + width p key ≤ o) :
    (rowCopy M p s src dst key t).mem r o = t.mem r o := by
  unfold rowCopy
  apply State.masked_scatter_frame
  rcases h with hr | hlo | hhi
  · exact Or.inl hr
  · right; intro i _ hi; omega
  · right; intro i _ hi; omega

theorem rowCopy_read {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (key : Bool) (t : State α)
    (i : Nat) (hi : i < width p key) (covered : width p key ≤ lanes p key) :
    (rowCopy M p s src dst key t).mem (output key) (dst * width p key + i) =
      .mk .real (M.fp32Load ((s.mem (input key) (src * width p key + i)).read .real)) := by
  let ix : TileIndex [lanes p key] := (⟨i, by omega⟩, PUnit.unit)
  unfold rowCopy
  apply State.masked_scatter_readback (i := ix)
    (active := fun j : TileIndex [lanes p key] => j.1.val < width p key)
  · exact hi
  · intro a b he
    have h := Nat.add_left_cancel he
    cases a with | mk a au =>
      cases b with | mk b bu =>
        have hab : a = b := Fin.ext h
        cases hab
        rfl

/-- Complete K/V row values for the executed policy-0 branch. -/
theorem result_read {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (shape : ∀ key, capacity p key = width p key)
    (covered : ∀ key, width p key ≤ lanes p key)
    (key : Bool) (i : Nat) (hi : i < width p key) :
    (result M p s src dst).mem (output key) (dst * width p key + i) =
      .mk .real (M.fp32Load ((s.mem (input key) (src * width p key + i)).read .real)) := by
  unfold result
  rw [scatter_eq M p s src dst Bool.false _ (shape Bool.false),
    scatter_eq M p s src dst Bool.true _ (shape Bool.true)]
  cases key
  · exact rowCopy_read M p s src dst Bool.false _ i hi (covered Bool.false)
  ·
    rw [rowCopy_frame M p s src dst Bool.false _ _ _ (Or.inl (by decide))]
    exact rowCopy_read M p s src dst Bool.true _ i hi (covered Bool.true)

/-- Full logical-memory frame: all regions and every unwritten cache cell. -/
theorem result_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat) (shape : ∀ key, capacity p key = width p key)
    (r : RegionName) (o : Nat)
    (outside : ∀ key, r ≠ output key ∨ o < dst * width p key ∨ dst * width p key + width p key ≤ o) :
    (result M p s src dst).mem r o = s.mem r o := by
  unfold result
  rw [scatter_eq M p s src dst Bool.false _ (shape Bool.false),
    scatter_eq M p s src dst Bool.true _ (shape Bool.true),
    rowCopy_frame M p s src dst Bool.false _ r o (outside Bool.false),
    rowCopy_frame M p s src dst Bool.true _ r o (outside Bool.true)]
  rfl

/-- Two successful executions with the same full state. Floating arithmetic
is unchanged, so no numerical rewrite assumption is needed. -/
theorem same_result {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (src dst : Nat)
    (hs : (s.mem "src" (s.pids 0)).read .int = (src : Int))
    (hd : (s.mem "dst" (s.pids 0)).read .int = (dst : Int)) :
    ∃ final, exec M (originalKernel p) s = some final ∧
      exec M (commutedKernel p) s = some final :=
  ⟨result M p s src dst, execute M p s src dst hs hd Bool.false,
    execute M p s src dst hs hd Bool.true⟩

#axiomsClean execute
#axiomsClean result_read
#axiomsClean result_frame
#axiomsClean same_result
end
end VeriTile.Bench.Repairs.KvCacheFilling.CopyExecution
