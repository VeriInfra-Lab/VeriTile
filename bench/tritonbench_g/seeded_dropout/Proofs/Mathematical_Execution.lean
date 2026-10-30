import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Algorithm
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Memory
import VeriTile.Triton.Semantics.Eval
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Execution
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 3000

def original (seedKind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier seedKind) (B : Nat)
    (hB : SourceDomain.LegalBlock B) : Kernel :=
  Erasure.reference seedKind Bool.false x y N p seed B hB

def seedState (kind : Random.IntegerType) (seed : SeedCarrier kind) (u : BlockState) : BlockState :=
  match kind, seed with
  | .i32, s | .i64, s => u.setReg "seed_arg" .int [] (Tile.scalar s)
  | .u32, s | .u64, s => u.setReg "seed_arg" .nat [] (Tile.scalar s)

def loaded (u : BlockState) (x : RegionName) (N B : Nat) : Tile .real [B] :=
  ⟨fun j => if Index.offset u.pid B j.1 < N then
    some (u.readMem x (Index.offset u.pid B j.1)) else some (u.undef x (Index.offset u.pid B j.1))⟩

def prefixState (kind : Random.IntegerType) (x : RegionName)
    (N : Nat) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) : BlockState :=
  let s := seedState kind seed u
  let s := s.setReg "BLOCK_SIZE_arg" .nat [] (Tile.scalar B)
  let s := s.setReg "pid" .nat [] (Tile.scalar u.pid)
  let s := s.setReg "block_start" .nat [] (Tile.scalar (u.pid * B))
  let s := s.setReg "offsets" .nat [B] ⟨fun j => Index.offset u.pid B j.1⟩
  let s := s.setReg "mask" .bool [B] ⟨fun j => decide (Index.offset u.pid B j.1 < N)⟩
  s.setReg "x" .real [B] (loaded u x N B)

theorem prefix_exec (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    stepStmts ((Algorithm.kernel kind Bool.false x y N p seed B).body.take 7) u =
      some (prefixState kind x N seed B u) := by
  cases kind
  all_goals simp [Algorithm.kernel, Algorithm.seedStmt, prefixState, seedState, loaded,
    stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, Index.offset]
  all_goals rfl

theorem stepStmts_append (before after : List Stmt) (s : BlockState) :
    stepStmts (before ++ after) s = (stepStmts before s).bind (stepStmts after) := by
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts.nil, Option.bind_some]
  | cons st tail ih =>
    simp only [List.cons_append, stepStmts]
    cases stepStmt st s with
    | none => rfl
    | some t => exact ih t

theorem exec_prefix (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    exec (Algorithm.kernel kind Bool.false x y N p seed B) u =
      stepStmts ((Algorithm.kernel kind Bool.false x y N p seed B).body.drop 7)
        (prefixState kind x N seed B u) := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 7 (Algorithm.kernel kind Bool.false x y N p seed B).body]
  rw [stepStmts_append, prefix_exec, Option.bind_some]

def randomTile (seed : Int) (B : Nat) (u : BlockState) : Tile .real [B] :=
  ⟨fun j => some (Random.randReal seed (Index.offset u.pid B j.1 : Int) .bits32 10)⟩

theorem prefix_offsets (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    (prefixState kind x N seed B u).regs .nat [B] "offsets" =
      some ⟨fun j => Index.offset u.pid B j.1⟩ := by simp [prefixState]

theorem prefix_mask (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    (prefixState kind x N seed B u).regs .bool [B] "mask" =
      some ⟨fun j => decide (Index.offset u.pid B j.1 < N)⟩ := by simp [prefixState]

theorem prefix_x (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    (prefixState kind x N seed B u).regs .real [B] "x" = some (loaded u x N B) :=
  BlockState.setReg_same _ _ _ _ _

theorem prefix_seed (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    evalOp (Algorithm.seedOp kind) (prefixState kind x N seed B u) =
      some (Tile.scalar (seedInt kind seed)) := by
  cases kind
  all_goals simp [Algorithm.seedOp, prefixState, seedState, seedInt, evalOp.eq_def]

theorem prefix_rand (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B)
    (u : BlockState) (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    evalOp (.rand (RandomDomain.profile kind) (Algorithm.seedOp kind)
      (.castNatToInt (.ref .nat [B] "offsets"))) (prefixState kind x N seed B u) =
      some (randomTile (seedInt kind seed) B u) := by
  apply RandomDomain.eval_rand kind (seedInt kind seed) hseed hB hN hpid
  · exact prefix_seed kind x N seed B u
  · simp only [evalOp.eq_def, prefix_offsets, bind, Option.bind]
    rfl

theorem executes (seedKind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier seedKind) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : seedKind.inRange (seedInt seedKind seed) = Bool.true) :
    ∃ v, exec (original seedKind x y N p seed B hB) u = some v := by
  rw [original, Algorithm.source_eq, exec_prefix]
  have hr := prefix_rand seedKind x N seed B hB u hN hpid hseed
  cases he : stepStmts ((Algorithm.kernel seedKind Bool.false x y N p seed B).body.drop 7)
        (prefixState seedKind x N seed B u) with
  | some v => exact ⟨v, rfl⟩
  | none =>
    simp only [Algorithm.kernel, List.drop_succ_cons, List.drop_zero, stepStmts, stepStmt] at he
    rw [hr] at he
    simp [evalOp.eq_def, prefix_x, prefix_offsets, prefix_mask,
      Tile.bop, Tile.cop, Tile.select, NumericDType.sub, NumericDType.div] at he

def keepTile (seed : Int) (p : ℝ) (B : Nat) (u : BlockState) : Tile .bool [B] :=
  ⟨fun j => Formula.keep seed p (Index.offset u.pid B j.1)⟩

def outputTile (seed : Int) (p : ℝ) (x : RegionName) (N B : Nat)
    (u : BlockState) : Tile .real [B] :=
  ⟨fun j => if Formula.keep seed p (Index.offset u.pid B j.1) then
    Option.map (fun a => a / (1-p)) ((loaded u x N B).data j) else some 0.0⟩

def resultBase (kind : Random.IntegerType) (x : RegionName) (N : Nat) (p : ℝ)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) : BlockState :=
  let s := prefixState kind x N seed B u
  let s := s.setReg "random" .real [B] (randomTile (seedInt kind seed) B u)
  let s := s.setReg "x_keep" .bool [B] (keepTile (seedInt kind seed) p B u)
  s.setReg "output" .real [B] (outputTile (seedInt kind seed) p x N B u)

def rawResult (kind : Random.IntegerType) (x y : RegionName) (N : Nat) (p : ℝ)
    (seed : SeedCarrier kind) (B : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun s j =>
    if Index.offset u.pid B j.1 < N then
      s.writeMem y (Index.offset u.pid B j.1)
        (WithBot.unbotD 0 ((outputTile (seedInt kind seed) p x N B u).data j)) else s)
    (resultBase kind x N p seed B u)

theorem compare_real_gt (a b : ℝ) :
    ComparableDType.gt .real (some a) (some b) = decide (a > b) := by
  simp only [ComparableDType.gt, WithBot.some_eq_coe, WithBot.coe_lt_coe]

theorem exec_raw (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    exec (original kind x y N p seed B hB) u = some (rawResult kind x y N p seed B u) := by
  rw [original, Algorithm.source_eq, exec_prefix]
  have hr := prefix_rand kind x N seed B hB u hN hpid hseed
  simp only [Algorithm.kernel, List.drop_succ_cons, List.drop_zero, stepStmts, stepStmt]
  rw [hr]
  simp [evalOp.eq_def, prefix_x, prefix_offsets, prefix_mask,
    Tile.bop, Tile.cop, Tile.select, NumericDType.sub, NumericDType.div,
    compare_real_gt, rawResult, resultBase, outputTile, keepTile, randomTile, Formula.keep]

theorem output_active (seed : Int) (p : ℝ) (x : RegionName) (N B : Nat)
    (u : BlockState) (j : TileIndex [B]) (hj : Index.offset u.pid B j.1 < N) :
    WithBot.unbotD 0 ((outputTile seed p x N B u).data j) =
      Formula.expected (u.readMem x) p seed (Index.offset u.pid B j.1) := by
  simp only [outputTile, loaded, hj, if_true, Formula.keep, Formula.expected_eq_if,
    decide_eq_true_eq]
  split <;> norm_num

theorem raw_eq_storeBlock (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    rawResult kind x y N p seed B u =
      Memory.storeBlock u (resultBase kind x N p seed B u) x y N B p (seedInt kind seed) := by
  rw [rawResult, Memory.storeBlock_scatter]
  congr 1
  funext s j
  by_cases hj : Index.offset u.pid B j.1 < N
  · simp only [hj, if_true, output_active _ _ _ _ _ _ _ hj]
  · simp only [hj, if_false]

theorem exec_storeBlock (kind : Random.IntegerType) (x y : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    exec (original kind x y N p seed B hB) u =
      some (Memory.storeBlock u (resultBase kind x N p seed B u) x y N B p (seedInt kind seed)) := by
  rw [exec_raw kind x y N p seed B hB u hN hpid hseed, raw_eq_storeBlock]

theorem resultBase_mem (kind : Random.IntegerType) (x : RegionName) (N : Nat)
    (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    (resultBase kind x N p seed B u).mem = u.mem := by
  cases kind <;> rfl

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Execution
