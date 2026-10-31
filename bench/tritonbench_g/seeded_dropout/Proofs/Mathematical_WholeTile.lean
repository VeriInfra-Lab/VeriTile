import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_TypedExecution
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Layout where
  X : RegionName
  OUT : RegionName
  n : Nat
  B : Nat
  legal : SourceDomain.LegalBlock B

def Layout.source (c : Layout) (kind : Random.IntegerType) (p : ℝ)
    (seed : SeedCarrier kind) : ComputeKernel :=
  TypedSource.source .fp32 kind Bool.false c.X c.OUT c.n p seed c.B c.legal

/-- Independent pointwise source formula: fixed canonical random bits and the
original strict threshold, eager FP32 divide and explicit store conversion. -/
def scalar (M : Algebra α) (lt : α → α → Bool) (p : ℝ) (seed : Int) (i : Nat) (x : α) : α :=
  let probability := M.literal (some .fp32) .real p
  let random := M.fp32Bits ⟨(Random.rand seed (i : Int) .bits32 10).bits⟩
  M.cast (some .fp32) .real .real
    (if lt (M.literal none .real p) random then
      M.binary (some .fp32) .real .div (M.fp32Load x)
        (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real 1) probability)
     else M.literal (some .fp32) .real 0.0)

def tileState {α : Type} [Inhabited α] (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else 0 }

def seedState {α : Type} [Inhabited α] (kind : Random.IntegerType)
    (seed : SeedCarrier kind) (pid : Nat) (s : State α) : State α :=
  match kind, seed with
  | .i32, seed | .i64, seed => (tileState pid s).setReg "seed_arg" .int [] (fun _ => seed)
  | .u32, seed | .u64, seed => (tileState pid s).setReg "seed_arg" .nat [] (fun _ => seed)

def prefixState {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : Random.IntegerType) (seed : SeedCarrier kind) (pid : Nat) (s : State α) : State α :=
  let s' := seedState kind seed pid s
  let s' := s'.setReg "BLOCK_SIZE_arg" .nat [] (fun _ => c.B)
  let s' := s'.setReg "pid" .nat [] (fun _ => pid)
  let s' := s'.setReg "block_start" .nat [] (fun _ => pid*c.B)
  let s' := s'.setReg "offsets" .nat [c.B] (fun j => pid*c.B+j.1.val)
  let s' := s'.setReg "mask" .bool [c.B] (fun j => decide (pid*c.B+j.1.val < c.n))
  s'.setReg "x" .real [c.B] (fun j => M.fp32Load
    (if pid*c.B+j.1.val < c.n then (s.mem c.X (pid*c.B+j.1.val)).read .real
     else s.undef .real c.X (pid*c.B+j.1.val)))

theorem prefix_executes {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α) :
    run M ((c.source kind p seed).surfaceBody.take 7) (tileState pid s) =
      some (prefixState M c kind seed pid s) := by
  cases kind
  all_goals simp [Layout.source, TypedSource.source,VeriTile.Bench.Repairs.SeededDropout.sourceFamily, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, natLt, numericLt, bop,
    Region.cast, Random.i32ArangeSize, prefixState, seedState, tileState]
  all_goals rfl

def randomTile (M : Algebra α) (c : Layout) (kind : Random.IntegerType)
    (seed : SeedCarrier kind) (pid : Nat) : Values α .real [c.B] :=
  fun j => M.fp32Bits ⟨(Random.rand (seedInt kind seed) (pid*c.B+j.1.val : Nat) .bits32 10).bits⟩

theorem prefix_rand {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : Random.IntegerType) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : kind.inRange (seedInt kind seed) = Bool.true) :
    evalOp M (some .fp32) (.rand (RandomDomain.profile kind) (TypedExecution.seedOp kind)
      (.castNatToInt (.ref .nat [c.B] "offsets"))) (prefixState M c kind seed pid s) =
      some (randomTile M c kind seed pid) := by
  have valid := RandomDomain.all_inputs_valid kind (seedInt kind seed) hs
    c.legal hn hp
  have se : evalOp M (some .fp32) (TypedExecution.seedOp kind) (prefixState M c kind seed pid s) =
      some (fun _ => seedInt kind seed) := by
    cases kind <;> simp [evalOp_unfold,TypedExecution.seedOp,prefixState,seedState,seedInt]
  have oe : evalOp M (some .fp32) (.castNatToInt (.ref .nat [c.B] "offsets"))
      (prefixState M c kind seed pid s) = some (fun j => (Index.offset pid c.B j.1 : Int)) := by
    simp [evalOp_unfold,prefixState,Index.offset]
  rw [evalOp_unfold]
  simp only [se,oe,bind,Option.bind]
  rw [valid]
  rfl

def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : kind.inRange (seedInt kind seed) = Bool.true) :
    { t : State α // run M (c.source kind p seed).surfaceBody (tileState pid s) = some t } := by
  cases kind
  case i32 =>
    have pre := prefix_executes M c .i32 p seed pid s
    have random := prefix_rand M c .i32 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 7 (c.source .i32 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,TypedSource.source,VeriTile.Bench.Repairs.SeededDropout.sourceFamily,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,TypedExecution.seedOp,RandomDomain.profile] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare]
    rfl
  case i64 =>
    have pre := prefix_executes M c .i64 p seed pid s
    have random := prefix_rand M c .i64 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 7 (c.source .i64 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,TypedSource.source,VeriTile.Bench.Repairs.SeededDropout.sourceFamily,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,TypedExecution.seedOp,RandomDomain.profile] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare]
    rfl
  case u32 =>
    have pre := prefix_executes M c .u32 p seed pid s
    have random := prefix_rand M c .u32 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 7 (c.source .u32 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,TypedSource.source,VeriTile.Bench.Repairs.SeededDropout.sourceFamily,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,TypedExecution.seedOp,RandomDomain.profile] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare]
    rfl
  case u64 =>
    have pre := prefix_executes M c .u64 p seed pid s
    have random := prefix_rand M c .u64 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 7 (c.source .u64 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,TypedSource.source,VeriTile.Bench.Repairs.SeededDropout.sourceFamily,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,TypedExecution.seedOp,RandomDomain.profile] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare]
    rfl

def writeCell (mem : RegionName → Nat → Cell α) (r : RegionName)
    (a : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

def raw {α : Type} [Inhabited α] (c : Layout) (pid : Nat) (s : State α)
    (j : TileIndex [c.B]) : α :=
  if pid*c.B+j.1.val < c.n then (s.mem c.X (pid*c.B+j.1.val)).read .real
  else s.undef .real c.X (pid*c.B+j.1.val)

def tileMemory {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [c.B]).foldl (fun mem i =>
    if pid*c.B+i.1.val < c.n then
      writeCell mem c.OUT (pid*c.B+i.1.val)
        (.mk .real (scalar M lt p (seedInt kind seed) (pid*c.B+i.1.val) (raw c pid s i)))
    else mem) s.mem

theorem scatter_mem {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      lanes.foldl (fun mem i => if active i then writeCell mem r (off i) (value i) else mem) s.mem := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi, if_true, State.write, writeCell] using ih (s.write r (off i) (value i))
    · simpa only [hi, if_false] using ih s

theorem sourceRun_mem {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : kind.inRange (seedInt kind seed) = Bool.true) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem =
      tileMemory M lt c kind p seed pid s := by
  cases kind
  all_goals simp only [sourceRun,store]
  all_goals rw [scatter_mem]
  all_goals simp only [tileMemory,scalar,raw,seedState,tileState,State.setReg,bop,randomTile,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

private theorem fold_written {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (target : ι) (unique : ∀ i, active i → off i = off target → i = target)
    (lanes : List ι) (mem : RegionName → Nat → Cell α)
    (h : mem r (off target) = value target) :
    (lanes.foldl (fun current i => if active i then writeCell current r (off i) (value i)
      else current) mem) r (off target) = value target := by
  induction lanes generalizing mem with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hi : active i
    · simp only [hi, if_true]
      by_cases ho : off i = off target
      · have hit := unique i hi ho
        subst i
        simp [writeCell]
      · simp [writeCell, Ne.symm ho, h]
    · simpa only [hi, if_false] using h

theorem tile_readback {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : kind.inRange (seedInt kind seed) = Bool.true) (target : TileIndex [c.B])
    (active : pid*c.B+target.1.val < c.n) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem c.OUT (pid*c.B+target.1.val) =
      Cell.mk .real (scalar M lt p (seedInt kind seed) (pid*c.B+target.1.val)
        ((s.mem c.X (pid*c.B+target.1.val)).read .real)) := by
  rw [sourceRun_mem]
  have unique : ∀ i : TileIndex [c.B], pid*c.B+i.1.val < c.n →
      pid*c.B+i.1.val = pid*c.B+target.1.val → i = target := by
    intro i hi he
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before,after,hall⟩ := List.append_of_mem (TileShape.mem_allIndices [c.B] target)
  unfold tileMemory
  rw [hall,List.foldl_append,List.foldl_cons]
  simp only [active,if_true]
  suffices h : _ = Cell.mk .real (scalar M lt p (seedInt kind seed) (pid*c.B+target.1.val)
      (raw c pid s target)) by simpa only [raw,active,if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell,raw,active]

def Outside (c : Layout) (pid : Nat) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.OUT ∨ ∀ i : TileIndex [c.B], pid*c.B+i.1.val < c.n → a ≠ pid*c.B+i.1.val

private theorem fold_frame {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (mem : RegionName → Nat → Cell α) (r : RegionName) (a : Nat)
    (outside : r ≠ region ∨ ∀ i, active i → a ≠ off i) :
    (lanes.foldl (fun current i => if active i then writeCell current region (off i) (value i)
      else current) mem) r a = mem r a := by
  induction lanes generalizing mem with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : active i
    · simp only [hi, if_true, writeCell]
      rw [if_neg]
      rcases outside with hr | ha
      · exact fun h => hr h.1
      · exact fun h => ha i hi h.2
    · simp only [hi, if_false]

theorem tile_frame {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : kind.inRange (seedInt kind seed) = Bool.true) (region : RegionName) (a : Nat)
    (outside : Outside c pid region a) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem region a = s.mem region a := by
  rw [sourceRun_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
