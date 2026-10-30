import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Projection
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.ParametricExecutionModel
import bench.tritonbench_g.seeded_dropout.Proofs.WholeMath
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SeededDropout.ActualTile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SeededDropout
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

def Layout.source (c : Layout) (kind : SeedKind) (p : ℝ)
    (seed : SeedCarrier kind) : ComputeKernel :=
  ParametricExecutionModel.source kind c.X c.OUT c.n c.B p seed
    (RandomDomain.arange_domain c.legal)

/-- Independent pointwise source formula: fixed canonical random bits and the
original strict threshold, eager FP32 divide and source store without an extra self-cast. -/
def scalar (M : Algebra α) (lt : α → α → Bool) (p : ℝ) (seed : Int) (i : Nat) (x : α) : α :=
  let probability := M.literal (some .fp32) .real p
  let random := M.fp32Bits ⟨(Random.rand seed (i : Int) .bits32 10).bits⟩
  (if lt probability random then
      M.binary (some .fp32) .real .div (M.fp32Load x)
        (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real 1) probability)
     else M.literal (some .fp32) .real 0.0)

def tileState {α : Type} [Inhabited α] (pid : Nat) (s : State α) : State α :=
  { s with pids := fun ax => if ax = 0 then pid else 0 }

def prefixState {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : SeedKind) (seed : SeedCarrier kind) (pid : Nat) (s : State α) : State α :=
  let s' := tileState pid s
  let s' := s'.setReg "pid" .nat [] (fun _ => pid)
  let s' := s'.setReg "block_start" .nat [] (fun _ => pid*c.B)
  let s' := s'.setReg "offsets" .nat [c.B] (fun j => pid*c.B+j.1.val)
  let s' := s'.setReg "mask" .bool [c.B] (fun j => decide (pid*c.B+j.1.val < c.n))
  s'.setReg "x" .real [c.B] (fun j => M.fp32Load
    (if pid*c.B+j.1.val < c.n then (s.mem c.X (pid*c.B+j.1.val)).read .real
     else s.undef .real c.X (pid*c.B+j.1.val)))

theorem prefix_executes {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α) :
    run M ((c.source kind p seed).surfaceBody.take 5) (tileState pid s) =
      some (prefixState M c kind seed pid s) := by
  cases kind
  all_goals simp [Layout.source, ParametricExecutionModel.source,ParametricExecutionModel.sourceI32,ParametricExecutionModel.sourceI64,ParametricExecutionModel.sourceU64,Random.sourceIntCallValue,Random.sourceNatCallValue, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, natLt, numericLt, bop,
    Region.cast, Random.i32ArangeSize, prefixState, tileState,DSL.TypedStores.fp32CallValue]
  all_goals rfl

def randomTile (M : Algebra α) (c : Layout) (kind : SeedKind)
    (seed : SeedCarrier kind) (pid : Nat) : Values α .real [c.B] :=
  fun j => M.fp32Bits ⟨(Random.rand (seedValue kind seed) (pid*c.B+j.1.val : Nat) .bits32 10).bits⟩

def seedOp (kind : SeedKind) (seed : SeedCarrier kind) : Op .int [] :=
  match kind,seed with
  | .i32,s | .i64,s => .constInt s
  | .u64,s => .castNatToInt (.constNat s)

theorem prefix_rand {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : SeedKind) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : (legacyKind kind).inRange (seedValue kind seed) = Bool.true) :
    evalOp M (some .fp32) (.rand (RandomDomain.profile (legacyKind kind)) (seedOp kind seed)
      (.castNatToInt (.ref .nat [c.B] "offsets"))) (prefixState M c kind seed pid s) =
      some (randomTile M c kind seed pid) := by
  have valid := RandomDomain.all_inputs_valid (legacyKind kind) (seedValue kind seed) hs
    c.legal hn hp
  have se : evalOp M (some .fp32) (seedOp kind seed) (prefixState M c kind seed pid s) =
      some (fun _ => seedValue kind seed) := by
    cases kind <;> simp [evalOp_unfold,seedOp,prefixState,seedValue,legacyKind]
  have oe : evalOp M (some .fp32) (.castNatToInt (.ref .nat [c.B] "offsets"))
      (prefixState M c kind seed pid s) = some (fun j => (Index.offset pid c.B j.1 : Int)) := by
    simp [evalOp_unfold,prefixState,Index.offset]
  rw [evalOp_unfold]
  simp only [se,oe,bind,Option.bind]
  rw [valid]
  rfl

def sourceRun {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : (legacyKind kind).inRange (seedValue kind seed) = Bool.true) :
    { t : State α // run M (c.source kind p seed).surfaceBody (tileState pid s) = some t } := by
  cases kind
  case i32 =>
    have pre := prefix_executes M c .i32 p seed pid s
    have random := prefix_rand M c .i32 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 5 (c.source .i32 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,ParametricExecutionModel.source,ParametricExecutionModel.sourceI32,ParametricExecutionModel.sourceI64,ParametricExecutionModel.sourceU64,Random.sourceIntCallValue,Random.sourceNatCallValue,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,seedOp,RandomDomain.profile,legacyKind,seedValue] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare,DSL.TypedStores.fp32CallValue]
    rfl
  case i64 =>
    have pre := prefix_executes M c .i64 p seed pid s
    have random := prefix_rand M c .i64 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 5 (c.source .i64 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,ParametricExecutionModel.source,ParametricExecutionModel.sourceI32,ParametricExecutionModel.sourceI64,ParametricExecutionModel.sourceU64,Random.sourceIntCallValue,Random.sourceNatCallValue,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,seedOp,RandomDomain.profile,legacyKind,seedValue] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare,DSL.TypedStores.fp32CallValue]
    rfl
  case u64 =>
    have pre := prefix_executes M c .u64 p seed pid s
    have random := prefix_rand M c .u64 seed pid s hn hp hs
    refine ⟨?_, ?_⟩
    swap
    rw [← List.take_append_drop 5 (c.source .u64 p seed).surfaceBody,run_append,pre]
    simp only [Option.bind_some]
    simp only [Layout.source,ParametricExecutionModel.source,ParametricExecutionModel.sourceI32,ParametricExecutionModel.sourceI64,ParametricExecutionModel.sourceU64,Random.sourceIntCallValue,Random.sourceNatCallValue,ComputeKernel.surfaceBody,List.drop_succ_cons,
      List.drop_zero,Random.i32ArangeSize,seedOp,RandomDomain.profile,legacyKind,seedValue] at random ⊢
    simp only [run,step,evalExpr,evalComputeOp]
    erw [random]
    simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,store,
      Option.bind,ComputeDType.eraseDType,numeric,numericLt,ofFloat,toFloat,
      Region.cast,prefixState,State.setReg,compare,DSL.TypedStores.fp32CallValue]
    rfl

def writeCell (mem : RegionName → Nat → Cell α) (r : RegionName)
    (a : Nat) (v : Cell α) : RegionName → Nat → Cell α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

def raw {α : Type} [Inhabited α] (c : Layout) (pid : Nat) (s : State α)
    (j : TileIndex [c.B]) : α :=
  if pid*c.B+j.1.val < c.n then (s.mem c.X (pid*c.B+j.1.val)).read .real
  else s.undef .real c.X (pid*c.B+j.1.val)

def tileMemory {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool) (c : Layout)
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α) :
    RegionName → Nat → Cell α :=
  (TileShape.allIndices [c.B]).foldl (fun mem i =>
    if pid*c.B+i.1.val < c.n then
      writeCell mem c.OUT (pid*c.B+i.1.val)
        (.mk .real (scalar M lt p (seedValue kind seed) (pid*c.B+i.1.val) (raw c pid s i)))
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
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : (legacyKind kind).inRange (seedValue kind seed) = Bool.true) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem =
      tileMemory M lt c kind p seed pid s := by
  cases kind
  all_goals simp only [sourceRun,store]
  all_goals rw [scatter_mem]
  all_goals simp only [tileMemory,scalar,raw,tileState,State.setReg,bop,randomTile,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

theorem fold_written {α ι : Type} (r : RegionName) (off : ι → Nat)
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
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : (legacyKind kind).inRange (seedValue kind seed) = Bool.true) (target : TileIndex [c.B])
    (active : pid*c.B+target.1.val < c.n) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem c.OUT (pid*c.B+target.1.val) =
      Cell.mk .real (scalar M lt p (seedValue kind seed) (pid*c.B+target.1.val)
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
  suffices h : _ = Cell.mk .real (scalar M lt p (seedValue kind seed) (pid*c.B+target.1.val)
      (raw c pid s target)) by simpa only [raw,active,if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell,raw,active]

def Outside (c : Layout) (pid : Nat) (region : RegionName) (a : Nat) : Prop :=
  region ≠ c.OUT ∨ ∀ i : TileIndex [c.B], pid*c.B+i.1.val < c.n → a ≠ pid*c.B+i.1.val

theorem fold_frame {α ι : Type} (region : RegionName) (off : ι → Nat)
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
    (kind : SeedKind) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat) (s : State α)
    (hn : c.n < 2^31) (hp : pid < SourceDomain.grid c.n c.B)
    (hs : (legacyKind kind).inRange (seedValue kind seed) = Bool.true) (region : RegionName) (a : Nat)
    (outside : Outside c pid region a) :
    (sourceRun M lt compare c kind p seed pid s hn hp hs).val.mem region a = s.mem region a := by
  rw [sourceRun_mem]
  exact fold_frame _ _ _ _ _ _ _ _ outside

theorem scalar_real (p : ℝ) (seed : Int) (i : Nat) (x : ℝ) :
    scalar Whole.realAlgebra (fun a b => decide (a < b)) p seed i x = Whole.realFormula p seed i x := by
  have random : ((((Random.rand seed (i : Int) .bits32 10).toRat?).getD 0 : Rat) : ℝ) =
      Random.randReal seed (i : Int) .bits32 10 := by
    rw [Random.rand_toRat_randRat]
    rfl
  simp only [scalar,Whole.realAlgebra,Whole.realFormula,id_eq,decide_eq_true_eq]
  change (if p < ((((Random.rand seed (i : Int) .bits32 10).toRat?).getD 0 : Rat) : ℝ)
    then x/(1-p) else 0.0) = _
  rw [random]
  norm_num


#axiomsClean sourceRun
#axiomsClean sourceRun_mem
#axiomsClean tile_readback
#axiomsClean tile_frame
#axiomsClean scalar_real

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout.ActualTile
