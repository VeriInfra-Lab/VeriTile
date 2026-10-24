import bench.tritonbench_g.multinomial_sampling.Proofs.FPInvariant
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def headCode (p : Parameters) := (prologue .fp32 p).take 7
def offsetExpr (p : Parameters) : Op .int [p.block] :=
  Op.intToSigned 32 (.load .int (.region p.offsets (.ref .nat [p.block] "off"))
    (.mask (.ref .bool [p.block] "off_mask")))
def offsetCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block] "offset" (.alg (offsetExpr p))]
def lastCode (p : Parameters) := (prologue .fp32 p).drop 8

theorem prologue_decomposition (d : Storage) (p : Parameters) :
    prologue d p = headCode p ++ offsetCode p ++ lastCode p := by cases d <;> rfl

@[simp] theorem eval_offsetExpr (M : Algebra α) (precision : Option ComputeDType)
    (p : Parameters) (s : State α) :
    FP.Structural.evalOp M precision (offsetExpr p) s = (do
      let off ← s.regs .nat [p.block] "off"
      let mask ← s.regs .bool [p.block] "off_mask"
      pure (fun i => (BitVec.ofInt 32 (intLoad s p.offsets (off i) (mask i))).toInt)) := by
  rw [offsetExpr, FP.Structural.eval_intToSigned]
  cases he : s.regs .nat [p.block] "off" <;>
    cases hm : s.regs .bool [p.block] "off_mask" <;>
    simp [FP.Structural.evalOp, intLoad, Region.cast, Option.bind, he, hm]
  try rfl

attribute [local irreducible] offsetExpr

def headExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest P) (headCode p) s = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [headCode, prologue, source, original, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, FP.Structural.evalOp, backend,
    FP.Structural.numeric, FP.Structural.natLt, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind,
    Random.i32ArangeSize]
  try rfl

@[local simp] theorem headExecution_off (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.regs .nat [p.block] "off" = some (row p s) := by
  simp [headExecution, row]
  funext i
  rfl
@[local simp] theorem headExecution_n_off (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.regs .nat [p.blockTokens] "n_off" = some (fun i => i.1.val) := by
  simp [headExecution]
@[local simp] theorem headExecution_mask (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.regs .bool [p.block] "off_mask" = some (fun i => decide (active p s i)) := by
  simp [headExecution, active, row]
  funext i
  rfl
@[local simp] theorem headExecution_seed (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.regs .int [p.block] "seed" = some (seedValue p s) := by
  simp [headExecution]
  funext i
  simp [seedValue, intLoad, active, row, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
@[local simp] theorem headExecution_mem (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.mem = s.mem := by rfl
@[local simp] theorem headExecution_undef (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (headExecution rest P p s).val.undef = s.undef := by rfl
attribute [local irreducible] headExecution

def offsetExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest P) (offsetCode p) (headExecution rest P p s).val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [offsetCode, run, step, evalExpr, eval_offsetExpr, intLoad, offsetValue, Option.bind]
  try rfl

@[local simp] theorem offsetExecution_offset (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.regs .int [p.block] "offset" = some (offsetValue p s) := by
  simp [offsetExecution]
  funext i
  simp [offsetValue, intLoad]
@[local simp] theorem offsetExecution_off (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.regs .nat [p.block] "off" = some (row p s) := by simp [offsetExecution]
@[local simp] theorem offsetExecution_n_off (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.regs .nat [p.blockTokens] "n_off" = some (fun i => i.1.val) := by simp [offsetExecution]
@[local simp] theorem offsetExecution_mask (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.regs .bool [p.block] "off_mask" = some (fun i => decide (active p s i)) := by simp [offsetExecution]
@[local simp] theorem offsetExecution_seed (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.regs .int [p.block] "seed" = some (seedValue p s) := by simp [offsetExecution]
@[local simp] theorem offsetExecution_mem (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.mem = s.mem := by simp [offsetExecution]
@[local simp] theorem offsetExecution_undef (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    (offsetExecution rest P p s).val.undef = s.undef := by simp [offsetExecution]
attribute [local irreducible] offsetExecution

theorem narrowed32_inRange (z : Int) :
    Random.IntegerType.i32.inRange (BitVec.ofInt 32 z).toInt = Bool.true := by
  simp only [Random.IntegerType.inRange, decide_eq_true_eq]
  constructor
  · simpa using BitVec.le_toInt (BitVec.ofInt 32 z)
  · simpa using @BitVec.toInt_lt 32 (BitVec.ofInt 32 z)

theorem narrowed32_bmod_inRange (z : Int) :
    Random.IntegerType.i32.inRange (z.bmod 4294967296) = Bool.true := by
  simpa only [BitVec.toInt_ofInt] using narrowed32_inRange z

theorem random_valid (p : Parameters) (s : State α) (h : SeedDomain p s) :
    (Random.SourceProfile.mk .i64 .i32 10).tensorInputsValid (seedValue p s) (offsetValue p s) = Bool.true := by
  simp only [Random.SourceProfile.tensorInputsValid_iff]
  exact ⟨h,fun i => narrowed32_inRange _⟩

/-- Every seed and offset remains a per-lane source operand. The concrete ten
round Philox implementation executes before its fp32 bit result is embedded. -/
def lastExecution (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α)
    (h : SeedDomain p s) :
    {t : State α // run (backend rest P) (lastCode p) (offsetExecution rest P p s).val = some t} := by
  have hs (a : Fin p.block) (b : TileIndex []) :
      Random.IntegerType.i64.inRange (seedValue p s (a,b)) = Bool.true := h (a,b)
  refine ⟨?_,?_⟩
  swap
  simp [lastCode, prologue, source, original, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, FP.Structural.evalOp, backend,
    hs, narrowed32_bmod_inRange, offsetValue, FP.Structural.numeric, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast, Option.bind,
    Random.i32ArangeSize]
  try rfl

theorem lastInvariant (rest : Algebra α) (P : Primitives α) (p : Parameters) (s : State α)
    (h : SeedDomain p s) :
    Invariant P p s (initialAccumulator P p s) (lastExecution rest P p s h).val := by
  constructor
  · simp [lastExecution]
  · simp [lastExecution]
  · simp [lastExecution]
  · simp [lastExecution]
  · simp [lastExecution]
    funext i
    simp [sample,offsetValue,Random.IntegerType.offsetWidth]
  · simp [lastExecution,initialAccumulator]
  · simp [lastExecution,initialAccumulator]
    funext i
    simp [initialOutput,intLoad,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def preludeExecution (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (s : State α) (h : SeedDomain p s) :
    {t : State α // run (backend rest P) (prologue d p) s=some t} := by
  refine ⟨(lastExecution rest P p s h).val,?_⟩
  rw [prologue_decomposition]
  simp only [run_append,(headExecution rest P p s).property,Option.bind_some,(offsetExecution rest P p s).property,(lastExecution rest P p s h).property]

theorem preludeInvariant (rest : Algebra α) (P : Primitives α) (d : Storage)
    (p : Parameters) (s : State α) (h : SeedDomain p s) :
    Invariant P p s (initialAccumulator P p s) (preludeExecution rest P d p s h).val :=
  lastInvariant rest P p s h

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean prologue_decomposition
#axiomsClean headExecution
#axiomsClean offsetExecution
#axiomsClean random_valid
#axiomsClean lastExecution
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
