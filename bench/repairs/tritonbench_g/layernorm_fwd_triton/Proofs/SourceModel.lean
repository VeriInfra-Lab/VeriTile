import bench.repairs.tritonbench_g.layernorm_fwd_triton.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section

def prelude (p : Parameters) := (kernel p).surfaceBody.take 6
def meanPost (p : Parameters) := ((kernel p).surfaceBody.drop 7).take 2
def varPost (p : Parameters) := ((kernel p).surfaceBody.drop 10).take 2
def loopBody (p : Parameters) (index : Nat) : List ComputeStmt :=
  match (kernel p).surfaceBody[index]? with
  | some (.forRange _ _ _ _ xs) => xs
  | _ => []

def xBase {α : Type} (p : Parameters) (s : State α) := s.pids 0*p.xSeqStride+s.pids 1*p.xHeadStride
def yBase {α : Type} (p : Parameters) (s : State α) := s.pids 0*p.ySeqStride+s.pids 1*p.yHeadStride
def wBase {α : Type} (p : Parameters) (s : State α) := s.pids 1*p.weightHeadStride
def initial {α : Type} (M : Algebra α) (p : Parameters) : TileIndex [p.block] → α :=
  fun _ => M.literal (some .fp32) .real 0

def xValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (n : Nat) : α :=
  M.cast (some .fp32) .fp32 .real (if n < p.columns then
    (s.mem p.x (xBase p s+n*p.xDimStride)).read .fp32
    else M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0.0))
def wValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (n : Nat) : α :=
  M.cast (some .fp32) .fp32 .real (if n < p.columns then
    (s.mem p.weight (wBase p s+n*p.weightDimStride)).read .fp32 else s.undef .fp32 p.weight (wBase p s+n*p.weightDimStride))
def meanStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (off : Nat) (a : TileIndex [p.block] → α) : TileIndex [p.block] → α := fun i =>
  M.binary (some .fp32) .real .add (a i) (xValue M p s (off+i.1.val))
def meanValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div
    (M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false
      (foldRange (meanStep M p s) 0 p.columns p.block (initial M p)) PUnit.unit)
    (M.literal (some .fp32) .real (p.columns : ℝ))
/-- The source explicitly selects zero on tail lanes before squaring. The
zero's square and accumulator addition remain operations in every iteration. -/
def centered {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (n : Nat) : α := if n < p.columns then
  M.binary (some .fp32) .real .sub (xValue M p s n) (meanValue M p s)
  else M.literal (some .fp32) .real 0.0
def varStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (off : Nat) (a : TileIndex [p.block] → α) : TileIndex [p.block] → α := fun i =>
  M.binary (some .fp32) .real .add (a i) (M.binary (some .fp32) .real .mul
    (centered M p s (off+i.1.val)) (centered M p s (off+i.1.val)))
def varianceValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div
    (M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false
      (foldRange (varStep M p s) 0 p.columns p.block (initial M p)) PUnit.unit)
    (M.literal (some .fp32) .real (p.columns : ℝ))
/-- The external epsilon register, supplied once per program by the launch.
The successful execution theorem requires this scalar register to be present. -/
def epsilonValue {α : Type} [Inhabited α] (s : State α) : α :=
  ((s.regs .real [] "eps_arg").getD (fun _ => default)) PUnit.unit

def rstdValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div (M.literal (some .fp32) .real 1)
    (M.unary (some .fp32) .sqrt (M.binary (some .fp32) .real .add
      (varianceValue M p s) (epsilonValue s)))
def yValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (n : Nat) : α := M.cast none .real .fp32
  (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .sub (xValue M p s n) (meanValue M p s))
      (rstdValue M p s)) (wValue M p s n))

def outputStep {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (off : Nat) (mem : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i => if off+i.1.val < p.columns then
    fun r o => if r=p.y ∧ o=yBase p s+(off+i.1.val)*p.yDimStride then
      .mk .fp32 (yValue M p s (off+i.1.val)) else mem r o else mem) mem
/-- Independent three-pass recurrence and scatter, with every original block
and its complete arithmetic. It is not an execution of a second kernel. -/
def referenceMem {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  foldRange (outputStep M p s) 0 p.columns p.block s.mem

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :=
  r ≠ p.y ∨ ∀ n, n < p.columns → o ≠ yBase p s+n*p.yDimStride
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Source
