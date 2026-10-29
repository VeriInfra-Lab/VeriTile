import bench.tritonbench_g.sin_kernel.Proofs.Precision

namespace SinKernelNative
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SinKernel
abbrev Storage := InputStorage
def width : Storage → Nat | .fp32 => 32 | .fp64 => 64
abbrev Word (d : Storage) := BitVec (width d)

/-- Exactly the two native scalar sine ports. Neither callback supplies
memory, access permission, a program trace or a final kernel result. -/
structure Numerics where
  sin32 : Word .fp32 → Word .fp32
  sin64 : Word .fp64 → Word .fp64

inductive Expr (d : Storage) where
  | input
  | sine (value : Expr d)

def evalSin (b : Numerics) : (d : Storage) → Word d → Word d
  | .fp32,x => b.sin32 x
  | .fp64,x => b.sin64 x

def Expr.eval (b : Numerics) (x : Word d) : Expr d → Option (Word d)
  | .input => some x
  | .sine e => (e.eval b x).map (evalSin b d)

/-- The unchanged scalar expression is sin(x), preserving fp32 or fp64.
The independently checked source precision lemma has no narrowing cast. -/
def source (d : Storage) : Expr d := .sine .input
def result (b : Numerics) : (d : Storage) → Word d → Option (Word d)
  | .fp32,x => some (b.sin32 x)
  | .fp64,x => some (b.sin64 x)

theorem source_scalar (b : Numerics) (d : Storage) (x : Word d) :
    (source d).eval b x=result b d x := by cases d <;> rfl
theorem source_total (b : Numerics) (d : Storage) (x : Word d) :
    ∃ y, (source d).eval b x=some y := by cases d <;> exact ⟨_,rfl⟩

theorem retained_source_precision (d : Storage) (X Y : RegionName) (n : Nat) :
    (typedKernel d Bool.false X Y n 1024).surfaceBody.drop 4 =
      [Precision.expectedLoad d X 1024,Precision.expectedSin d 1024,Precision.expectedStore d Y 1024] :=
  Precision.source_precision d Bool.false X Y n 1024

#specNonCircular result avoiding [source,Expr.eval]
#axiomsClean source_scalar
#axiomsClean source_total
#axiomsClean retained_source_precision
#guard_msgs (drop info) in
#auditModuleAxioms
end SinKernelNative
