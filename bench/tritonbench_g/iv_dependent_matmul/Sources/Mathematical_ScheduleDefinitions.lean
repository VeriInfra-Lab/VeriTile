import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_SourceDefinitions

/-! FP32 source for all five documented induction-variable schedules. The
shared prologue, load masks, dot/update, half narrowing and FP32 store are the
existing typed source. Only the original pointer staging statements vary.
The compiled DotAcc graph remains separate from source Dot+Add. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels
open VeriTile.Triton FPValue

inductive Schedule where
  | preLoad | postLoad | mixed | two | three
  deriving DecidableEq, Repr

def Schedule.name : Schedule → String
  | .preLoad => "pre_load"
  | .postLoad => "post_load"
  | .mixed => "post_pre_mixed"
  | .two => "post_load_two_iters"
  | .three => "post_load_three_iters"

def Schedule.stages (s : Schedule) : Nat := if s = .three then 4 else 3

def pointer (shape : TileShape) (target base : RegName) (offset : Op .nat []) : ComputeStmt :=
  .assign .ptr shape target (.alg (.ptrAdd (by cases shape with | nil => exact .nil | cons _ _ => exact .scalarR)
    (.ref .ptr shape base) offset))

def copyPointer (shape : TileShape) (target base : RegName) : ComputeStmt :=
  .assign .ptr shape target (.alg (.ref .ptr shape base))

def iteration (extra : Nat) : Op .nat [] :=
  if extra = 0 then .ref .nat [] "k" else .add .nat .nil (.ref .nat [] "k") (.constNat extra)

def strideOffset (p : Parameters) (stride : Nat) (k : Op .nat []) : Op .nat [] :=
  .mul .nat .nil (.mul .nat .nil k (.constNat p.blockInner)) (.constNat stride)

def aAt (p : Parameters) (target : RegName) (k : Op .nat []) : ComputeStmt :=
  pointer [p.blockRows,p.blockInner] target "a_ptr" (strideOffset p p.inputInnerStride k)

def bAt (p : Parameters) (target : RegName) (k : Op .nat []) : ComputeStmt :=
  pointer [p.blockInner,p.blockColumns] target "b_ptr" (strideOffset p p.weightInnerStride k)

def initialPointers (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  [copyPointer [p.blockRows,p.blockInner] "a_ptrs" "a_ptr",
   copyPointer [p.blockInner,p.blockColumns] "b_ptrs" "b_ptr"] ++
  match schedule with
  | .two => [aAt p "a_ptrs_next" (.constNat 1), bAt p "b_ptrs_next" (.constNat 1)]
  | .three => [aAt p "a_ptrs_next" (.constNat 1), bAt p "b_ptrs_next" (.constNat 1),
      aAt p "a_ptrs_next_next" (.constNat 2), bAt p "b_ptrs_next_next" (.constNat 2)]
  | _ => []

def beforeLoads (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  match schedule with
  | .preLoad => [aAt p "a_ptrs" (iteration 0), bAt p "b_ptrs" (iteration 0)]
  | .mixed => [aAt p "a_ptrs" (iteration 0)]
  | _ => []

def afterDot (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  match schedule with
  | .preLoad => []
  | .postLoad => [aAt p "a_ptrs" (iteration 1), bAt p "b_ptrs" (iteration 1)]
  | .mixed => [bAt p "b_ptrs" (iteration 1)]
  | .two => [copyPointer [p.blockRows,p.blockInner] "a_ptrs" "a_ptrs_next",
      copyPointer [p.blockInner,p.blockColumns] "b_ptrs" "b_ptrs_next",
      aAt p "a_ptrs_next" (iteration 2), bAt p "b_ptrs_next" (iteration 2)]
  | .three => [copyPointer [p.blockRows,p.blockInner] "a_ptrs" "a_ptrs_next",
      copyPointer [p.blockInner,p.blockColumns] "b_ptrs" "b_ptrs_next",
      copyPointer [p.blockRows,p.blockInner] "a_ptrs_next" "a_ptrs_next_next",
      copyPointer [p.blockInner,p.blockColumns] "b_ptrs_next" "b_ptrs_next_next",
      aAt p "a_ptrs_next_next" (iteration 3), bAt p "b_ptrs_next_next" (iteration 3)]

def prologueWithPointers (schedule : Schedule) (p : Parameters) : List ComputeStmt :=
  (FPValue.prologue p).take 9 ++ initialPointers schedule p ++ (FPValue.prologue p).drop 9

def body (schedule : Schedule) (graph : Graph) (p : Parameters) : List ComputeStmt :=
  beforeLoads schedule p ++ (FPValue.body graph p).drop 2 ++ afterDot schedule p

/-- The actual selected schedule, including initial aliases/prefetch pointers.
Pointer staging is natural-number source indexing; finite-width correspondence
is a separate host configuration condition. -/
def source (schedule : Schedule) (graph : Graph) (p : Parameters) : ComputeKernel :=
  .mk (FPValue.source graph p).inputs (FPValue.source graph p).outputs
    (prologueWithPointers schedule p ++ [.forRangeDyn "k" (.constNat 0) (stopOp p) (.constNat 1)
      (body schedule graph p)] ++ FPValue.suffix p)

end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels
