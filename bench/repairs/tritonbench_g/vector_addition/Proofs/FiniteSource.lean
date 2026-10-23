import bench.repairs.tritonbench_g.vector_addition.Proofs.Contract
import bench.tritonbench_g.add_example.Proofs.FinitePhysical

/-! Reuse the checked finite instruction grammar, primitive memory interface and
fold/store lemma. The only interpreter specialization is the source's 1024 lanes.
No library implementation, scalar numerical law or source result is a premise. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.FiniteSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def programLanes (n pid : Nat) : List Nat :=
  (List.range 1024).filter (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid)

def sourceCode (X Y O : RegionName) (n : Nat) : List Instruction :=
  [.indexAssign "pid" (.program 0),
   .indexAssign "block_start" (.mul (.register "pid") (.literal 1024)),
   .indexAssign "offsets" (.add (.register "block_start") (.lane 1024)),
   .maskAssign "mask" (.less (.register "offsets") (.literal n)),
   .load "x" X (.register "offsets") (.register "mask"),
   .load "y" Y (.register "offsets") (.register "mask"),
   .addAssign "output" "x" "y",
   .store O (.register "offsets") "output" (.register "mask")]

theorem original_ast_lowering (X Y O : RegionName) (n : Nat) :
    lowerBody (add_kernel X Y O n 1024).body=some (sourceCode X Y O n) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerBool,lowerMask,add_kernel,VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel,ComputeKernel.body,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Region.cast,sourceCode,bind,pure,Except.bind,Option.bind]

/-- Task-specific lowering checks precision before producing finite code.
Only explicit FP32 loads, addition and output are accepted. Index/mask
expressions reuse the existing discrete lowerer. -/
def lowerFP32Assign {dtype : TileDType} {shape : TileShape}
    (name : RegName) : ComputeExpr dtype shape → Option Instruction
  | .alg value =>
      if dtype = .nat ∨ dtype = .bool then
        lowerStatement (.assign dtype shape name value) else none
  | .compute (.load .fp32 (.region region offset) mask _) =>
      return .load name region (← lowerIndex offset) (← lowerMask mask)
  | .compute (.alg .fp32 (.add .real _ (.ref _ _ a) (.ref _ _ b))) =>
      some (.addAssign name a b)
  | _ => none

def lowerFP32Store {dtype : TileDType} {shape : TileShape} :
    ComputeExpr dtype shape → MemAccess dtype shape → MaskOpt dtype shape → Option Instruction
  | .compute (.alg .fp32 (.ref _ _ value)), .region region offset, mask =>
      return .store region (← lowerIndex offset) value (← lowerMask mask)
  | _, _, _ => none

def lowerFP32Statement : ComputeStmt → Option Instruction
  | .assign _ _ name value => lowerFP32Assign name value
  | .store _ _ mem value mask => lowerFP32Store value mem mask
  | _ => none

def lowerFP32Body : List ComputeStmt → Option (List Instruction)
  | [] => some []
  | stmt :: rest => return (← lowerFP32Statement stmt) :: (← lowerFP32Body rest)

/-- The finite interpreter consumes the canonical source with all FP32 tags
checked, without first erasing the precision information. -/
theorem source_fp32_lowering (X Y O : RegionName) (n : Nat) :
    lowerFP32Body (add_kernel X Y O n 1024).surfaceBody =
      some (sourceCode X Y O n) := by
  simp [lowerFP32Body, lowerFP32Statement, lowerFP32Assign, lowerFP32Store, lowerStatement, lowerIndex,
    lowerBool, lowerMask, add_kernel, ComputeKernel.surfaceBody, sourceCode,
    Region.cast, bind, pure, Option.bind]
  exact ⟨rfl, rfl⟩

/-- An untagged mathematical source cannot enter this FP32 execution path. -/
theorem untyped_source_rejected (X Y O : RegionName) (n : Nat) :
    lowerFP32Body
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel X Y O n 1024).surfaceBody = none := by
  simp [VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel, ComputeKernel.surfaceBody,
    lowerFP32Body, lowerFP32Statement, lowerFP32Assign, lowerFP32Store,
    lowerStatement, lowerIndex, lowerBool, bind, pure, Option.bind]

def executeInstruction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat) (s : Registers M A) : Instruction → Registers M A
  | .store region offset value mask => {s with memory := (List.range 1024).foldl (fun memory lane ↦ if evalMask pid lane s.indices s.masks mask then ops.write memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) (s.values value lane) else memory) s.memory}
  | instruction => VeriTile.Bench.TritonBenchG.AddExample.Finite.executeInstruction
      ops compute views undef pid s instruction

def executeBody {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat) : List Instruction → Registers M A → Registers M A
  | [],s => s
  | stmt::rest,s => executeBody ops compute views undef pid rest (executeInstruction ops compute views undef pid s stmt)

def runProgram {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (X Y O : RegionName) (n pid : Nat) (memory : M) : Option M := do
  let code ← lowerFP32Body (add_kernel X Y O n 1024).surfaceBody
  let initial : Registers M A := ⟨fun _ _ => 0,fun _ _ => Bool.false,fun _ _ => undef,memory⟩
  pure (executeBody ops compute views undef pid code initial).memory

def uncheckedWrites {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (x y output : PhysicalView) (n pid : Nat) (s : M) : List (Write A) :=
  (programLanes n pid).map fun lane => ⟨output.at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane),
    compute (ops.read s x.dtype (x.at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane))) (ops.read s y.dtype (y.at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane)))⟩

theorem source_program_reduction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (X Y O : RegionName) (n pid : Nat) (s : M) :
    runProgram ops compute views undef X Y O n pid s=
      some (applyWrites ops (views O).dtype (uncheckedWrites ops compute (views X) (views Y) (views O) n pid s) s) := by
  simp only [runProgram,source_fp32_lowering,Option.bind_some,bind,pure,Option.bind,sourceCode,
    executeBody,executeInstruction,
    VeriTile.Bench.TritonBenchG.AddExample.Finite.executeInstruction,evalIndex,evalMask]
  try simp only [if_pos rfl,if_true,String.reduceEq,if_false]
  congr 1
  unfold uncheckedWrites programLanes
  rw [←fold_source_stores]
  apply congrArg (fun f => List.foldl f s (List.range 1024))
  funext memory lane
  change (if VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid lane then _ else _)=_
  by_cases h : VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid lane=Bool.true <;>
    simp [h,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.start32,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32] at *
  all_goals simp [h]

#axiomsClean untyped_source_rejected
#axiomsClean source_fp32_lowering
#axiomsClean original_ast_lowering
#axiomsClean source_program_reduction
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.FiniteSource
