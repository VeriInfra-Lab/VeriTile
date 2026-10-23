import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Projection
import bench.tritonbench_g.vector_addition_custom.Proofs.NativeArithmetic
import bench.tritonbench_g.add_example.Proofs.FinitePhysical

/-! Source-specific finite interpreter. Index/mask expressions and the ordinary
memory interface are reused. The custom source stores its add directly and
executes sixteen lanes. Inactive loads have arbitrary values; only active stores
are observable. No desired output is supplied to this interpreter. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

abbrev offset32 := VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.offset
abbrev mask32 := VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.mask
def programLanes (n pid : Nat) : List Nat := (List.range 16).filter (mask32 n pid)

def nativeLane16 (v : PhysicalView) (pid lane : Nat) : Nat := v.nativeAt (offset32 pid lane)
def lane16 (v : PhysicalView) (pid lane : Nat) : Nat := v.at (offset32 pid lane)

inductive Instruction where
  | indexAssign (name : RegName) (value : IndexExpr)
  | load (name : RegName) (region : RegionName) (offset : IndexExpr) (mask : MaskExpr)
  | storeAdd (region : RegionName) (offset : IndexExpr) (a b : RegName) (mask : MaskExpr)
  deriving DecidableEq

def lowerStatement : Stmt → Option Instruction
  | .assign .nat _ name value => return .indexAssign name (←lowerIndex value)
  | .assign .real _ name (.load .real (.region region offset) mask _) =>
      return .load name region (←lowerIndex offset) (←lowerMask mask)
  | .store .real _ (.region region offset) (.add .real _ (.ref _ _ a) (.ref _ _ b)) mask =>
      return .storeAdd region (←lowerIndex offset) a b (←lowerMask mask)
  | _ => none

def lowerBody : List Stmt → Option (List Instruction)
  | [] => some []
  | stmt::rest => return (←lowerStatement stmt)::(←lowerBody rest)

def sourceCode (X Y O : RegionName) (n : Nat) : List Instruction :=
  [.indexAssign "prog_id" (.program 0),
   .indexAssign "offs" (.add (.mul (.register "prog_id") (.literal 16)) (.lane 16)),
   .load "a" X (.register "offs") (.less (.register "offs") (.literal n)),
   .load "b" Y (.register "offs") (.less (.register "offs") (.literal n)),
   .storeAdd O (.register "offs") "a" "b" (.less (.register "offs") (.literal n))]

theorem original_ast_lowering (X Y O : RegionName) (n : Nat) :
    lowerBody (add_kernel X Y O n 16).body = some (sourceCode X Y O n) := by
  change lowerBody (add_kernel X Y O n 16).toAlgKernel.body = _
  rw [Projection.kernel_algorithm]
  simp [lowerBody,lowerStatement,lowerIndex,lowerBool,lowerMask,
    VeriTile.Bench.TritonBenchG.VectorAdditionCustom._add_kernel,ComputeKernel.body,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,bind,pure,Except.bind,Option.bind]

/-- Typed and mathematical entries share the same original source statements. -/
theorem typed_source_projection (X Y O : RegionName) (n : Nat) :
    (fp32Kernel X Y O n).eraseDType.toAlgKernel.body =
      (add_kernel X Y O n 16).toAlgKernel.body := by
  simp only [fp32Kernel, add_kernel, blockSize,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

/-- Check the actual source precision before choosing the finite primitives. -/
def lowerFP32Assign {dtype : TileDType} {shape : TileShape}
    (name : RegName) : ComputeExpr dtype shape → Option Instruction
  | .alg value =>
      if dtype = .nat then lowerStatement (.assign dtype shape name value) else none
  | .compute (.load .fp32 (.region region offset) mask _) =>
      return .load name region (← lowerIndex offset) (← lowerMask mask)
  | _ => none

def lowerFP32Store {dtype : TileDType} {shape : TileShape} :
    ComputeExpr dtype shape → MemAccess dtype shape → MaskOpt dtype shape → Option Instruction
  | .compute (.alg .fp32 (.add .real _ (.ref _ _ a) (.ref _ _ b))), .region region offset, mask =>
      return .storeAdd region (← lowerIndex offset) a b (← lowerMask mask)
  | _, _, _ => none

def lowerFP32Statement : ComputeStmt → Option Instruction
  | .assign _ _ name value => lowerFP32Assign name value
  | .store _ _ mem value mask => lowerFP32Store value mem mask
  | _ => none

def lowerFP32Body : List ComputeStmt → Option (List Instruction)
  | [] => some []
  | stmt :: rest => return (← lowerFP32Statement stmt) :: (← lowerFP32Body rest)

theorem source_fp32_lowering (X Y O : RegionName) (n : Nat) :
    lowerFP32Body (add_kernel X Y O n 16).surfaceBody = some (sourceCode X Y O n) := by
  simp [lowerFP32Body, lowerFP32Statement, lowerFP32Assign, lowerFP32Store,
    lowerStatement, lowerIndex, lowerBool, lowerMask, add_kernel,
    ComputeKernel.surfaceBody, sourceCode, Region.cast, bind, pure, Option.bind]
  exact ⟨rfl, rfl⟩

theorem untyped_source_rejected (X Y O : RegionName) (n : Nat) :
    lowerFP32Body
      (VeriTile.Bench.TritonBenchG.VectorAdditionCustom._add_kernel X Y O n 16).surfaceBody = none := by
  simp [VeriTile.Bench.TritonBenchG.VectorAdditionCustom._add_kernel,
    ComputeKernel.surfaceBody, lowerFP32Body, lowerFP32Statement, lowerFP32Assign,
    lowerFP32Store, lowerStatement, lowerIndex, bind, pure, Option.bind]

def executeInstruction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : Nat → RegName → Nat → A) (pid : Nat)
    (s : Registers M A) : Instruction → Registers M A
  | .indexAssign name value => {s with indices := fun key lane ↦ if key=name then evalIndex pid lane s.indices value else s.indices key lane}
  | .load name region offset mask => {s with values := fun key lane ↦ if key=name then if evalMask pid lane s.indices s.masks mask then ops.read s.memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) else undef pid name lane else s.values key lane}
  | .storeAdd region offset a b mask => {s with memory := (List.range 16).foldl (fun memory lane =>
      if evalMask pid lane s.indices s.masks mask then
        ops.write memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset))
          (compute (s.values a lane) (s.values b lane)) else memory) s.memory}

def executeBody {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : Nat → RegName → Nat → A) (pid : Nat) :
    List Instruction → Registers M A → Registers M A
  | [],s => s
  | stmt::rest,s => executeBody ops compute views undef pid rest (executeInstruction ops compute views undef pid s stmt)

def runProgram {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : Nat → RegName → Nat → A) (X Y O : RegionName)
    (n pid : Nat) (memory : M) : Option M := do
  let code ← lowerFP32Body (add_kernel X Y O n 16).surfaceBody
  let initial : Registers M A := ⟨fun _ _ => 0,fun _ _ => Bool.false,undef pid,memory⟩
  pure (executeBody ops compute views undef pid code initial).memory

def uncheckedWrites {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (x y output : PhysicalView) (n pid : Nat) (s : M) : List (Write A) :=
  (programLanes n pid).map fun lane => ⟨lane16 output pid lane,
    compute (ops.read s x.dtype (lane16 x pid lane)) (ops.read s y.dtype (lane16 y pid lane))⟩

theorem source_program_reduction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : Nat → RegName → Nat → A) (X Y O : RegionName) (n pid : Nat) (s : M) :
    runProgram ops compute views undef X Y O n pid s =
      some (applyWrites ops (views O).dtype (uncheckedWrites ops compute (views X) (views Y) (views O) n pid s) s) := by
  simp only [runProgram,source_fp32_lowering,Option.bind_some,
    bind,pure,Option.bind,sourceCode,executeBody,executeInstruction,evalIndex,evalMask]
  try simp only [if_pos rfl,if_true,String.reduceEq,if_false]
  congr 1
  unfold uncheckedWrites programLanes
  rw [←fold_source_stores]
  apply congrArg (fun f => List.foldl f s (List.range 16))
  funext memory lane
  change (if mask32 n pid lane then _ else _) = _
  by_cases h : mask32 n pid lane = Bool.true <;>
    simp [h,lane16,offset32,mask32,
      VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.offset,
      VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.mask,
      VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native.bound] at *
  all_goals simp [h]

/-- Any independent undefined load values at every program/register/lane give
exactly the same observable result. The host may choose a representative. -/
theorem undefined_irrelevant {M A : Type} (ops : MemoryPrimitives M A)
    (compute : A → A → A) (views : RegionName → PhysicalView)
    (u v : Nat → RegName → Nat → A) (X Y O : RegionName) (n pid : Nat) (s : M) :
    runProgram ops compute views u X Y O n pid s =
      runProgram ops compute views v X Y O n pid s := by
  rw [source_program_reduction, source_program_reduction]

#axiomsClean undefined_irrelevant
#axiomsClean original_ast_lowering
#axiomsClean typed_source_projection
#axiomsClean source_fp32_lowering
#axiomsClean untyped_source_rejected
#axiomsClean source_program_reduction
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Source
