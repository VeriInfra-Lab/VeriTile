import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Contract
import bench.tritonbench_g.add_example.Proofs.FinitePhysical

/-! Finite execution of the unchanged binary source. Reuses the established
index grammar, typed byte-memory interface and local frame laws. The ratioLogProduct instruction is the exact nested div/log/mul expression;
inactive loads are arbitrary and never contribute a stored observation. -/
namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
noncomputable section
open Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def programLanes (n pid : Nat) : List Nat :=
  (List.range 1024).filter (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid)

def laneAddress (v : PhysicalView) (pid lane : Nat) : Nat :=
  v.at (VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32 pid lane)

inductive Instruction where
  | indexAssign (name : RegName) (value : IndexExpr)
  | maskAssign (name : RegName) (value : MaskExpr)
  | load (name : RegName) (region : RegionName) (offset : IndexExpr) (mask : MaskExpr)
  | ratioLogProduct (name a b : RegName)
  | store (region : RegionName) (offset : IndexExpr) (value : RegName) (mask : MaskExpr)

def lowerStatement : Stmt → Option Instruction
  | .assign .nat _ name value => return .indexAssign name (←lowerIndex value)
  | .assign .bool _ name value => return .maskAssign name (←lowerBool value)
  | .assign .real _ name (.load .real (.region region offset) mask _) =>
      return .load name region (←lowerIndex offset) (←lowerMask mask)
  | .assign .real _ name (.mul .real _ (.ref _ _ a) (.log (.div .real _ (.ref _ _ numerator) (.ref _ _ b)))) =>
      if a=numerator then some (.ratioLogProduct name a b) else none
  | .store .real _ (.region region offset) (.ref _ _ value) mask =>
      return .store region (←lowerIndex offset) value (←lowerMask mask)
  | _ => none

def lowerBody : List Stmt → Option (List Instruction)
  | [] => some []
  | stmt::rest => return (←lowerStatement stmt)::(←lowerBody rest)

def sourceCode (X Y O : RegionName) (n : Nat) : List Instruction :=
  [.indexAssign "pid" (.program 0),
   .indexAssign "block_start" (.mul (.register "pid") (.literal 1024)),
   .indexAssign "offsets" (.add (.register "block_start") (.lane 1024)),
   .maskAssign "mask" (.less (.register "offsets") (.literal n)),
   .load "x" X (.register "offsets") (.register "mask"),
   .load "y" Y (.register "offsets") (.register "mask"),
   .ratioLogProduct "output" "x" "y",
   .store O (.register "offsets") "output" (.register "mask")]

theorem original_ast_lowering (X Y O : RegionName) (n : Nat) :
    lowerBody (kernel X Y O n 1024).body=some (sourceCode X Y O n) := by
  simp [lowerBody,lowerStatement,lowerIndex,lowerBool,lowerMask,kernel,VeriTile.Bench.TritonBenchG.KldivCompute.kldivergence_kernel,ComputeKernel.body,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,sourceCode,bind,pure,Except.bind,Option.bind]

structure Registers (M A : Type) where
  indices : RegName → Nat → BitVec 32
  masks : RegName → Nat → Bool
  values : RegName → Nat → A
  memory : M

def evalIndex (pid lane : Nat) (registers : RegName → Nat → BitVec 32) : IndexExpr → BitVec 32
  | .literal n => BitVec.ofNat 32 n
  | .program ax => if ax=0 then BitVec.ofNat 32 pid else 0
  | .register name => registers name lane
  | .lane _ => BitVec.ofNat 32 lane
  | .add a b => evalIndex pid lane registers a + evalIndex pid lane registers b
  | .mul a b => evalIndex pid lane registers a * evalIndex pid lane registers b

def evalMask (pid lane : Nat) (indices : RegName → Nat → BitVec 32)
    (masks : RegName → Nat → Bool) : MaskExpr → Bool
  | .register name => masks name lane
  | .less a b => (evalIndex pid lane indices a).slt (evalIndex pid lane indices b)

/-- Unchecked structural execution. Applicability observations are separately
computed from every actual active access; none is a source runtime guard.
Masked lanes never perform memory operations, as specified by
[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
and [Triton store](https://triton-lang.org/main/python-api/generated/triton.language.store.html).
For the corresponding instruction guards see
[PTX Predicated Execution](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#predicated-execution).
The inactive load value is arbitrary; `source_program_reduction` proves that
it cannot affect this pointwise kernel's observable writes. -/
def executeInstruction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat) (s : Registers M A) : Instruction → Registers M A
  | .indexAssign name value => {s with indices := fun key lane ↦ if key=name then evalIndex pid lane s.indices value else s.indices key lane}
  | .maskAssign name value => {s with masks := fun key lane ↦ if key=name then evalMask pid lane s.indices s.masks value else s.masks key lane}
  | .load name region offset mask => {s with values := fun key lane ↦ if key=name then if evalMask pid lane s.indices s.masks mask then ops.read s.memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) else undef else s.values key lane}
  | .ratioLogProduct name a b => {s with values := fun key lane ↦ if key=name then compute (s.values a lane) (s.values b lane) else s.values key lane}
  | .store region offset value mask => {s with memory := (List.range 1024).foldl (fun memory lane ↦ if evalMask pid lane s.indices s.masks mask then ops.write memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) (s.values value lane) else memory) s.memory}

def executeBody {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat) : List Instruction → Registers M A → Registers M A
  | [],s => s
  | stmt::rest,s => executeBody ops compute views undef pid rest (executeInstruction ops compute views undef pid s stmt)

def runProgram {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (X Y O : RegionName) (n pid : Nat) (memory : M) : Option M := do
  let code ← lowerBody (kernel X Y O n 1024).body
  let initial : Registers M A := ⟨fun _ _ => 0,fun _ _ => Bool.false,fun _ _ => undef,memory⟩
  pure (executeBody ops compute views undef pid code initial).memory

def uncheckedWrites {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (x y output : PhysicalView) (n pid : Nat) (s : M) : List (Write A) :=
  (programLanes n pid).map fun lane => ⟨laneAddress output pid lane,
    compute (ops.read s x.dtype (laneAddress x pid lane)) (ops.read s y.dtype (laneAddress y pid lane))⟩

theorem source_program_reduction {M A : Type} (ops : MemoryPrimitives M A) (compute : A → A → A)
    (views : RegionName → PhysicalView) (undef : A) (X Y O : RegionName) (n pid : Nat) (s : M) :
    runProgram ops compute views undef X Y O n pid s=
      some (applyWrites ops (views O).dtype (uncheckedWrites ops compute (views X) (views Y) (views O) n pid s) s) := by
  simp only [runProgram,original_ast_lowering,Option.bind_some,bind,pure,Option.bind,sourceCode,
    executeBody,executeInstruction,evalIndex,evalMask]
  try simp only [if_pos rfl,if_true,String.reduceEq,if_false]
  congr 1
  unfold uncheckedWrites programLanes
  rw [←fold_source_stores]
  apply congrArg (fun f => List.foldl f s (List.range 1024))
  funext memory lane
  change (if VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid lane then _ else _)=_
  by_cases h : VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32 n pid lane=Bool.true <;>
    simp [h,laneAddress,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.offset32,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.start32,VeriTile.Bench.TritonBenchG.VectorAddition.Finite.mask32] at *
  all_goals simp [h]

#axiomsClean original_ast_lowering
#axiomsClean source_program_reduction
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.Source
