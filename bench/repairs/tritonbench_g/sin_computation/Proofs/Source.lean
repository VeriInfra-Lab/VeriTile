import bench.repairs.tritonbench_g.sin_computation.Proofs.LegacySource
import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract
import bench.tritonbench_g.sin_computation.Proofs.NativeSource
import bench.tritonbench_g.add_example.Proofs.FinitePhysical

/-! Interpret the existing sine AST lowering, including all seven statements.
Only FP32 memory is used by the public wrapper. Undefined inactive loads remain
arbitrary; the matching store mask removes them from observable outputs. -/
namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Source
open VeriTile.Triton
open SinComputationNative
open VeriTile.Bench.TritonBenchG.VectorAddition.DTypeScope
open VeriTile.Bench.TritonBenchG.AddExample.Finite (MemoryPrimitives PhysicalView Write applyWrites fold_source_stores)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Registers (M A : Type) where
  indices : RegName → Nat → BitVec 32
  masks : RegName → Nat → Bool
  values : RegName → Nat → A
  memory : M

def executeInstruction {M A : Type} (ops : MemoryPrimitives M A) (sine : A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat)
    (s : Registers M A) : Instruction → Registers M A
  | .indexAssign name value => {s with indices := fun key lane => if key=name then evalIndex pid lane s.indices value else s.indices key lane}
  | .maskAssign name value => {s with masks := fun key lane => if key=name then evalMask pid lane s.indices s.masks value else s.masks key lane}
  | .load name region offset mask => {s with values := fun key lane => if key=name then if evalMask pid lane s.indices s.masks mask then ops.read s.memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) else undef else s.values key lane}
  | .sinAssign name a => {s with values := fun key lane => if key=name then sine (s.values a lane) else s.values key lane}
  | .store region offset value mask => {s with memory := (List.range 4).foldl (fun memory lane => if evalMask pid lane s.indices s.masks mask then ops.write memory (views region).dtype ((views region).at (evalIndex pid lane s.indices offset)) (s.values value lane) else memory) s.memory}

def executeBody {M A : Type} (ops : MemoryPrimitives M A) (sine : A → A)
    (views : RegionName → PhysicalView) (undef : A) (pid : Nat) :
    List Instruction → Registers M A → Registers M A
  | [],s => s
  | stmt::rest,s => executeBody ops sine views undef pid rest (executeInstruction ops sine views undef pid s stmt)

def runProgram {M A : Type} (ops : MemoryPrimitives M A) (sine : A → A)
    (views : RegionName → PhysicalView) (undef : A) (n pid : Nat) (memory : M) : Option M := do
  let code ← lowerBody (sourceKernel "x_ptr" "output_ptr" n 4).body
  let initial : Registers M A := ⟨fun _ _ => 0,fun _ _ => Bool.false,fun _ _ => undef,memory⟩
  pure (executeBody ops sine views undef pid code initial).memory

def laneAddress (v : PhysicalView) (pid lane : Nat) : Nat := v.at (SinComputationNative.offset pid lane)

def uncheckedWrites {M A : Type} (ops : MemoryPrimitives M A) (sine : A → A)
    (x output : PhysicalView) (n pid : Nat) (s : M) : List (VeriTile.Bench.TritonBenchG.AddExample.Finite.Write A) :=
  (activeTile n pid).map fun lane => ⟨laneAddress output pid lane,
    sine (ops.read s x.dtype (laneAddress x pid lane))⟩

theorem source_program_reduction {M A : Type} (ops : MemoryPrimitives M A) (sine : A → A)
    (views : RegionName → PhysicalView) (undef : A) (n pid : Nat) (s : M) :
    runProgram ops sine views undef n pid s =
      some (applyWrites ops (views "output_ptr").dtype
        (uncheckedWrites ops sine (views "x_ptr") (views "output_ptr") n pid s) s) := by
  simp only [runProgram,sourceKernel,original_ast_lowering,Option.bind_some,bind,pure,
    Option.bind,sourceCode,executeBody,executeInstruction,evalIndex,evalMask]
  simp only [if_pos rfl,if_true,String.reduceEq,if_false]
  congr 1
  unfold uncheckedWrites activeTile
  rw [←fold_source_stores]
  apply congrArg (fun f => List.foldl f s (List.range 4))
  funext memory lane
  change (if SinComputationNative.mask n pid lane then _ else _) = _
  by_cases h : SinComputationNative.mask n pid lane=Bool.true <;>
    simp [h,laneAddress,SinComputationNative.mask,SinComputationNative.offset,
      SinComputationNative.bound,VeriTile.Bench.TritonBenchG.VectorAddition.Native.bound,
      VeriTile.Bench.TritonBenchG.AddExample.Finite.offset32,
      VeriTile.Bench.TritonBenchG.AddExample.Finite.mask32,
      VeriTile.Bench.TritonBenchG.AddExample.Finite.extent32] at *
  all_goals simp [h]

#axiomsClean source_program_reduction
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Source
