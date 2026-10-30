import bench.tritonbench_g.int_scaled_matmul.Proofs.FiniteExecution
import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordGrid

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteSource
open VeriTile.Triton FP.Structural
open FiniteExecution FiniteIndex
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem unscaled_loads (x : Launch) (s : State Word) (k : Nat) :
    naturalA x Bool.false s k=BlockWord.aValue (blockParameters x) s k ∧
    naturalB x Bool.false s k=BlockWord.bValue (blockParameters x) s k := by
  constructor <;> funext i
  · simp [naturalA,BlockWord.aValue,BlockWord.loaded,BlockWord.aPointer,blockParameters,
      TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
      selectedNaturalRow,naturalRow,BlockWord.rowBlock,BlockWord.firstRow,BlockWord.rowsInGroup,
      BlockWord.blocksM,BlockWord.blocksN,GroupedGrid.row,GroupedGrid.first,GroupedGrid.size,
      Contract.ceil,Nat.mul_comm,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  · simp [naturalB,BlockWord.bValue,BlockWord.loaded,BlockWord.bPointer,blockParameters,
      TileShape.indexToList,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
      selectedNaturalColumn,naturalColumn,BlockWord.columnBlock,BlockWord.firstRow,BlockWord.rowsInGroup,
      BlockWord.blocksM,BlockWord.blocksN,GroupedGrid.col,GroupedGrid.first,GroupedGrid.size,
      Contract.ceil,Nat.mul_comm,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc,and_comm]

theorem scaled_loads (x : Launch) (scale : RegionName) (s : State Word) (k : Nat) :
    naturalA x Bool.true s k=ScaledWord.aValue (scaledParameters x scale) s k ∧
    naturalB x Bool.true s k=ScaledWord.bValue (scaledParameters x scale) s k := by
  have masks (e : Nat) : e<x.K-k*x.config.blockK ↔ k*x.config.blockK+e<x.K := by omega
  constructor <;> funext i
  · simp [naturalA,ScaledWord.aValue,ScaledWord.aOffset,ScaledWord.inputRow,ScaledWord.inputInner,
      scaledParameters,selectedNaturalRow,naturalRow,even,masks,BlockWord.wrap,ScaledWord.wrap,
      ScaledWord.rowBlock,ScaledWord.firstRow,ScaledWord.rowsInGroup,ScaledWord.blocksM,ScaledWord.blocksN,
      GroupedGrid.row,GroupedGrid.first,GroupedGrid.size,Contract.ceil,
      Nat.mul_comm,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
  · simp [naturalB,ScaledWord.bValue,ScaledWord.bOffset,ScaledWord.inputColumn,ScaledWord.inputInner,
      scaledParameters,selectedNaturalColumn,naturalColumn,even,masks,BlockWord.wrap,ScaledWord.wrap,
      ScaledWord.columnBlock,ScaledWord.firstRow,ScaledWord.rowsInGroup,ScaledWord.blocksM,ScaledWord.blocksN,
      GroupedGrid.col,GroupedGrid.first,GroupedGrid.size,Contract.ceil,
      Nat.mul_add,Nat.add_mul,Nat.mul_comm,Nat.mul_left_comm,Nat.mul_assoc,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]

theorem unscaled_accumulation (x : Launch) (s : State Word) (k : Nat) :
    accumulation x Bool.false s k=BlockWord.accumulation (blockParameters x) s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [accumulation,BlockWord.accumulation,ih,(unscaled_loads x s k).1,(unscaled_loads x s k).2]
    rfl

theorem scaled_accumulation (x : Launch) (scale : RegionName) (s : State Word) (k : Nat) :
    accumulation x Bool.true s k=ScaledWord.accumulation (scaledParameters x scale) s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [accumulation,ScaledWord.accumulation,ih,(scaled_loads x scale s k).1,(scaled_loads x scale s k).2]
    rfl

theorem unscaled_written (rest : Algebra Word) (x : Launch) (s : State Word) :
    written x Bool.false s (fun carry i => carry i)=BlockWord.written rest (blockParameters x) s := by
  unfold written BlockWord.written
  congr 1
  funext t i
  simp only [unscaled_accumulation,BlockWord.outputValue]
  have mask : active x s i=BlockWord.active (blockParameters x) s i := rfl
  have address : offset x s i=BlockWord.outputOffset (blockParameters x) s i := by
    change naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N+
      naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val =
      x.N*naturalRow x.M x.N x.config (s.pids 0) i.1.val+1*naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val
    ring
  simp only [mask,address]
  rfl

theorem scaled_written (cfg : FP.Config) (u : Nat → Word → Int) (x : Launch) (scale : RegionName)
    (legal : Contract.ConfigContract x.config) (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) :
    written x Bool.true s (scaledSuffix cfg u x scale s)=TotalScaled.written cfg u (scaledParameters x scale) s := by
  unfold written TotalScaled.written
  congr 1
  funext t i
  have mask : active x s i=ScaledWord.active (scaledParameters x scale) s i := rfl
  have address : offset x s i=ScaledWord.outputOffset (scaledParameters x scale) s i := by
    change naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N+
      naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val =
      naturalColumn x.M x.N x.config (s.pids 0) i.2.1.val*1+naturalRow x.M x.N x.config (s.pids 0) i.1.val*x.N
    ring
  have rw := (coordinates x.M x.N x.K x.config legal env (s.pids 0) i.1.val i.2.1.val hp i.1.isLt i.2.1.isLt).1
  have rb := (coordinate_bounds x.M x.N x.K x.config legal env (s.pids 0) i.1.val i.2.1.val hp i.1.isLt i.2.1.isLt).1
  have ms := (ceil_small x.M x.config.blockM (Contract.positive_tiles x.config legal).1
    (Contract.envelope_fields x.M x.N x.K x.config env).1).2.2.2
  have pointer : (row x.M x.N x.config (s.pids 0) i.1.val).toNat=naturalRow x.M x.N x.config (s.pids 0) i.1.val := by
    rw [rw,toNat (by omega)]
  have rs : ScaledWord.outputRow (scaledParameters x scale) s i*(scaledParameters x scale).scaleRowStride =
      naturalRow x.M x.N x.config (s.pids 0) i.1.val := by
    change naturalRow x.M x.N x.config (s.pids 0) i.1.val*1=_
    omega
  simp only [mask,address,scaledSuffix,TotalScaled.expected,ScaledWord.scaled,scaled_accumulation x scale s,pointer,rs]
  rfl

theorem unscaled_source_memory (rest : Algebra Word) (x : Launch) (s : State Word)
    (bk : 0<x.config.blockK) :
    ∃ final,FP.Structural.exec (BlockWord.backend rest) (BlockWord.source (blockParameters x)) s=some final ∧
      final.mem=(BlockWord.written rest (blockParameters x) s).mem := by
  let p := blockParameters x
  let start := BlockWord.preludeExecution rest p s
  obtain ⟨middle,hm,hi⟩ := BlockWord.loopExecution rest p bk s start.val 0 (by omega)
    (BlockWord.preludeInvariant rest p s)
  let finish := BlockWord.epilogueExecution rest p s middle hi
  refine ⟨finish.val,?_,BlockWord.epilogueMemory rest p s middle hi⟩
  change run (BlockWord.backend rest) (BlockWord.source p).surfaceBody s=some finish.val
  rw [BlockWord.source_decomposition,run_append,run_append,start.property]
  have singleton (t : State Word) : run (BlockWord.backend rest) [.forRange "k" 0 p.inner p.blockInner (BlockWord.body p)] t =
      FP.Structural.range (BlockWord.backend rest) "k" 0 p.inner p.blockInner (BlockWord.body p) t := by simp [run,step]
  simp only [Nat.zero_mul] at hm
  simp only [Option.bind_some,singleton,hm]
  exact finish.property

theorem unscaled_program_source (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config) (env : Contract.IndexEnvelope x.M x.N x.K x.config)
    (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) :
    ∃ actual typed,intProgram x s=some actual ∧
      FP.Structural.exec (BlockWord.backend rest) (BlockWord.source (blockParameters x)) s=some typed ∧
      actual.mem=typed.mem := by
  obtain ⟨typed,he,hm⟩ := unscaled_source_memory rest x s (Contract.positive_tiles x.config legal).2.2.1
  refine ⟨BlockWord.written rest (blockParameters x) s,typed,?_,he,hm.symm⟩
  rw [intProgram,program_eq x legal env Bool.false s hp,unscaled_written]

theorem scaled_program_source (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config)
    (env : Contract.IndexEnvelope x.M x.N x.K x.config) (s : State Word) (hp : s.pids 0 < count x.M x.N x.config) :
    ∃ actual typed,scaledProgram cfg u x scale s=some actual ∧
      FP.Structural.exec (ScaledWord.backend rest cfg u) (ScaledWord.source (scaledParameters x scale)) s=some typed ∧
      actual.mem=typed.mem := by
  obtain ⟨typed,he,hm⟩ := TotalScaled.executionMemory rest cfg u (scaledParameters x scale) s
  refine ⟨TotalScaled.written cfg u (scaledParameters x scale) s,typed,?_,he,hm.symm⟩
  rw [scaledProgram,program_eq x legal env Bool.true s hp,scaled_written cfg u x scale legal env s hp]

#axiomsClean unscaled_program_source
#axiomsClean scaled_program_source
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.FiniteSource
