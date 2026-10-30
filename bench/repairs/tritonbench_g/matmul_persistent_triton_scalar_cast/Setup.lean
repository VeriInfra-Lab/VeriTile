import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tile
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem min_lt_form_int (a b : Int) : min a b = if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (le_of_lt h)]
  · rw [if_neg h,min_eq_right (le_of_not_gt h)]

theorem tile_rows (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s)
    (ht : p.problem.tile pid ordinal<p.problem.tiles) :
    (tilePreludeExecution P p pid ordinal s h).val.regs .int [128] "offs_cm"=
      some (fun i => (p.problem.row (p.problem.tile pid ordinal) i.1.val : Int)) := by
  have hc := (coordinates p.problem (p.problem.tile pid ordinal) ht).1
  simp only [Problem.tile,Schedule.tileId,Int.natCast_add,Int.natCast_mul] at hc
  have hn : ¬ (pid : Int)+(ordinal : Int)*p.problem.SMS<0 := by
    apply not_lt_of_ge
    positivity
  simp [tilePreludeExecution,hn,IntegerExpr.sameShape,← Int.emod_def,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ← min_lt_form_int,hc,Problem.row]
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,← Int.emod_def,hc,Problem.tile,Schedule.tileId]

theorem tile_columns (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s)
    (ht : p.problem.tile pid ordinal<p.problem.tiles) :
    (tilePreludeExecution P p pid ordinal s h).val.regs .int [256] "offs_cn"=
      some (fun i => (p.problem.column (p.problem.tile pid ordinal) i.1.val : Int)) := by
  have hc := (coordinates p.problem (p.problem.tile pid ordinal) ht).2
  simp only [Problem.tile,Schedule.tileId,Int.natCast_add,Int.natCast_mul] at hc
  have hn : ¬ (pid : Int)+(ordinal : Int)*p.problem.SMS<0 := by
    apply not_lt_of_ge
    positivity
  have hg : ¬ (pid : Int)+(ordinal : Int)*p.problem.SMS <
      ((p.problem.group : Int)*p.problem.columns)*
        (((pid : Int)+(ordinal : Int)*p.problem.SMS)/((p.problem.group : Int)*p.problem.columns)) := by
    have hnat : p.problem.group*p.problem.columns*((pid+ordinal*p.problem.SMS)/(p.problem.group*p.problem.columns))≤
        pid+ordinal*p.problem.SMS := by
      simpa only [Nat.mul_comm] using Nat.div_mul_le_self (pid+ordinal*p.problem.SMS) (p.problem.group*p.problem.columns)
    exact_mod_cast (not_lt_of_ge hnat)
  simp [tilePreludeExecution,hn,hg,IntegerExpr.sameShape,← Int.emod_def,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,← min_lt_form_int,hc,Problem.column]
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,← Int.emod_def,hc,Problem.tile,Schedule.tileId]

theorem tile_clamped_rows (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s)
    (ht : p.problem.tile pid ordinal<p.problem.tiles) :
    (tilePreludeExecution P p pid ordinal s h).val.regs .int [128] "offs_am"=
      some (fun i => (clamp p.problem.M (p.problem.row (p.problem.tile pid ordinal) i.1.val) : Int)) := by
  have relation : (tilePreludeExecution P p pid ordinal s h).val.regs .int [128] "offs_am"=
      ((tilePreludeExecution P p pid ordinal s h).val.regs .int [128] "offs_cm").map
        (fun row i => if row i<(p.problem.M : Int) then row i else 0) := by
    simp [tilePreludeExecution,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.sameShape]
  rw [relation,tile_rows P p pid ordinal s h ht]
  simp [clamp]

theorem tile_clamped_columns (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s)
    (ht : p.problem.tile pid ordinal<p.problem.tiles) :
    (tilePreludeExecution P p pid ordinal s h).val.regs .int [256] "offs_bn"=
      some (fun i => (clamp p.problem.N (p.problem.column (p.problem.tile pid ordinal) i.1.val) : Int)) := by
  have relation : (tilePreludeExecution P p pid ordinal s h).val.regs .int [256] "offs_bn"=
      ((tilePreludeExecution P p pid ordinal s h).val.regs .int [256] "offs_cn").map
        (fun column i => if column i<(p.problem.N : Int) then column i else 0) := by
    simp [tilePreludeExecution,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.sameShape]
  rw [relation,tile_columns P p pid ordinal s h ht]
  simp [clamp]

theorem tile_registers (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s) :
    Registers p pid (tilePreludeExecution P p pid ordinal s h).val := by
  constructor <;> simp [tilePreludeExecution,h.start,h.rows,h.columns,h.count,h.work,
    h.group,h.span,h.lanesM,h.lanesN,h.lanesK,h.dimM,h.dimN,h.dimK,h.strideAM,h.strideAK,h.strideBK,h.strideBN,h.strideCM,h.strideCN]

theorem tile_initial (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s)
    (ht : p.problem.tile pid ordinal<p.problem.tiles) :
    InnerInvariant P p s (p.problem.tile pid ordinal) 0 (tilePreludeExecution P p pid ordinal s h).val := by
  refine ⟨?_,tile_rows P p pid ordinal s h ht,tile_columns P p pid ordinal s h ht,
    tile_clamped_rows P p pid ordinal s h ht,tile_clamped_columns P p pid ordinal s h ht,?_,?_, ?_ , ?_ , ?_ , ?_ , ?_ , ?_ , ?_ , ?_ , ?_⟩
  · simp [tilePreludeExecution]
  · exact (tile_registers P p pid ordinal s h).lanesK
  · simp [tilePreludeExecution,accumulation]
  · simpa [tilePreludeExecution] using h.dimM
  · simpa [tilePreludeExecution] using h.dimN
  · simpa [tilePreludeExecution] using h.dimK
  · simpa [tilePreludeExecution] using h.strideAM
  · simpa [tilePreludeExecution] using h.strideAK
  · simpa [tilePreludeExecution] using h.strideBK
  · simpa [tilePreludeExecution] using h.strideBN
  · simpa [tilePreludeExecution] using h.strideCM
  · simpa [tilePreludeExecution] using h.strideCN

#axiomsClean tile_initial
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
