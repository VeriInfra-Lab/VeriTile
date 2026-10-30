import bench.tritonbench_g.triton_argmax.Kernels
import bench.tritonbench_g.triton_argmax.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITDimension
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
open Typed
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def arguments {α : Type} (s : State α) (M N K base : Nat) : State α :=
  (((s.setReg "M" .int [] (fun _ => (M : Int))).setReg "N" .int [] (fun _ => (N : Int))).setReg
    "K" .int [] (fun _ => (K : Int))).setReg "program_base" .int [] (fun _ => (base : Int))

def rangeBody (code : List ComputeStmt) : List ComputeStmt :=
  match (code[7]? : Option ComputeStmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

def body (X : RegionName) (O : Region .int) (BM BN : Nat) (index64 : Bool) :=
  rangeBody (JITSource.dimension X O BM BN index64).surfaceBody

def Ready (M N K BM row inner : Nat) (current : Fin BM → Word × Nat) (s : State Word) : Prop :=
  s.regs .int [] "M"=some (fun _ => (M : Int)) ∧
  s.regs .int [] "N"=some (fun _ => (N : Int)) ∧
  s.regs .int [] "K"=some (fun _ => (K : Int)) ∧
  s.regs .int [] "pid_k"=some (fun _ => (inner : Int)) ∧
  s.regs .int [BM] "m_offset"=some (fun i => ((row+i.1.val : Nat) : Int)) ∧
  s.regs .real [BM] "max_values"=some (fun i => (current i.1).1) ∧
  s.regs .int [BM] "argmax_values"=some (fun i => Int.ofNat (current i.1).2)

theorem body_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0<BN)
    (current : Fin BM → Word × Nat) (ready : Ready M N K BM row inner current s) :
    ∃ t, run (backend rest) (body X O BM BN index64)
      (s.setReg "start_n" .int [] (fun _ => (cur : Int)))=some t ∧
      t.mem=s.mem ∧ Ready M N K BM row inner (next s X M N K BM BN row inner cur positive current) t := by
  obtain ⟨hM,hN,hK,hk,hm,hv,hi⟩ := ready
  simp [body,rangeBody,JITSource.dimension,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,backend,hM,hN,hK,hk,hm,hv,hi,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
    TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
    ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
    TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
    TypedReduction.selected,flag,clean,loaded,next,Ready,update_exact,Option.bind, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul]
  all_goals simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, Int.toNat_natCast]
  constructor <;> funext i
  all_goals dsimp only [flag,clean,loaded]
  all_goals simp only [apply_ite Prod.fst,apply_ite Prod.snd]
  all_goals simp [-Nat.cast_ite, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, ← apply_ite (fun n : Nat => (n : Int)), flag, clean, loaded]
  all_goals rfl



#axiomsClean body_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITDimension
