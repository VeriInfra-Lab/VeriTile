import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedBackend
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Dimension

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def Ready (BM row inner : Nat) (current : Fin BM → Word × Nat) (s : State Word) : Prop :=
  s.regs .int [] "pid_k"=some (fun _ => (inner : Int)) ∧
  s.regs .int [BM] "m_offset"=some (fun i => ((row+i.1.val : Nat) : Int)) ∧
  s.regs .real [BM] "max_values"=some (fun i => (current i.1).1) ∧
  s.regs .int [BM] "argmax_values"=some (fun i => Int.ofNat (current i.1).2)

def loaded (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (i : Fin BM) (j : Fin BN) : Word :=
  if row+i.val<M ∧ cur+j.val<N then (s.mem X ((row+i.val)*N*K+(cur+j.val)*K+inner)).read .real
  else 0xff800000

def next (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Word × Nat) (i : Fin BM) : Word × Nat :=
  let column := cur+TypedReduction.selected positive (loaded s X M N K BM BN row inner cur i)
  let value : Word := if row+i.val<M ∧ column<N then
    (s.mem X ((row+i.val)*N*K+column*K+inner)).read .real else 0xff800000
  if greater value (current i).1 then (value,column) else current i

theorem body_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0<BN)
    (current : Fin BM → Word × Nat) (ready : Ready BM row inner current s) :
    ∃ t, run (backend rest) (body X O M N K BM BN base index64)
      (s.setReg "start_n" .nat [] (fun _ => cur))=some t ∧
      t.mem=s.mem ∧ Ready BM row inner (next s X M N K BM BN row inner cur positive current) t := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [body,rangeBody,TypedSource.dimension,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,ComputeDType.eraseDType,backend,State.setReg,hk,hm,hv,hi,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.indexed,
    VeriTile.Bench.TritonBenchG.TritonArgmax.DimBackend.normalize,
    TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
    ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
    TileShape.discreteIntArgMax,TypedReduction.self_equal,TypedReduction.presence,
    TypedReduction.selected,flag,clean,loaded,next,Ready,update_exact,Option.bind, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul]
  constructor <;> funext i
  all_goals dsimp only [flag,clean,loaded]
  all_goals simp only [apply_ite Prod.fst,apply_ite Prod.snd,apply_ite Int.ofNat,apply_ite (fun n : Nat => (n : Int)),Nat.cast_add]
  all_goals rfl

#axiomsClean body_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
