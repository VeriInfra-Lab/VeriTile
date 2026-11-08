import bench.tritonbench_g.matmul_persistent_triton.Proofs.Support.Model
namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem update_congr {s t : State α} {name : RegName} {d : TileDType} {sh : TileShape}
    {v w : Values α d sh} (states : s=t) (values : v=w) : s.setReg name d sh v=t.setReg name d sh w := by
  cases states; cases values; rfl

def intCeil (n d : Nat) : Int := (n : Int)/d + if (n : Int)%d=0 then 0 else 1

theorem intCeil_nonneg (n d : Nat) : 0 ≤ intCeil n d := by
  unfold intCeil
  apply add_nonneg
  · exact Int.ediv_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
  · split_ifs <;> omega

def preludeState (p : Parameters) (s : State α) : State α :=
  let rows := intCeil p.problem.M 128
  let columns := intCeil p.problem.N 256
  let count := intCeil p.problem.K p.problem.profile.blockK
  let work := (rows*columns)/p.problem.SMS + if (s.pids 0 : Int) < (rows*columns)%p.problem.SMS then 1 else 0
  let group := if (8 : Int)<rows then 8 else rows
  let t := s.setReg "start_pid" .int [] (fun _ => (s.pids 0 : Int))
  let t := t.setReg "M" .int [] (fun _ => (p.problem.M : Int))
  let t := t.setReg "N" .int [] (fun _ => (p.problem.N : Int))
  let t := t.setReg "K" .int [] (fun _ => (p.problem.K : Int))
  let t := t.setReg "stride_am" .int [] (fun _ => (p.am : Int))
  let t := t.setReg "stride_ak" .int [] (fun _ => (p.ak : Int))
  let t := t.setReg "stride_bk" .int [] (fun _ => (p.bk : Int))
  let t := t.setReg "stride_bn" .int [] (fun _ => (p.bn : Int))
  let t := t.setReg "stride_cm" .int [] (fun _ => (p.cm : Int))
  let t := t.setReg "stride_cn" .int [] (fun _ => (p.cn : Int))
  let t := t.setReg "num_pid_m" .int [] (fun _ => (rows : Int))
  let t := t.setReg "num_pid_n" .int [] (fun _ => (columns : Int))
  let t := t.setReg "k_tiles" .int [] (fun _ => (count : Int))
  let t := t.setReg "num_tiles" .int [] (fun _ => (rows*columns : Int))
  let t := t.setReg "tiles_per_SM" .int [] (fun _ => (work : Int))
  let t := t.setReg "active_group_m" .int [] (fun _ => (group : Int))
  let t := t.setReg "num_pid_in_group" .int [] (fun _ => (group*columns : Int))
  let t := t.setReg "lanes_m" .int [128] (fun i => (i.1.val : Int))
  let t := t.setReg "lanes_n" .int [256] (fun i => (i.1.val : Int))
  t.setReg "offs_k_for_mask" .int [p.problem.profile.blockK] (fun i => (i.1.val : Int))

def preludeExecution (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    {t : State α // run (backend P) (prelude p) s=some t} := by
  refine ⟨preludeState p s,?_⟩
  have hn : ¬ intCeil p.problem.M 128 * intCeil p.problem.N 256 < 0 :=
    not_lt_of_ge (mul_nonneg (intCeil_nonneg _ _) (intCeil_nonneg _ _))
  dsimp only [intCeil] at hn
  simp only [← Int.dvd_iff_emod_eq_zero] at hn
  norm_num only at hn
  rw [prelude_raw]
  simp [rawPrelude,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,TypedSource.originalKernel,
    TypedSource.candidateKernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,backend,Broadcast.leftIndex,Broadcast.rightIndex,
    preludeState,intCeil,FP.Structural.bop,Int.not_ofNat_neg,← Int.emod_def,hn]
  repeat' apply update_congr
  all_goals first
  | rfl
  | funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,hn,← Int.emod_def,ite_mul]

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
