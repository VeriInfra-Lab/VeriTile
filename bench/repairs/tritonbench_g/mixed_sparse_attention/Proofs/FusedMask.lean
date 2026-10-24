import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOGrid
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (c : Config) : ComputeKernel :=
  match c.storage with
  | .fp16=>TypedSource.FP16.fusedMask c.q c.k c.v c.seq c.scale c.blocks c.offsets c.counts c.columns c.out
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svn c.svk
    c.soz c.soh c.som c.sok c.Z c.H c.T c.NR c.NS c.NV c.BM c.BN c.BD
  | .bf16=>TypedSource.BF16.fusedMask c.q c.k c.v c.seq c.scale c.blocks c.offsets c.counts c.columns c.out
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svn c.svk
    c.soz c.soh c.som c.sok c.Z c.H c.T c.NR c.NS c.NV c.BM c.BN c.BD

def active (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel c).surfaceBody
  match code[3]? with
  | some (ComputeStmt.ifThen _ b)=>b
  | _=>[]

def column (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := active c
  match code[23]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b)=>b
  | _=>[]

theorem overwrite (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (u v : Values α d sh) : (s.setReg n d sh u).setReg n d sh v=s.setReg n d sh v := by
  simp only [State.setReg]
  congr 1
  funext d' sh' n'
  split <;> simp_all

theorem local_run (M : Algebra α) (c : Config) (s : State α) :
    run M ((columnBody c).drop 5 |>.take 2) s = run M ((column c).drop 5 |>.take 1) s := by
  cases hs : c.storage
  all_goals
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,hs,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType]
  all_goals
    cases hm : s.regs .bool [c.BM,1] "m_mask" <;> cases hn : s.regs .bool [c.BN] "n_mask" <;> simp [hm,hn,Option.bind,State.setReg,overwrite]
  all_goals funext d sh n; split <;> simp_all

end
end VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
