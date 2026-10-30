import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourcePrelude

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSetup
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def rowIndex (tileRow : Nat) (i : Fin 32) := i.val + tileRow*32
def columnIndex (tileColumn : Nat) (i : Fin 32) := i.val + tileColumn*32

def prepared {α : Type} (M : Algebra α) (start length columns tileRow tileColumn : Nat)
    (s : State α) : State α :=
  let s := s.setReg "cur_seq_start" .int [] (fun _ => (start : Int))
  let s := s.setReg "offset_m" .int [32] (fun i => (rowIndex tileRow i.1 : Int))
  let s := s.setReg "offset_n" .int [32] (fun i => (columnIndex tileColumn i.1 : Int))
  let s := s.setReg "offset_k" .int [16] (fun i => (i.1.val : Int))
  let s := s.setReg "ram" .int [32] (fun i => (rowIndex tileRow i.1 % length : Int))
  let s := s.setReg "rbn" .int [32] (fun i => (columnIndex tileColumn i.1 % columns : Int))
  s.setReg "accumulator" .real [32,32] (fun _ => M.literal (some .fp32) .real 0)

theorem run_setup {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α)
    (start length tileRow tileColumn : Nat)
    (hstart : (s.mem p.starts p.batch).read .int = (start : Int))
    (hbatch : s.regs .int [] "cur_batch" = some (fun _ => (p.batch : Int)))
    (hm : s.regs .int [] "M" = some (fun _ => (length : Int)))
    (hn : s.regs .int [] "N" = some (fun _ => (p.columns : Int)))
    (hrow : s.regs .int [] "pid_m" = some (fun _ => (tileRow : Int)))
    (hcol : s.regs .int [] "pid_n" = some (fun _ => (tileColumn : Int))) :
    run M (SourceParts.activePrelude abi p) s =
      some (prepared M start length p.columns tileRow tileColumn s) := by
  rw [SourcePrelude.active_common]
  simp [SourceParts.activePrelude,SourceParts.active,SourceParts.afterTile,
    SourceParts.afterLength,SourceParts.thenBody,SourceParts.outer,
    kernel,primaryHalf,sourceFace,sourceHalfSingle323232,ComputeKernel.surfaceBody,
    ScalarFormat.dtype,LowFormat.scalar,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,numericLe,numericEq,
    integralFloorDiv,integralMod,Region.cast,State.setReg,hstart,hbatch,hm,hn,hrow,hcol,
    prepared,rowIndex,columnIndex]
  funext d sh name
  by_cases hd : TileDType.int = d
  · subst d
    by_cases hs : sh = [32]
    · subst sh
      by_cases hrbn : name = "rbn"
      · simp [hrbn]
        funext i
        have hpos : ¬ (i.1.val : Int) + tileColumn*32 < 0 := by omega
        simp [bop,IntegerExpr.sameShape,hpos,←Int.emod_def]
      · by_cases hram : name = "ram"
        · simp [hrbn,hram]
          funext i
          have hpos : ¬ (i.1.val : Int) + tileRow*32 < 0 := by omega
          simp [bop,IntegerExpr.sameShape,hpos,←Int.emod_def]
        · simp [hrbn,hram]
          rfl
    · simp [hs,Ne.symm hs]
  · simp [hd]

#axiomsClean run_setup
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceSetup
