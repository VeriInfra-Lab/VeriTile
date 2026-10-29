import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourcePrelude

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGuards
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def guarded (abi : SourceABI) (p : KernelArgs) (code : List ComputeStmt) : List ComputeStmt :=
  [.ifThen (SourceParts.condition (SourceParts.outer abi p) 16)
    [.ifThen (SourceParts.condition (SourceParts.afterLength abi p) 0)
      ((SourceParts.afterTile abi p).take 1 ++
        [.ifThen (SourceParts.condition (SourceParts.afterTile abi p) 1) code])]]

theorem common (abi : SourceABI) (p : KernelArgs) (code : List ComputeStmt) :
    guarded abi p code = guarded primaryHalf p code := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem outer_split (abi : SourceABI) (p : KernelArgs) :
    SourceParts.outer abi p = SourceParts.prelude abi p ++ guarded abi p (SourceParts.active abi p) := by
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases weight <;> cases output <;> cases starts <;> cases lengths <;> cases adapters <;> rfl

theorem run_guarded {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (code : List ComputeStmt) (s : State α)
    (length adapter : Int) (tileRow : Nat)
    (hm : s.regs .int [] "M" = some (fun _ => length))
    (hrow : s.regs .int [] "pid_m" = some (fun _ => (tileRow : Int)))
    (hbatch : s.regs .int [] "cur_batch" = some (fun _ => (p.batch : Int)))
    (hadapter : (s.mem p.adapters p.batch).read .int = adapter) :
    run M (guarded abi p code) s =
      if length ≤ 0 then some s else
      if length ≤ (tileRow : Int)*32 then some s else
      let t := s.setReg "lora_index" .int [] (fun _ => adapter)
      if adapter = -1 then some t else run M code t := by
  rw [common]
  by_cases hl : length ≤ 0 <;>
    by_cases ht : length ≤ (tileRow : Int)*32 <;>
    by_cases ha : adapter = -1 <;>
    simp [guarded,SourceParts.condition,SourceParts.afterTile,SourceParts.afterLength,
      SourceParts.thenBody,SourceParts.outer,kernel,primaryHalf,sourceFace,sourceHalfSingle323232,
      ScalarFormat.dtype,LowFormat.scalar,ComputeKernel.surfaceBody,run,step,evalOp_unfold,
      evalExpr,evalComputeOp,ComputeDType.eraseDType,numeric,bop,
      numericLe,numericLt,numericEq,Region.cast,State.setReg,hm,hrow,hbatch,hadapter,
      hl,ht,ha]

#axiomsClean common
#axiomsClean outer_split
#axiomsClean run_guarded
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceGuards
