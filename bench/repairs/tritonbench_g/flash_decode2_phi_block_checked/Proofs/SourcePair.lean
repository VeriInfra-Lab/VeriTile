import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.NativeIO
import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.SourcePair
open VeriTile Triton FP.Structural FlashDecode2
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def source (shared : Bool) (H S D Q BD : Nat) : ComputeKernel :=
  if shared then Candidates.phiSharedBase H S D Q BD else phi H S D Q BD

/-- Proof-only view of the two actual address assignments. -/
def addressBlock (shared : Bool) (H S D Q BD : Nat) : List ComputeStmt :=
  ((source shared H S D Q BD).surfaceBody.drop 8).take 2

theorem split (shared : Bool) (H S D Q BD : Nat) :
    (source shared H S D Q BD).surfaceBody =
      (phi H S D Q BD).surfaceBody.take 8 ++ addressBlock shared H S D Q BD ++
        (phi H S D Q BD).surfaceBody.drop 10 := by
  cases shared <;> rfl

theorem address_exec {α : Type} [Inhabited α] (A : Algebra α) (H S D Q BD : Nat) (s : State α) :
    run A (addressBlock Bool.false H S D Q BD) s = run A (addressBlock Bool.true H S D Q BD) s := by
  cases hb : s.regs .int [] "cur_batch" <;>
    cases hh : s.regs .int [] "cur_head" <;>
    cases hd : s.regs .nat [BD] "offs_d" <;>
    simp [addressBlock,source,phi,Candidates.phiSharedBase,FlashDecode2.phiSharedBase,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,State.setReg,numeric,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,hb,hh,hd]
  funext dtype shape name
  split_ifs <;> simp_all [Int.add_mul,Int.mul_assoc]
  subst dtype
  subst shape
  funext k
  simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.add_mul,Int.mul_assoc]

theorem address_safe {α : Type} [Inhabited α] (A : Algebra α) (H S D Q BD : Nat)
    (shared : Bool) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList A bounds (addressBlock shared H S D Q BD) s := by
  cases shared <;>
    simp [addressBlock,source,phi,Candidates.phiSharedBase,FlashDecode2.phiSharedBase,ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeOp_unfold]

theorem source_exec {α : Type} [Inhabited α] (A : Algebra α) (H S D Q BD : Nat) (s : State α) :
    exec A (phi H S D Q BD) s = exec A (Candidates.phiSharedBase H S D Q BD) s := by
  change run A (source Bool.false H S D Q BD).surfaceBody s = run A (source Bool.true H S D Q BD).surfaceBody s
  simp only [split,run_append,address_exec]

theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (H S D Q BD : Nat)
    (s : State α) (bounds : RegionName → Nat) :
    TraceSafe A bounds (phi H S D Q BD) s ↔
      TraceSafe A bounds (Candidates.phiSharedBase H S D Q BD) s := by
  change Safety.traceSafeList A bounds (source Bool.false H S D Q BD).surfaceBody s ↔
    Safety.traceSafeList A bounds (source Bool.true H S D Q BD).surfaceBody s
  simp only [split,Safety.traceSafeList_append,run_append,address_exec,address_safe]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.SourcePair
