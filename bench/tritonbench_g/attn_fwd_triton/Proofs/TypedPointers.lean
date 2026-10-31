import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedPrefix
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def PointerReady {α : Type} (BD : Nat) (s : State α) : Prop :=
  (∃ v, s.regs .int [] "n_ctx"=some v) ∧
  (∃ v, s.regs .int [] "stride_kn64"=some v) ∧
  (∃ v, s.regs .int [] "stride_vk64"=some v) ∧
  (∃ v, s.regs .int [] "start_m"=some v) ∧
  (∃ v, s.regs .int [128] "offs_m"=some v) ∧
  (∃ v, s.regs .int [64] "offs_n"=some v) ∧
  (∃ v, s.regs .ptr [BD,64] "K_ptrs"=some v) ∧
  (∃ v, s.regs .ptr [] "K_scale_ptr"=some v) ∧
  (∃ v, s.regs .ptr [64,BD] "V_ptrs"=some v) ∧
  (∃ v, s.regs .ptr [128,BD] "O_block_ptr"=some v) ∧
  (∃ v, s.regs .ptr [128,BD] "Q_ptrs"=some v) ∧
  (∃ v, s.regs .ptr [] "Q_scale_ptr"=some v)
def pointerCode (p : Config) (BD : Nat) := ((code p BD).drop 13).take 9

theorem pointers_success {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (s : State α) (hs : CoordinateReady s) :
    ∃ t, run (engine A) (pointerCode p BD) s=some t ∧ PointerReady BD t := by
  rcases hs with ⟨⟨x0,h0⟩,⟨x1,h1⟩,⟨x2,h2⟩,⟨x3,h3⟩,⟨x4,h4⟩,⟨x5,h5⟩,⟨x6,h6⟩,⟨x7,h7⟩,⟨x8,h8⟩,⟨x9,h9⟩,⟨x10,h10⟩,⟨x11,h11⟩,⟨x12,h12⟩⟩
  simp only [pointerCode,code,source,Bool.false_eq_true,↓reduceIte,
    TypedSource.repaired,ComputeKernel.surfaceBody,ControlView.compute,List.drop_succ_cons,List.drop_zero,
    List.take_succ_cons,List.take_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,PointerReady,
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution

