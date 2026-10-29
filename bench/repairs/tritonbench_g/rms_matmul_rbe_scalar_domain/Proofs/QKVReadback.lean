import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.QKV

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

theorem calleeWrite_other {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (initial current : State α) (epsilon theta : α) (region : RegionName) (offset : Nat)
    (apart : region≠p.OUT) :
    (calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta).mem region offset=
      current.mem region offset :=
  tileWrite_frame a outputFP8 p _ _ _ _ current region offset (Or.inl apart)

theorem calleeWrite_readback {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (initial current : State α) (epsilon theta : α)
    (denseRow : p.som=p.N) (denseCol : p.son=1) (i : TileIndex [16,64])
    (active : outputActive p (programRow p initial) (programCol p initial) i) :
    (calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta).mem p.OUT
        (outputOffset p (initial.pids 0) (programRow p initial) (programCol p initial) i)=
      outputCell a outputFP8 (calleeTarget a native plan fp8 p initial epsilon theta i) :=
  tileWrite_readback a outputFP8 p _ _ _ _ current denseRow denseCol i active

theorem qkv_readback {α : Type} [Inhabited α] (a : Arithmetic α)
    (native fp8 : Bool) (p : Fin 3 → Arguments) (s : State α) (epsilon theta : α)
    (outputs : Function.Injective (fun phase => (p phase).OUT))
    (dense : ∀ phase,(p phase).som=(p phase).N ∧ (p phase).son=1)
    (phase : Fin 3) (i : TileIndex [16,64])
    (active : outputActive (qkvArgument p phase) (programRow (qkvArgument p phase) s)
      (programCol (qkvArgument p phase) s) i) :
    (qkvWrite a native fp8 p s epsilon theta).mem (p phase).OUT
      (outputOffset (qkvArgument p phase) (s.pids 0) (programRow (qkvArgument p phase) s)
        (programCol (qkvArgument p phase) s) i)=
      outputCell a (qkvOutputFP8 fp8 phase)
        (calleeTarget a native (selectedRotaryPlan phase) fp8 (qkvArgument p phase) s epsilon theta i) := by
  have h01 : (qkvArgument p 0).OUT≠(qkvArgument p 1).OUT := by
    intro h
    have bad : (0:Fin 3)=1 := outputs h
    contradiction
  have h02 : (qkvArgument p 0).OUT≠(qkvArgument p 2).OUT := by
    intro h
    have bad : (0:Fin 3)=2 := outputs h
    contradiction
  have h12 : (qkvArgument p 1).OUT≠(qkvArgument p 2).OUT := by
    intro h
    have bad : (1:Fin 3)=2 := outputs h
    contradiction
  let qw := calleeWrite a native (selectedRotaryPlan 0) fp8 Bool.false
    (qkvArgument p 0) s s epsilon theta
  let kw := calleeWrite a native (selectedRotaryPlan 1) fp8 fp8
    (qkvArgument p 1) s qw epsilon theta
  fin_cases phase
  · exact (calleeWrite_other a native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2)
      s kw epsilon theta (p 0).OUT _ h02).trans
      ((calleeWrite_other a native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1)
        s qw epsilon theta (p 0).OUT _ h01).trans
        (calleeWrite_readback a native (selectedRotaryPlan 0) fp8 Bool.false (qkvArgument p 0)
          s s epsilon theta (dense 0).1 (dense 0).2 i active))
  · exact (calleeWrite_other a native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2)
      s kw epsilon theta (p 1).OUT _ h12).trans
      (calleeWrite_readback a native (selectedRotaryPlan 1) fp8 fp8 (qkvArgument p 1)
        s qw epsilon theta (dense 1).1 (dense 1).2 i active)
  · exact calleeWrite_readback a native (selectedRotaryPlan 2) fp8 fp8 (qkvArgument p 2)
      s kw epsilon theta (dense 2).1 (dense 2).2 i active

#axiomsClean calleeWrite_readback
#axiomsClean qkv_readback
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
