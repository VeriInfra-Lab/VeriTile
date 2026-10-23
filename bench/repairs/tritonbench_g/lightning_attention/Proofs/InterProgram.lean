import bench.repairs.tritonbench_g.lightning_attention.Proofs.InterReverseProgram
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]
theorem inter_parts (p : Parameters) :
    (inter p).surfaceBody=(interForwardParts p).before ++
    [.forRangeDyn "i" (.constNat 0) (.constNat p.NT) (.constNat 1) (interForwardParts p).body] ++
    (interReverseParts p).before ++
    [.forRangeDyn "reverse_i" (.constNat 0) (.constNat p.NT) (.constNat 1) (interReverseParts p).body] := rfl

/-- Both actual nested scans execute from their zero initial carries. Gradient
updates consume the intra stage's current memory, with no supplied-gradient or
successful-execution premise. -/
theorem interProgram (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (hbalance : p.NC*p.CB=p.BT) (input : Memory α) (s : State α) (hin : Inputs s.mem input) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (inter p) s=some t ∧
      t.mem=interMemory M D p (s.pids 0) input s.mem ∧ Inputs t.mem input := by
  let start := interFEntry M D p s
  have hstart := interFEntry_ready M D p s
  have ctx : InterFReady p (s.pids 0) 0 0 input (fun _ => zero M) start.val :=
    {hstart with inputs:=hstart.inputs.trans hin}
  obtain ⟨q,hq,hqc,hqm⟩ := interForwardScan M D p (s.pids 0) input start.val ctx
  let reset := interREntry M D p (s.pids 0) _ _ input _ q hqc
  have hreset := interREntry_ready M D p (s.pids 0) _ _ input _ q hqc
  have hrm := interREntry_memory M D p (s.pids 0) _ _ input _ q hqc
  obtain ⟨t,ht,htc,htm⟩ := interReverseScan M D p hbalance (s.pids 0) input reset.val hreset
  refine ⟨t,?_,?_,htc.inputs⟩
  · change run (engine (withDot M D)) (inter p).surfaceBody s=some t
    rw [inter_parts]
    simp only [run_append,start.property,Option.bind_some,run_single_range _ _ _ _ _ _ hq,
      reset.property,run_single_range _ _ _ _ _ _ ht]
  · rw [htm,hrm,hqm]
    dsimp only [start]
    rw [interFEntry_memory]
    rfl
#axiomsClean interProgram
end
end VeriTile.Bench.Repairs.LightningAttention
