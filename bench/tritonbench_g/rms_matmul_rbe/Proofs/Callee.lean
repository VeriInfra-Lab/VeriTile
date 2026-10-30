import bench.tritonbench_g.rms_matmul_rbe.Proofs.SourceComplete
import bench.tritonbench_g.rms_matmul_rbe.Proofs.InputFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def calleeCode (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :=
  if native then selectedSource plan fp8 outputFP8 p else (source fp8 outputFP8 p).surfaceBody

def calleeTarget {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool) (plan : RotaryPlan)
    (fp8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α) :=
  if native then nativeRawTile a plan fp8 p s epsilon theta else rawTile a fp8 p s epsilon theta

def calleeWrite {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool) (plan : RotaryPlan)
    (fp8 outputFP8 : Bool) (p : Arguments) (initial current : State α) (epsilon theta : α) :=
  tileWrite a outputFP8 p (initial.pids 0) (programRow p initial) (programCol p initial)
    (calleeTarget a native plan fp8 p initial epsilon theta) current

theorem callee_complete {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : Int8Memory s) (even : p.N%2=0) :
    ∃ final,run a.model (calleeCode native plan fp8 outputFP8 p) s=some final ∧
      final.mem=(calleeWrite a native plan fp8 outputFP8 p s s epsilon theta).mem ∧
      final.regs .real [] "EPS"=some (fun _ => epsilon) ∧
      final.regs .real [] "THETA"=some (fun _ => theta) ∧ final.pids=s.pids := by
  cases native
  · obtain ⟨final,hx,hm,he,ht,hp⟩ := sourceExecutionFull a fp8 outputFP8 p s epsilon theta he ht bytes
    obtain ⟨reference,hr,hmr⟩ := source_reference a z fp8 outputFP8 p s epsilon theta
      (by assumption) (by assumption) bytes even
    have eq : reference=final := Option.some.inj (hr.symm.trans hx)
    subst reference
    exact ⟨final,hx,hmr,he,ht,hp⟩
  · obtain ⟨final,hx,hm,he,ht,hp⟩ := nativeSourceExecutionFull a plan fp8 outputFP8 p s epsilon theta he ht bytes
    obtain ⟨reference,hr,hmr⟩ := native_reference a z plan fp8 outputFP8 p s epsilon theta
      (by assumption) (by assumption) bytes even
    have eq : reference=final := Option.some.inj (hr.symm.trans hx)
    subst reference
    exact ⟨final,hx,hmr,he,ht,hp⟩

theorem calleeTarget_stable {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 : Bool) (p : Arguments)
    (s t : State α) (h : InputMemory p s t) (hp : s.pids=t.pids) (epsilon theta : α) :
    calleeTarget a native plan fp8 p s epsilon theta=calleeTarget a native plan fp8 p t epsilon theta := by
  cases native
  · exact rawTile_stable a fp8 p s t h hp epsilon theta
  · exact nativeRawTile_stable a plan fp8 p s t h hp epsilon theta

theorem calleeWrite_inputs {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p reader : Arguments)
    (initial current : State α) (epsilon theta : α)
    (apart : reader.X≠p.OUT ∧ reader.W≠p.OUT ∧ reader.RMS≠p.OUT) :
    InputMemory reader (calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta) current :=
  tileWrite_input_frame a outputFP8 p reader _ _ _ _ current apart

theorem calleeWrite_int8 {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (initial current : State α) (epsilon theta : α) (h : Int8Memory current) :
    Int8Memory (calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta) :=
  tileWrite_int8 a outputFP8 p _ _ _ _ current h

theorem calleeWrite_memory {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (initial current replacement : State α) (epsilon theta : α) (h : current.mem=replacement.mem) :
    (calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta).mem=
      (calleeWrite a native plan fp8 outputFP8 p initial replacement epsilon theta).mem :=
  tileWrite_memory a outputFP8 p _ _ _ _ current replacement h

theorem calleeWrite_initial {α : Type} [Inhabited α] (a : Arithmetic α)
    (native : Bool) (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments)
    (initial current : State α) (epsilon theta : α) (h : InputMemory p current initial)
    (hp : current.pids=initial.pids) :
    calleeWrite a native plan fp8 outputFP8 p current current epsilon theta=
      calleeWrite a native plan fp8 outputFP8 p initial current epsilon theta := by
  simp only [calleeWrite,programRow,programCol,hp,
    calleeTarget_stable a native plan fp8 p current initial h hp]

#axiomsClean callee_complete
#axiomsClean calleeTarget_stable
#axiomsClean calleeWrite_initial
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
