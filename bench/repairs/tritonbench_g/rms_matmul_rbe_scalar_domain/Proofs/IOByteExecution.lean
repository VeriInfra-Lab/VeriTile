import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.IOByteNativeIteration
namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def code (native fp8 outputFP8 : Bool) (p : Arguments) :=
  if native then selectedBody fp8 outputFP8 p else body fp8 outputFP8 p

def values (native : Bool) (a : Arithmetic α) (fp8 : Bool) (p : Arguments)
    (s : State α) (v : LoopValues α) (t : Nat) : LoopValues α :=
  if native then nativeNextValues a fp8 p s v t else nextValues a fp8 p s v t

def iteration (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    {next : State α // run a.model (code native fp8 outputFP8 p)
      (s.setReg "step" .nat [] (fun _=>t))=some next} := by
  cases native
  · exact sourceIteration a fp8 outputFP8 p s v h bytes regions t
  · exact nativeIteration a fp8 outputFP8 p s v h bytes regions t

theorem iteration_values (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    LoopRegisters (values native a fp8 p s v t)
      (iteration a native fp8 outputFP8 p s v h bytes regions t).val := by
  have zeroScientific : (0.0:ℝ)=0 := by norm_num
  cases native <;> cases fp8 <;> cases outputFP8
  all_goals
    constructor <;>
      simp [iteration,sourceIteration,nativeIteration,values,nextValues,nativeNextValues,xTile,rmsTile,weightTile,scaledTile,liveKey,
        kernelCast,binary,FP.Structural.bop,h.keys,ofFloat,toFloat,
        Broadcast.leftIndex,Broadcast.rightIndex,Fin.eq_zero,zeroScientific]
    all_goals try rfl

theorem iteration_frame (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    (iteration a native fp8 outputFP8 p s v h bytes regions t).val.mem=s.mem := by
  cases native <;> cases fp8 <;> cases outputFP8 <;> rfl

theorem iteration_header (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    HeaderPreserved s (iteration a native fp8 outputFP8 p s v h bytes regions t).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals cases native <;> cases fp8 <;> cases outputFP8 <;> simp [iteration,sourceIteration,nativeIteration]

theorem iteration_pids (a : Arithmetic α) (native fp8 outputFP8 : Bool) (p : Arguments) (s : State α)
    (v : LoopValues α) (h : LoopRegisters v s) (bytes : InputBytes fp8 p s)
    (regions : PointerRegions p v) (t : Nat) :
    (iteration a native fp8 outputFP8 p s v h bytes regions t).val.pids=s.pids := by
  cases native <;> cases fp8 <;> cases outputFP8 <;> rfl

#axiomsClean iteration_values
#axiomsClean iteration_frame
#axiomsClean iteration_header
#axiomsClean iteration_pids
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.IOBytes
