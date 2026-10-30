import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeArrayBridge

/-! Public array IO for the original FP32 source. All typed raw-array aliases
and every finite launch list remain admitted. The output channel enumerates
store events; overlapping events observe the final last-writer value. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeIO
open VeriTile.Triton FP.Structural NativeFormula
noncomputable section

def inputs (r : Regions) : List ArrayIOChannel :=
  [⟨.real,r.query⟩,⟨.real,r.key⟩,⟨.real,r.value⟩,
   ⟨.int,r.start⟩,⟨.int,r.sequence⟩,⟨.int,r.prompt⟩]

def arrays (r : Regions) (v : ArrayValues α (inputs r)) : Arrays α where
  query := v ⟨0,by simp [inputs]⟩
  key := v ⟨1,by simp [inputs]⟩
  value := v ⟨2,by simp [inputs]⟩
  start := v ⟨3,by simp [inputs]⟩
  sequence := v ⟨4,by simp [inputs]⟩
  prompt := v ⟨5,by simp [inputs]⟩

def eventOperands (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch)
    (v : ArrayValues α (inputs r)) (k : Fin jobs.length) : FPReference.Inputs α :=
  operands p jobs[k] (before p r M dots scale jobs (arrays r v) k.val)

def outputs (p : Layout) (r : Regions) (count : Nat) : List IOChannel :=
  [⟨.real,[count,p.BM,p.D],r.out⟩]

/-- Named typed reads and exact evolving masks/strides, with no scratch.
Source precision, invocation metadata and actual Structural execution are
fixed by the schedule; declared capacities are needed only for access safety. -/
def io (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) : ArrayProgramIO α α where
  inputs := inputs r
  outputs := outputs p r jobs.length
  scratch := []
  buffers := [r.query,r.key,r.value,r.start,r.sequence,r.prompt,r.out]
  reads := fun i v => readFootprint p r M dots scale ⟨i.val,i.isLt⟩ jobs (arrays r v)
  write := by
    intro o
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    exact fun v i => address p (eventOperands p r M dots scale jobs v i.1) i.2
  outputMask := by
    intro o
    have ho : o=(0 : Fin 1) := Fin.eq_zero o
    subst o
    exact fun v i => active p (eventOperands p r M dots scale jobs v i.1) i.2
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => jobs.map (NativeArrayBridge.invocation p r scale M)
  encode := id
  observe := fun _ _ => id

def formula [Inhabited α] (p : Layout) (r : Regions) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (scale : ℝ) (jobs : List Launch) :
    ArrayValues α (inputs r) → ChannelValues α (outputs p r jobs.length) := by
  intro v o
  have ho : o=(0 : Fin 1) := Fin.eq_zero o
  subst o
  change Values α .real [jobs.length,p.BM,p.D]
  exact fun i => finalValue p r M dots scale jobs (arrays r v)
    (address p (eventOperands p r M dots scale jobs v i.1) i.2)

#specNonCircular formula avoiding [FP.Structural.exec,FP.Structural.run,
  NativeArrayBridge.applyWrites,FPReference.inputs,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeIO
