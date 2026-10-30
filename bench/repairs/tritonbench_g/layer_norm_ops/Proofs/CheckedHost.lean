import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PublicFrame

namespace VeriTile.Bench.Repairs.LayerNormOps.Checked
open VeriTile.Triton FP.Structural Admission
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section

def epsilonValue : Admission.ScalarArgument → ℝ
  | .float (some value) => value
  | _ => 0

/-- Pointer regions come from the ordinary Tensor/allocator binding. All
operational shape, stride, branch and launch fields come from checked calls. -/
def forwardParameters (c : ForwardCall) (regions : ForwardParameters) : ForwardParameters :=
  { regions with
    xStride := c.common.columns
    yStride := c.common.columns
    residualStride := if c.residual.present then c.common.columns else 0
    residualOutStride := if c.residual.present then c.common.columns else 0
    columns := c.common.columns
    block := blockSize c.common.columns
    epsilon := epsilonValue c.common.epsilon
    rms := c.common.rms.getD Bool.false
    hasResidual := c.residual.present
    storeResidual := c.residual.present
    hasBias := c.common.bias.present }

def backwardParameters (c : BackwardCall) (regions : BackwardParameters) : BackwardParameters :=
  { regions with
    xStride := c.common.columns
    yStride := if c.recompute.getD Bool.false then c.common.columns else 0
    dyStride := c.common.columns
    dxStride := c.common.columns
    dresStride := if c.dresidual.present then c.common.columns else 0
    dresInStride := 0
    meanStride := if c.common.rms.getD Bool.false then 0 else 1
    rstdStride := 1
    rows := c.common.rows
    columns := c.common.columns
    rowsPerProgram := c.rowWidth
    block := blockSize c.common.columns
    epsilon := epsilonValue c.common.epsilon
    rms := c.common.rms.getD Bool.false
    hasDres := c.dresidual.present
    storeDres := Bool.false
    hasBias := c.common.bias.present
    recompute := c.recompute.getD Bool.false }

/-- Ordinary binding for a composed use of the two existing Python functions:
backward receives forward's returned Z, Mean and Rstd, with the same W/bias and
normalization mode. No statistic value or kernel output equation is a field. -/
structure CompositionBinding (f : ForwardCall) (b : BackwardCall)
    (fr : ForwardParameters) (br : BackwardParameters) : Prop where
  rows : b.common.rows = f.common.rows
  columns : b.common.columns = f.common.columns
  rms : b.common.rms = f.common.rms
  bias : b.common.bias.present = f.common.bias.present
  hasResidual : b.hasResidual.getD Bool.false = f.residual.present
  weight : br.weight = fr.weight
  biasRegion : br.bias = fr.bias
  z : br.x = if f.residual.present then fr.residualOut else fr.x
  mean : br.mean = fr.mean
  rstd : br.rstd = fr.rstd

def forward {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardCall)
    (regions : ForwardParameters) (s : State α) : Except Unit (Option (Host.ForwardReturn α)) :=
  if forwardCheck c then .ok (Host.forward M (forwardParameters c regions) c.common.rows s) else .error ()

def backward {α : Type} [Inhabited α] (M : Algebra α) (ops : Host.Scalar α)
    (c : BackwardCall) (regions : BackwardParameters)
    (pw pb : ReductionPlan.Plan c.programs) (dw db : RegionName) (s : State α) :
    Except Unit (Option (Host.BackwardReturn α)) :=
  if backwardCheck c then .ok (Host.backward M ops (backwardParameters c regions)
    c.programs pw pb dw db (c.hasResidual.getD Bool.false) s) else .error ()

def composed {α : Type} [Inhabited α] (M : Algebra α) (ops : Host.Scalar α)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State α) :
    Except Unit (Option (Host.ForwardReturn α × Host.BackwardReturn α)) := do
  let forwardResult ← forward M f fr s
  match forwardResult with
  | none => return none
  | some first =>
      let backwardResult ← backward M ops b br pw pb dw db first.state
      return backwardResult.map (fun second => (first,second))

theorem composed_admitted {α : Type} [Inhabited α] (M : Algebra α) (ops : Host.Scalar α)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State α) :
    composed M ops f b fr br pw pb dw db s =
      .ok (Host.composed M ops (forwardParameters f fr) (backwardParameters b br)
        f.common.rows b.programs pw pb dw db s) := by
  have fc := (forwardCheck_iff f).mpr hf
  have bc := (backwardCheck_iff b).mpr hb
  simp only [composed, forward, backward, fc, bc, if_true,
    Host.composed_complete, Host.forward_complete, Host.backward_complete]
  simp only [bind, Except.bind, pure, Except.pure, Option.map]
  simp only [binding.hasResidual, forwardParameters]

theorem supported (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br) :
    Scope.Supported (forwardParameters f fr) (backwardParameters b br) f.common.rows b.programs := by
  obtain ⟨hn, hlimit, _, _, _, _, _, _, hforward⟩ := hf.1
  obtain ⟨_, _, _, _, _, _, _, _, hp, hwidth, hbackward, hpartial⟩ := hb
  obtain ⟨cover, blockLimit, power⟩ := blockSize_properties f.common.columns hn hlimit
  have backBound : b.programs * b.rowWidth * f.common.columns + blockSize f.common.columns < 2^31 := by
    simpa only [binding.columns] using hbackward
  have partialBound : b.programs * f.common.columns + blockSize f.common.columns < 2^31 := by
    simpa only [binding.columns] using hpartial
  refine ⟨hn, hlimit, ?_, hp, ?_, binding.columns, binding.rows, ?_, cover, blockLimit, power,
    ⟨rfl,rfl,rfl,rfl⟩, ⟨rfl,rfl,rfl,rfl,rfl⟩, ?_, ?_, binding.bias,
    binding.weight, binding.biasRegion, rfl, rfl, binding.z, binding.mean, binding.rstd,
    ⟨rfl,rfl⟩, ⟨hforward, backBound, partialBound⟩⟩
  · change f.common.rows * f.common.columns < 2147483648
    omega
  · change b.programs * f.common.columns < 2147483648
    omega
  · exact congrArg blockSize binding.columns
  · simpa only [backwardParameters, Admission.rowsPerProgram, Launch.rowsPerProgram, binding.rows] using hwidth
  · exact congrArg (fun value => value.getD Bool.false) binding.rms

theorem caller_frame {α : Type} [Inhabited α] (M : Algebra α) (ops : Host.Scalar α)
    (f : ForwardCall) (b : BackwardCall) (fr : ForwardParameters) (br : BackwardParameters)
    (hf : ForwardContract f) (hb : BackwardContract b) (binding : CompositionBinding f b fr br)
    (pw pb : ReductionPlan.Plan b.programs) (dw db : RegionName) (s : State α)
    (live : List RegionName)
    (fresh : Scope.Allocations (forwardParameters f fr) (backwardParameters b br) dw db live) :
    ∃ result, composed M ops f b fr br pw pb dw db s = .ok (some result) ∧
      (∀ r, r ∈ live → ∀ offset, result.2.state.mem r offset = s.mem r offset) := by
  rw [composed_admitted M ops f b fr br hf hb binding, Host.composed_complete]
  refine ⟨_, rfl, ?_⟩
  intro r hr offset
  obtain ⟨result, run, frame⟩ := Host.caller_frame M ops (forwardParameters f fr) (backwardParameters b br)
    f.common.rows b.programs pw pb dw db s live fresh r offset hr
  rw [Host.composed_complete] at run
  cases run
  exact frame

#axiomsClean composed_admitted
#axiomsClean supported
#axiomsClean caller_frame
end
end VeriTile.Bench.Repairs.LayerNormOps.Checked
