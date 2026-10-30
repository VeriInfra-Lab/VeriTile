import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Bound
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Readback

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

def InputMemory (p : Arguments) (s t : State α) : Prop :=
  s.mem p.X=t.mem p.X ∧ s.mem p.W=t.mem p.W ∧ s.mem p.RMS=t.mem p.RMS

theorem rawRow_stable {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s t : State α) (h : InputMemory p s t) (batch row : Nat) :
    rawRow a fp8 p s batch row=rawRow a fp8 p t batch row := by
  rcases h with ⟨hx,hw,hr⟩
  simp only [rawRow,hx,hw,hr]

theorem rawTile_stable {α : Type} [Inhabited α] (a : Arithmetic α) (fp8 : Bool)
    (p : Arguments) (s t : State α) (h : InputMemory p s t) (hp : s.pids=t.pids)
    (epsilon theta : α) : rawTile a fp8 p s epsilon theta=rawTile a fp8 p t epsilon theta := by
  funext i
  simp only [rawTile,rawValue,programRow,programCol,hp,rawRow_stable a fp8 p s t h]

theorem nativeRawTile_stable {α : Type} [Inhabited α] (a : Arithmetic α) (plan : RotaryPlan)
    (fp8 : Bool) (p : Arguments) (s t : State α) (h : InputMemory p s t) (hp : s.pids=t.pids)
    (epsilon theta : α) :
    nativeRawTile a plan fp8 p s epsilon theta=nativeRawTile a plan fp8 p t epsilon theta := by
  funext i
  simp only [nativeRawTile,nativeRawValue,programRow,programCol,hp,rawRow_stable a fp8 p s t h]

theorem boundTarget_stable {α : Type} [Inhabited α] (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (plan : RotaryPlan)
    (fp8 : Bool) (p : Arguments) (epsilon theta : HostScalar) (s t : State α)
    (h : InputMemory p s t) (hp : s.pids=t.pids) :
    boundTarget a materialize native plan fp8 p epsilon theta s=
      boundTarget a materialize native plan fp8 p epsilon theta t := by
  cases native
  · exact rawTile_stable a fp8 p s t h hp _ _
  · exact nativeRawTile_stable a plan fp8 p s t h hp _ _

theorem tileWrite_input_frame (a : Arithmetic α) (outputFP8 : Bool) (p reader : Arguments)
    (pb pm pn : Nat) (v : Values α .real [16,64]) (s : State α)
    (apart : reader.X≠p.OUT ∧ reader.W≠p.OUT ∧ reader.RMS≠p.OUT) :
    InputMemory reader (tileWrite a outputFP8 p pb pm pn v s) s := by
  constructor
  · funext o
    exact tileWrite_frame a outputFP8 p pb pm pn v s reader.X o (Or.inl apart.1)
  constructor
  · funext o
    exact tileWrite_frame a outputFP8 p pb pm pn v s reader.W o (Or.inl apart.2.1)
  · funext o
    exact tileWrite_frame a outputFP8 p pb pm pn v s reader.RMS o (Or.inl apart.2.2)

theorem InputMemory.trans (p : Arguments) (s t u : State α)
    (h : InputMemory p s t) (k : InputMemory p t u) : InputMemory p s u :=
  ⟨h.1.trans k.1,h.2.1.trans k.2.1,h.2.2.trans k.2.2⟩

#axiomsClean rawRow_stable
#axiomsClean rawTile_stable
#axiomsClean nativeRawTile_stable
#axiomsClean boundTarget_stable
#axiomsClean tileWrite_input_frame
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
