import bench.tritonbench_g.rms_rbe_matmul.Proofs.PublicSourceIO
import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealOutput

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Independent mathematical RMS projection and optional rotary transform. -/
def mathematical (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues ℝ (inputs fp8 B p)) : ChannelValues ℝ (outputs fp8 B p) :=
  Fin.cases (fun j => ofFloat (outputDType fp8)
    (KernelMath.output p.K p.headDim p.start p.rotary
      (KernelReal.logical d8 fp8 (logical fp8 B p xs) j.1.val j.2.1.val)
      j.2.1.val j.2.2.1.val)) (fun k => k.elim0)

theorem formula_real (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (B : Nat) (p : Parameters) (evenN : p.N%2=0) :
    formula (KernelReal.operations d8) fp8 B p=mathematical d8 fp8 B p := by
  funext xs o j
  fin_cases o
  let x := coordinate B p j
  obtain ⟨hrow,hcol⟩ := KernelGrid.owner_coordinate p x
  change (KernelGrid.tile p (KernelGrid.owner p x)).row*16+(KernelGrid.lane x).1.val=x.2.1.val at hrow
  change (KernelGrid.tile p (KernelGrid.owner p x)).column*64+(KernelGrid.lane x).2.1.val=x.2.2.val at hcol
  have real := KernelReal.output_real d8 zero fp8 p (logical fp8 B p xs)
    (KernelGrid.tile p (KernelGrid.owner p x)) (KernelGrid.lane x)
    (by rw [hrow]; exact x.2.1.isLt) (by rw [hcol]; exact x.2.2.isLt) evenN
  rw [hrow,hcol] at real
  exact congrArg (ofFloat (outputDType fp8)) real

theorem real_implements (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (B : Nat) (p : Parameters) (evenN : p.N%2=0) :
    (io (KernelReal.operations d8) fp8 B p).ImplementsOn (domain fp8 B p) (mathematical d8 fp8 B p) := by
  rw [← formula_real d8 zero fp8 B p evenN]
  exact implements (KernelReal.operations d8) fp8 B p

#specNonCircular mathematical avoiding [FP.Structural.exec,KernelSource.inputs,KernelGraph.output,KernelGrid.launch]
#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
