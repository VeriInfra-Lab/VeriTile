import bench.tritonbench_g.fp4_to_bf16.IO
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Spec

/-! The existing conversion comparison changes the complete CTA order. The
source, typed channels, finite count/PID ABI and byte domain stay fixed. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.ScheduleIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- An actual order of invocations of the selected finite-ABI source. -/
structure Program (h : Input) where
  jobs : List Nat

def Program.source {h : Input} (_p : Program h) : ComputeKernel :=
  Kernels.source (Launch.parameters h)

def Program.io {h : Input} (p : Program h) (rest : Algebra Word) : ProgramIO Word Word :=
  ConversionIO.nativeIO rest h p.jobs

def variant {h : Input} (rest : Algebra Word) (left right : Program h) :
    ProgramIO.Variant (left.io rest) where
  schedule := (right.io rest).schedule
  observe := (left.io rest).observe

/-- Both complete schedules are safe and succeed on the original packed-byte
domain, agree at every BF16 output, and preserve both entire exterior frames.
Unused primitive interpretations remain arbitrary. -/
def Equivalent {h : Input} (left right : Program h) : Prop :=
  ∀ rest : Algebra Word,
    (left.io rest).EquivalentOn (variant rest left right) (ConversionIO.ByteDomain h)

instance {h : Input} : Spec.ProgramSyntax (Program h) where
  Statement := ComputeStmt
  Signature := List IOChannel × List IOChannel
  signature _ := (ConversionIO.inputChannels h,ConversionIO.outputChannels h)
  body p := p.jobs.flatMap (fun _ => p.source.surfaceBody)
  structural := some Equivalent
  -- Syntax steps must retain the actual PID order as well as the source body.
  sameContext := fun left right => left=right

def original (h : Input) (jobs : List Nat) : Program h := ⟨jobs⟩
def candidate (h : Input) (jobs : List Nat) : Program h := ⟨jobs⟩
def assumptions : Spec.Assumptions ComputeStmt := []

end
end VeriTile.Bench.Repairs.Fp4ToBf16.ScheduleIO
