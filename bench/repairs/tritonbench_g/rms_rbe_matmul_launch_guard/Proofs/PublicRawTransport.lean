import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicSourceIO

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
variable {α : Type} [Inhabited α]

/-- Extract logical channels from the arbitrary raw initial state used by the
previous public endpoints. No exterior cell or register is changed. -/
def channels (fp8 : Bool) (B : Nat) (p : Parameters) (s : State α) :
    ChannelValues α (inputs fp8 B p) := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact (s.mem "X" (j.1.val*(p.M*p.K)+j.2.1.val*p.K+j.2.2.1.val)).read .real
  · intro j; exact (s.mem "W" (j.2.1.val+j.1.val*p.K)).read (if fp8 then .int else .fp16)
  · intro j; exact (s.mem "RMS" j.1.val).read (if fp8 then .int else .fp16)

theorem raw_binding (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters) (s : State α) :
    (io a fp8 B p).InputBinding (channels fp8 B p s) s := by
  intro port j active
  fin_cases port <;> cases fp8 <;> rfl

theorem raw_domain (fp8 : Bool) (B : Nat) (p : Parameters) (s : State α)
    (bytes : ByteMemory s) : domain fp8 B p (channels fp8 B p s) := by
  cases fp8
  · trivial
  · exact ⟨fun j => bytes "W" (.inl rfl) _,fun j => bytes "RMS" (.inr rfl) _⟩

/-- The shared IO and retained arbitrary-state endpoints execute the same
complete source schedule, including all original programs and the empty case. -/
theorem raw_schedule (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters) (s : State α) :
    runSchedule ((io a fp8 B p).schedule (channels fp8 B p s)) s=KernelGrid.launch a fp8 B p s :=
  schedule_run a fp8 p (KernelGrid.programs B p) s

#axiomsClean raw_binding
#axiomsClean raw_domain
#axiomsClean raw_schedule
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
