import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicInputBinding
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicKernelSafety

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem effect_binding (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (s t : State α) (qs : List KernelGrid.Program)
    (effect : KernelGrid.Effect a fp8 p qs s t) (binding : (io a fp8 B p).InputBinding xs s) :
    (io a fp8 B p).InputBinding xs t := by
  intro port j active
  have cell : t.mem (inputs fp8 B p)[port].buf (read fp8 B p port xs j)=
      s.mem (inputs fp8 B p)[port].buf (read fp8 B p port xs j) := by
    fin_cases port
    · exact effect.frame "X" _ (Or.inl (by decide))
    · exact effect.frame "W" _ (Or.inl (by decide))
    · exact effect.frame "RMS" _ (Or.inl (by decide))
  change (t.mem (inputs fp8 B p)[port].buf (read fp8 B p port xs j)).read _ = _
  rw [cell]
  exact binding port j active

theorem programs_safe (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (admitted : domain fp8 B p xs)
    (bounds : RegionName → Nat) (bounded : (io a fp8 B p).InBounds bounds xs)
    (qs : List KernelGrid.Program) (covered : ∀ q, q∈qs → q∈KernelGrid.programs B p)
    (state : State α) (binding : (io a fp8 B p).InputBinding xs state) :
    ScheduleSafe bounds (qs.map (invocation a fp8 p)) state := by
  induction qs generalizing state with
  | nil => trivial
  | cons q qs ih =>
    have member := covered q (by simp)
    have bytes := fun cur (_ : cur<steps p) => program_bytes a fp8 B p xs admitted state binding q member cur
    obtain ⟨t,ran,effect⟩ := KernelGrid.one_complete_reads a fp8 p state q bytes
    constructor
    · change TraceSafe (engine a) bounds (kernel fp8 p) (KernelGrid.atProgram state q)
      exact kernel_safe a fp8 p bounds _ bytes (program_bounds a fp8 B p xs bounds bounded state q member)
    · intro u hu
      change FP.Structural.exec (engine a) (kernel fp8 p) (KernelGrid.atProgram state q)=some u at hu
      rw [ran] at hu
      cases hu
      exact ih (fun r hr => covered r (List.mem_cons_of_mem q hr)) t
        (effect_binding a fp8 B p xs state t [q] effect binding)

/-- The shared source IO judgment for the complete original launch. Neither
execution, source access safety nor agreement with output values is assumed. -/
theorem implements (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters) :
    (io a fp8 B p).ImplementsOn (domain fp8 B p) (formula a fp8 B p) := by
  intro xs admitted bounds bounded state binding
  exact ⟨programs_safe a fp8 B p xs admitted bounds bounded (KernelGrid.programs B p)
    (fun _ member => member) state binding,values_frame a fp8 B p xs admitted state binding⟩

#axiomsClean implements
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
