import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicGraphCongruence
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicInputBounds

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem finite_binding (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (state : State α)
    (binding : (io a fp8 B p).InputBinding xs state) :
    FiniteAgreement fp8 B p (KernelSource.inputs p state) (logical fp8 B p xs) := by
  constructor
  · intro b m k hb hm hk
    have h := binding (0 : Fin 3) (⟨b,hb⟩,⟨m,hm⟩,⟨k,hk⟩,PUnit.unit) trivial
    change (state.mem "X" (b*(p.M*p.K)+m*p.K+k)).read .real=xs (0 : Fin 3)
      (⟨b,hb⟩,⟨m,hm⟩,⟨k,hk⟩,PUnit.unit) at h
    cases fp8 <;> simpa [KernelSource.inputs,logical,tensor3,hb,hm,hk] using h
  · intro hf n k hn hk
    subst fp8
    have h := binding (1 : Fin 3) (⟨n,hn⟩,⟨k,hk⟩,PUnit.unit) trivial
    change (state.mem "W" (k+n*p.K)).read .fp16=xs (1 : Fin 3) (⟨n,hn⟩,⟨k,hk⟩,PUnit.unit) at h
    simpa [KernelSource.inputs,logical,tensor2,hn,hk] using h
  · intro hf k hk
    subst fp8
    have h := binding (2 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
    change (state.mem "RMS" k).read .fp16=xs (2 : Fin 3) (⟨k,hk⟩,PUnit.unit) at h
    simpa [KernelSource.inputs,logical,tensor1,hk] using h
  · intro hf n k hn hk
    subst fp8
    have h := binding (1 : Fin 3) (⟨n,hn⟩,⟨k,hk⟩,PUnit.unit) trivial
    change (state.mem "W" (k+n*p.K)).read .int=xs (1 : Fin 3) (⟨n,hn⟩,⟨k,hk⟩,PUnit.unit) at h
    simpa [KernelSource.inputs,logical,tensor2,hn,hk] using h
  · intro hf k hk
    subst fp8
    have h := binding (2 : Fin 3) (⟨k,hk⟩,PUnit.unit) trivial
    change (state.mem "RMS" k).read .int=xs (2 : Fin 3) (⟨k,hk⟩,PUnit.unit) at h
    simpa [KernelSource.inputs,logical,tensor1,hk] using h

/-- Original grid success, typed logical outputs and complete cell frame,
derived from finite input binding and the same already-proved execution. -/
theorem values_frame (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (xs : ChannelValues α (inputs fp8 B p)) (admitted : domain fp8 B p xs)
    (state : State α) (binding : (io a fp8 B p).InputBinding xs state) :
    ∃ final, runSchedule ((io a fp8 B p).schedule xs) state=some final ∧
      (io a fp8 B p).OutputBinding xs (formula a fp8 B p xs) final ∧
      (io a fp8 B p).Frame xs state final := by
  obtain ⟨final,ran,effect⟩ := KernelGrid.complete_reads a fp8 B p state
    (fun q member cur _ => program_bytes a fp8 B p xs admitted state binding q member cur)
  refine ⟨final,?_,?_,?_⟩
  · change runSchedule ((KernelGrid.programs B p).map (invocation a fp8 p)) state=some final
    rw [schedule_run]
    exact ran
  · intro o j active
    fin_cases o
    let x := coordinate B p j
    have out := effect.output (KernelGrid.owner p x) (KernelGrid.owner_mem p x)
      (KernelGrid.lane x) (KernelGrid.owner_active p x)
    rw [KernelGrid.owner_offset] at out
    unfold KernelGrid.result at out
    have graph := graph_output_congr a fp8 B p _ _ (finite_binding a fp8 B p xs state binding)
      (KernelGrid.tile p (KernelGrid.owner p x)) x.1.isLt (by have := x.2.1.isLt; omega)
      (by have := x.2.2.isLt; omega)
    rw [graph] at out
    change mapCell id (final.mem "OUT" (KernelGrid.flat p x))=
      Cell.mk (outputDType fp8).toTileDType (ofFloat (outputDType fp8)
        (KernelGraph.output a fp8 p (logical fp8 B p xs)
          (KernelGrid.tile p (KernelGrid.owner p x)) (KernelGrid.lane x)))
    rw [out]
    cases fp8 <;> rfl
  · intro r off outside
    apply effect.frame r off
    by_cases different : r≠"OUT"
    · exact Or.inl different
    · right
      intro q member i active
      obtain ⟨x,rfl,rfl⟩ := KernelGrid.covers p q member i active
      rw [KernelGrid.owner_offset]
      have h := outside.1 (0 : Fin 1) (x.1,x.2.1,x.2.2,PUnit.unit) trivial
      exact h.resolve_left different

#axiomsClean finite_binding
#axiomsClean values_frame
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
