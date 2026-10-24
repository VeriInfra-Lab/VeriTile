import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.ProjectedIO
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NativeDimensionSafety
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeDimensionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000

/-- Source-independent first-NaN/first-maximum index of a logical row. -/
def formula (c : DimConfig) (nonempty : 0 < c.extent)
    (input : Nat → Nat → Nat → Word) (row inner : Nat) : Int :=
  Int.ofNat (reference c.extent nonempty (fun j => input row inner j.val)).val

theorem quotient (c : DimConfig) (job : Nat × Nat) (hi : job.2 < c.inner) :
    (job.1*c.inner+job.2)/c.inner = job.1 := by
  have positive : 0 < c.inner := by omega
  simpa [Nat.add_comm, Nat.mul_comm, Nat.div_eq_of_lt hi] using Nat.add_mul_div_left job.2 job.1 positive

theorem remainder (c : DimConfig) (job : Nat × Nat) (hi : job.2 < c.inner) :
    (job.1*c.inner+job.2)%c.inner = job.2 := by simp [Nat.add_mod, Nat.mod_eq_of_lt hi]

theorem coordinates (K r r' i i' : Nat) (hi : i < K) (hi' : i' < K)
    (equal : r*K+i = r'*K+i') : r = r' ∧ i = i' := by
  have rows := congrArg (fun o => o/K) equal
  have cols := congrArg (fun o => o%K) equal
  have positive : 0 < K := by omega
  have qr : (r*K+i)/K = r := by
    simpa [Nat.add_comm, Nat.mul_comm, Nat.div_eq_of_lt hi] using Nat.add_mul_div_left i r positive
  have qr' : (r'*K+i')/K = r' := by
    simpa [Nat.add_comm, Nat.mul_comm, Nat.div_eq_of_lt hi'] using Nat.add_mul_div_left i' r' positive
  constructor
  · simpa only [qr, qr'] using rows
  · simpa [Nat.add_mod, Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hi'] using cols

/-- One complete dimension quotation, connected to the logical input row and
its exact active output footprint. -/
theorem program (rest : Algebra Word) (c : DimConfig) (index64 : Bool)
    (nonempty : 0 < c.extent) (job : Nat × Nat) (hi : job.2 < c.inner)
    (s : State Word) (input : Nat → Nat → Nat → Word)
    (reads : ∀ r col inner, r < c.outer → col < c.extent → inner < c.inner →
      (s.mem BoundSource.X (r*c.extent*c.inner+col*c.inner+inner)).read .real = input r inner col) :
    ∃ t, (launch rest c index64 job).run s = some t ∧
      (∀ lane : Fin 128, job.1*128+lane.val < c.outer →
        t.mem (Region.cast BoundSource.O) ((job.1*128+lane.val)*c.inner+job.2) =
          .mk .int (formula c nonempty input (job.1*128+lane.val) job.2)) ∧
      (∀ r o, (r ≠ Region.cast BoundSource.O ∨ ∀ lane : Fin 128,
        job.1*128+lane.val < c.outer → o ≠ (job.1*128+lane.val)*c.inner+job.2) →
          t.mem r o = s.mem r o) := by
  let initial := (launch rest c index64 job).initial s
  have hp : initial.pids 0 = job.1*c.inner+job.2 := by simp [initial, launch]
  have hm : initial.mem = s.mem := Invocation.initial_mem _ _
  obtain ⟨t,ran,output,frame⟩ := Typed.source_reference rest initial BoundSource.X BoundSource.O
    c.outer c.extent c.inner 128 128 0 index64 (by decide) nonempty (by omega)
  refine ⟨t, ran, ?_, ?_⟩
  · intro lane active
    have result := output lane (by simpa [hp, quotient c job hi] using active)
    simp only [hp, Nat.add_zero, quotient c job hi, remainder c job hi] at result
    have same : (fun j : Fin c.extent => Typed.inputRow initial BoundSource.X c.extent c.inner
        (job.1*128+lane.val) job.2 j.val) =
        (fun j : Fin c.extent => input (job.1*128+lane.val) job.2 j.val) := by
      funext col
      change (initial.mem BoundSource.X _).read .real = _
      rw [hm]
      exact reads _ _ _ active col.isLt hi
    simpa only [same, formula] using result
  · intro r o outside
    simpa only [hp, Nat.add_zero, quotient c job hi, remainder c job hi, hm] using
      frame r o (by simpa only [hp, Nat.add_zero, quotient c job hi, remainder c job hi] using outside)

#axiomsClean program
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.NativeDimensionIO
