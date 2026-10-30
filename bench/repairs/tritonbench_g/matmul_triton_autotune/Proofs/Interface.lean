import bench.repairs.tritonbench_g.matmul_triton_autotune.IO
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.AccessSafety
import bench.tritonbench_g.matmul_triton_autotune.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 4000000
variable {A : Type} [Inhabited A]

theorem invocation_initial (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry)
    (pid : Nat) (s : State A) : (invocation M D g pid).initial s = instanceState g.request pid s := rfl

theorem run_eq (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry)
    (ids : List Nat) (s : State A) :
    runSchedule (ids.map (invocation M D g)) s = Repair.runPrograms M D g.request ids s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, runSchedule, Invocation.run, invocation_initial, invocation,
      Repair.runPrograms, ih]
    rfl

theorem schedule_safe (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry)
    (ids : List Nat) (valid : ∀ pid ∈ ids, pid < programs g.request)
    (values : Inputs A g) (bounds : RegionName → Nat)
    (hb : (programIO M D g ids).InBounds bounds values) (s : State A) :
    ScheduleSafe bounds (ids.map (invocation M D g)) s := by
  induction ids generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_, ?_⟩
    · have dims := nonempty_program g.request pid (valid pid (by simp))
      change FP.Structural.TraceSafe (FPValue.backend M D) bounds
        (Source.kernel g.request.parameters) (instanceState g.request pid s)
      apply Source.source_safe
      · intro q i active
        have row : FPValue.aRow g.request.parameters (instanceState g.request pid s) i.1 < g.rows :=
          Nat.mod_lt _ dims.1
        exact hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩
          (⟨_, row⟩, ⟨_, active⟩, PUnit.unit) trivial
      · intro q i active
        have col : FPValue.bCol g.request.parameters (instanceState g.request pid s) i.2.1 < g.columns :=
          Nat.mod_lt _ dims.2
        exact hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩
          (⟨_, active⟩, ⟨_, col⟩, PUnit.unit) trivial
      · intro i active
        have output := hb.2.2.1 ⟨0, by simp [programIO, memory, outputs]⟩
          (⟨_, active.1⟩, ⟨_, active.2⟩, PUnit.unit) trivial
        simpa [programIO, memory, write, FPValue.cOffset, Request.parameters, Geometry.request] using output
    · intro t _
      exact ih (fun j hj => valid j (by simp [hj])) hb t

theorem input_values (M : Algebra A) (D : FPValue.DotAcc A) (g : Geometry)
    (ids : List Nat) (values : Inputs A g) (s : State A)
    (hin : (programIO M D g ids).InputBinding values s) :
    (∀ i k, inputA g.request s i.val k.val = left values i k) ∧
      (∀ k j, inputB g.request s k.val j.val = right values k j) := by
  constructor
  · intro i k
    exact hin ⟨0, by simp [programIO, memory, inputs]⟩ (i, k, PUnit.unit) trivial
  · intro k j
    exact hin ⟨1, by simp [programIO, memory, inputs]⟩ (k, j, PUnit.unit) trivial

theorem implements (M : Algebra A) (D : FPValue.DotAcc A) (Q : Numerics A)
    (law : PrimitiveLaws M D Q) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C ≠ g.A) (sepB : g.C ≠ g.B) :
    (programIO M D g ids).Implements (fun values => result g
      (logicalReference Q g.request (left values) (right values))) := by
  intro values _ bounds hb s hin
  refine ⟨schedule_safe M D g ids (fun pid hp => List.mem_range.mp (order.mem_iff.mp hp)) values bounds hb s, ?_⟩
  obtain ⟨a, b⟩ := input_values M D g ids values s hin
  obtain ⟨final, ran, computed, frame⟩ := candidate_grid_order_correct M D Q law g.request s ids order sepA sepB
  refine ⟨final, (run_eq M D g ids s).trans ran, ?_, ?_⟩
  · intro o j active
    fin_cases o
    have cell := computed j.1.val j.1.isLt j.2.1.val j.2.1.isLt
    rw [reference_logical Q g.request s (left values) (right values) a b] at cell
    change mapCell id (final.mem g.request.C (outputOffset g.request j.1.val j.2.1.val)) =
      .mk .fp16 (logicalReference Q g.request (left values) (right values) j.1 j.2.1)
    rw [cell]
    rfl
  · intro r i outside
    apply frame r i
    rintro ⟨same, row, hr, col, hc, offset⟩
    have miss := outside.1 ⟨0, by simp [programIO, memory, outputs]⟩
      (⟨row, hr⟩, ⟨col, hc⟩, PUnit.unit) trivial
    exact miss.elim (fun h => h same) (fun h => h offset)

#axiomsClean schedule_safe
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.Interface
