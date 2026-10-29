import bench.repairs.tritonbench_g.fast_rope_embedding.IO
import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
set_option maxHeartbeats 4000000

theorem implements {A : Type} [Inhabited A] (op : Operations A) (r : Request)
    (qStride cosineStride sineStride : Nat) (backward : Bool) (row group : Nat)
    (launch : r.Launched row group) (strideEq : row * qStride = row * (r.heads * r.dimension))
    (cosineEq : cosineStride = r.half) (sineEq : sineStride = r.half) :
    (programIO op r qStride cosineStride sineStride backward row group).Implements
      (fun values => result (fun h c => scalarPair op backward
        (query values h (left r (lane r c))) (query values h (right r (lane r c)))
        (cosine values (lane r c)) (sine values (lane r c)) (decide (r.half ≤ c.val)))) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    rw [show (programIO op r qStride cosineStride sineStride backward row group).schedule values =
      [invocation op r qStride cosineStride sineStride backward row group] from rfl,
      scheduleSafe_singleton]
    apply SourceSafety.source_safe op.backend (KernelInterface.rawParameters r qStride cosineStride sineStride)
    · simpa [invocation, Invocation.initial, bindRegisters, KernelInterface.rawParameters]
        using Nat.le_of_lt (group_head r group launch.2.2)
    · intro i active
      have hi : i.1.val < r.half := active
      simpa [invocation, Invocation.initial, bindRegisters, KernelInterface.rawParameters,
        sineEq, programIO, memory, read, inputs]
        using hb.2.1 ⟨2, by simp [programIO, memory, inputs]⟩ (⟨i.1.val, hi⟩, PUnit.unit) trivial
    · intro i active
      have hi : i.1.val < r.half := active
      simpa [invocation, Invocation.initial, bindRegisters, KernelInterface.rawParameters,
        cosineEq, programIO, memory, read, inputs]
        using hb.2.1 ⟨1, by simp [programIO, memory, inputs]⟩ (⟨i.1.val, hi⟩, PUnit.unit) trivial
    · intro h hlo hhi side i active
      have lower : group * 4 ≤ h := hlo
      have upper : h < min (group * 4 + 4) r.heads := hhi
      have hh : h < r.heads := lt_of_lt_of_le upper (min_le_right _ _)
      have hi : i.1.val < r.half := active
      have hc : i.1.val + (if side then r.half else 0) < 2 * r.half := by cases side <;> simp <;> omega
      have bound := hb.2.1 ⟨0, by simp [programIO, memory, inputs]⟩
        (⟨h, hh⟩, ⟨_, hc⟩, PUnit.unit) ⟨lower, upper⟩
      simpa [invocation, Invocation.initial, bindRegisters, KernelInterface.rawParameters,
        Source.offset, programIO, memory, read, inputs, Request.offset,
        strideEq, Nat.add_mul, Nat.mul_assoc, Nat.add_assoc] using bound
  · intro values s hin
    let before : State A := {s with pids := fun _ => 0}
    obtain ⟨final, ran, post⟩ := KernelInterface.executes op r qStride cosineStride sineStride
      backward before row group launch strideEq cosineEq sineEq
    refine ⟨final, ?_, ?_, ?_⟩
    · change runSchedule [invocation op r qStride cosineStride sineStride backward row group] s = some final
      rw [runSchedule_singleton]
      exact ran
    · intro o j active
      fin_cases o
      have value := post.values j.1.val j.2.1.val active.1 active.2 j.2.1.isLt
      have qleft := hin ⟨0, by simp [programIO, memory, inputs]⟩
        (j.1, left r (lane r j.2.1), PUnit.unit) active
      have qright := hin ⟨0, by simp [programIO, memory, inputs]⟩
        (j.1, right r (lane r j.2.1), PUnit.unit) active
      have c := hin ⟨1, by simp [programIO, memory, inputs]⟩ (lane r j.2.1, PUnit.unit) trivial
      have si := hin ⟨2, by simp [programIO, memory, inputs]⟩ (lane r j.2.1, PUnit.unit) trivial
      simp only [programIO, memory, inputs, inputMask, read, mapValue] at qleft qright c si
      have target : scalarResult op r backward
          (fun o => (before.mem "Q" o).read .real) (fun o => (before.mem "C" o).read .real)
          (fun o => (before.mem "S" o).read .real) row j.1.val j.2.1.val =
          scalarPair op backward (query values j.1 (left r (lane r j.2.1)))
            (query values j.1 (right r (lane r j.2.1)))
            (cosine values (lane r j.2.1)) (sine values (lane r j.2.1))
            (decide (r.half ≤ j.2.1.val)) := by
        simp only [scalarResult, j.2.1.isLt, ↓reduceIte]
        change scalarPair op backward _ _ _ _ _ = _
        change (s.mem "Q" (r.offset row j.1.val (lane r j.2.1).val)).read .real = _ at qleft
        change (s.mem "Q" (r.offset row j.1.val ((lane r j.2.1).val + r.half))).read .real = _ at qright
        congr 1
      simpa [programIO, memory, outputs, write, result, mapCell, mapValue, target]
        using congrArg (mapCell (id : A → A)) value
    · intro region offset outside
      apply post.frame
      rintro ⟨same, h, c, hlo, hhi, hc, address⟩
      have hh := lt_of_lt_of_le hhi (min_le_right _ _)
      have miss := outside.1 ⟨0, by simp [programIO, memory, outputs]⟩
        (⟨h, hh⟩, ⟨c, hc⟩, PUnit.unit) ⟨hlo, hhi⟩
      exact (miss.resolve_left (by simpa [outputs] using not_ne_iff.mpr same)) address

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
