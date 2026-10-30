import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOReal
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.RealBody

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
noncomputable section
set_option maxHeartbeats 4000000
namespace ForwardOSource
variable {α : Type} [Inhabited α]

theorem address_injective (c : Shape) (s : State α) (i j : Index c)
    (hi : active c s i) (hj : active c s j)
    (he : outputAddress c s i=outputAddress c s j) : i=j := by
  have hh : row s i.1.val*c.V+column c s i.2.1.val=
      row s j.1.val*c.V+column c s j.2.1.val := by
    simpa only [outputAddress,Nat.add_assoc,Nat.add_left_cancel_iff] using he
  obtain ⟨hr,hv⟩ := digit_injective c.V _ _ _ _ hi.2 hj.2 hh
  have h1 : i.1=j.1 := Fin.ext (by simpa [row] using hr)
  have h2 : i.2.1=j.2.1 := Fin.ext (by simpa [column] using hv)
  rcases i with ⟨ik,iv,⟨⟩⟩
  rcases j with ⟨jk,jv,⟨⟩⟩
  simp_all

end ForwardOSource
namespace KernelReal.O
open ForwardOSource

/-- Independent mathematical output of one selected output program. Its saved
state tile is an ordinary kernel input, read at the quoted history addresses. -/
def target (c : Shape) (s : State ℝ) (scale : ℝ) (i : Index c) : ℝ :=
  ScalarSource.outputTile c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (column c s i.2.1.val) (history c s)

theorem value_target (c : Shape) (s : State ℝ) (scale : ℝ) (i : Index c)
    (hi : active c s i) : value algebra dot c scale s i=target c s scale i := by
  rw [value_body]
  exact RealScalar.output_real c (xs c s) scale (s.pids 2) (s.pids 1)
    i.1.val (column c s i.2.1.val) (history c s) hi.1 i.1.isLt hi.2

theorem source_result (c : Shape) (s : State ℝ) (scale : ℝ) :
    ∃ t, FP.Structural.exec (engine (withDot algebra dot)) (kernel c scale) s=some t ∧
      (∀ i, active c s i →
        (t.mem "o" (outputAddress c s i)).read .real=target c s scale i) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"o" ∨ ∀ i, active c s i → off≠outputAddress c s i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory algebra dot c scale s
  refine ⟨t,ht,?_,?_⟩
  · intro i hi
    rw [hm,readback algebra dot c scale s i hi (address_injective c s)]
    exact value_target c s scale i hi
  · intro r off hf
    rw [hm]
    exact frame algebra dot c scale s r off hf

#specNonCircular target avoiding [run,FP.Structural.exec,ForwardOSource.value]
#axiomsClean source_result
end KernelReal.O
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
