import bench.tritonbench_g.embedding_triton_kernel.Proofs.Execution
import bench.tritonbench_g.embedding_triton_kernel.Proofs.Bounds

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.KernelProof
open VeriTile.Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α]

def expected (zero : α) (p : Parameters) (s : State α) (r d : Nat) : α :=
  target zero p.first p.last (fun r => (s.mem p.ids r).read .int)
    (fun r d => (s.mem p.weight (r*p.weightStride+d)).read .real) r d

def Written (p : Parameters) (pid off : Nat) : Prop :=
  ∃ a,a < 64 ∧ pid*64+a < p.n ∧ ∃ d,d < p.hidden ∧ off = (pid*64+a)*p.outputStride+d

/-- The source's selected signed weight offset, before masked dereference. -/
def weightOffset (p : Parameters) (s : State α) (a d : Nat) : Int :=
  let r := s.pids 0*64+a
  let token := if r < p.n then (s.mem p.ids r).read .int else 0
  let valid := r < p.n ∧ p.first ≤ token ∧ token < p.last
  ((if valid then token else p.first)-p.first)*(p.weightStride : Int)+(d : Int)

/-- The source's masked-off output-row selection, before masked dereference. -/
def outputOffset (p : Parameters) (s : State α) (a d : Nat) : Nat :=
  (if s.pids 0*64+a < p.n then s.pids 0*64+a else 0)*p.outputStride+d

theorem accesses (p : Parameters) (bounds : RegionBounds) (s : State α)
    (h : KernelContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    (∀ a,a < 64 → s.pids 0*64+a < p.n → s.pids 0*64+a < bounds p.ids) ∧
    (∀ a,a < 64 → s.pids 0*64+a < p.n →
      p.first ≤ (s.mem p.ids (s.pids 0*64+a)).read .int →
      (s.mem p.ids (s.pids 0*64+a)).read .int < p.last → ∀ d,d < p.hidden →
      0 ≤ weightOffset p s a d ∧ (weightOffset p s a d).toNat < bounds p.weight) ∧
    (∀ a,a < 64 → s.pids 0*64+a < p.n → ∀ d,d < p.hidden →
      outputOffset p s a d < bounds p.output) := by
  refine ⟨h.idAccess,?_,?_⟩
  · intro a ha hr lo hi d hd
    have offset : weightOffset p s a d =
        ((s.mem p.ids (s.pids 0*64+a)).read .int-p.first)*(p.weightStride : Int)+(d : Int) := by
      simp [weightOffset,hr,lo,hi]
    rw [offset]
    constructor
    · exact Int.add_nonneg (Int.mul_nonneg (by omega) (Int.natCast_nonneg _)) (Int.natCast_nonneg _)
    · rw [Execution.nonnegative_offset _ _ _ _ lo]
      exact h.weightAccess a ha hr lo hi d hd
  · intro a ha hr d hd
    simpa only [outputOffset,if_pos hr] using h.outputAccess a ha hr d hd

/-- Complete typed-kernel execution over arbitrary payload values and initial
register contents. The only interpreted floating operation is copy/zero. -/
theorem correct (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α)
    (h : KernelContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    ∃ t,FP.Structural.exec (Execution.backend rest zero) (source .int32 p) s = some t ∧
      (∀ a,a < 64 → s.pids 0*64+a < p.n → ∀ d,d < p.hidden →
        t.mem p.output ((s.pids 0*64+a)*p.outputStride+d) =
          .mk .real (expected zero p s (s.pids 0*64+a) d)) ∧
      (∀ r off,r ≠ p.output ∨ ¬Written p (s.pids 0) off → t.mem r off = s.mem r off) := by
  obtain ⟨t,run,_,values,frame⟩ := Execution.program_correct rest zero .int32 p
    ⟨h.ids_distinct,h.weight_distinct,h.width_covers,h.stride_covers⟩ s
  refine ⟨t,run,?_,?_⟩
  · intro a ha hr d hd
    simpa [Execution.row,Execution.value,Execution.valid,Execution.token,expected,target,hr]
      using values a ha hr d hd
  · intro r off miss
    apply frame
    rcases miss with miss | miss
    · exact Or.inl miss
    · right
      intro a ha hr d hd eq
      exact miss ⟨a,ha,hr,d,hd,eq⟩

#specNonCircular expected avoiding [source,indexCommutedSource,FP.Structural.exec]
#axiomsClean accesses
#axiomsClean correct
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.KernelProof
