import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.HostReturn
namespace VeriTile.Bench.Repairs.TritonLinearActivation
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent dense-matrix specification, indexed directly by the logical
caller matrices, with no tile, instruction, block or grid in the target. -/
def densePre (r : Request) (before : Memory ℝ) (m n : Nat) : ℝ :=
  (if r.hasBias then before (r.storage 2,n) else 0) +
    ∑ k∈Finset.range r.k, before (r.storage 0,m*r.k+k)*before (r.storage 1,n*r.k+k)

def denseOutput (r : Request) (before : Memory ℝ) (m n : Nat) : ℝ :=
  realActivation r.activation (densePre r before m n)

@[simp] theorem inputConversion_real (mode : Mode) (raw : ℝ) :
    inputConversion realArithmetic mode raw=raw := by
  cases mode <;> rfl

theorem callerPre_real (r : Request) (c : Fin 98) (before : Memory ℝ) (m n : Nat) :
    callerPre realArithmetic r c before m n=densePre r before m n := by
  unfold callerPre densePre
  rw [accumulation_real]
  simp only [inputConversion_real]
  congr 1
  · cases r.hasBias <;> rfl
  · apply Finset.sum_congr rfl
    intro k hk
    have positive : 1≤r.k := by have := Finset.mem_range.mp hk; omega
    simp only [max_eq_left positive]

variable {r : Request} {caller : Set Nat} {before : Memory ℝ}

def RealPost (r : Request) (h : Allocation r caller before) (result : HostResult ℝ)
    (currentDevice : Nat) : Prop :=
  (∀ m n,m<r.batch.rows → n<r.n →
    result.memory (h.storage 0,outputOffset r.shape m n)=denseOutput r before m n) ∧
  (∀ m n,m<r.batch.rows → n<r.n → r.hasSaved=true →
    result.memory (r.storage 3,outputOffset r.shape m n)=densePre r before m n) ∧
  (∀ p,p.1∈caller → ¬(r.hasSaved=true ∧ p.1=r.storage 3 ∧ p.2<r.batch.rows*r.n) →
    result.memory p=before p) ∧ HostMetadata r h result currentDevice

theorem whole_real (h : Allocation r caller before) (c : Fin 98) (valid : r.Valid) (currentDevice : Nat) :
    RealPost r h (h.hostResult realArithmetic c currentDevice) currentDevice := by
  refine ⟨?_,?_,?_,h.host_metadata realArithmetic c currentDevice⟩
  · intro m n hm hn
    change h.finalMemory realArithmetic c _=_
    rw [h.final_output realArithmetic c valid m n hm hn,callerPre_real]
    exact activation_real r.activation _
  · intro m n hm hn saved
    change h.finalMemory realArithmetic c _=_
    rw [h.final_saved realArithmetic c valid m n hm hn saved,callerPre_real]
    cases r.mode <;> rfl
  · intro p hp outside
    exact h.final_frame realArithmetic c valid p hp outside

theorem densePre_zero_k (r : Request) (before : Memory ℝ) (m n : Nat) (hk : r.k=0) :
    densePre r before m n=(if r.hasBias then before (r.storage 2,n) else 0) := by
  simp [densePre,hk]

theorem no_launch (h : Allocation r caller before) (a : Arithmetic ℝ) (c : Fin 98)
    (empty : r.batch.rows=0 ∨ r.n=0) : h.kernel a c=h.prepared a := by
  have off : ¬r.launch := by unfold Request.launch; omega
  simp [Allocation.kernel,off]

#axiomsClean callerPre_real
#axiomsClean whole_real
#axiomsClean densePre_zero_k
#axiomsClean no_launch
end
end VeriTile.Bench.Repairs.TritonLinearActivation

