import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Visit

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem accumulation_succ {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial : State (Replicas α)) (visit : Nat) :
    accumulation a native p initial (visit+1) =
      update a native p initial (csrColumn p initial (first p initial+visit))
        (accumulation a native p initial visit) := by
  simp [accumulation,List.range_succ,List.foldl_append]

/-- The dynamic signed CSR range preserves its endpoint semantics. Reversed
endpoints execute zero iterations, and duplicate columns remain occurrences. -/
theorem outerLoopExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (visit : Nat) (bound : visit≤visitCount p initial)
    (h : Runtime a p initial (accumulation a native p initial visit) current)
    (integers : Int32Memory initial) :
    ∃ final, rangeInt a.model "col_idx_idx" (first p initial+visit) (last p initial) 1
      (loopBody native p) current=some final ∧
      Runtime a p initial (accumulation a native p initial (visitCount p initial)) final := by
  have aux : ∀ n k (current : State (Replicas α)), visitCount p initial-k=n →
      k≤visitCount p initial → Runtime a p initial (accumulation a native p initial k) current →
      ∃ final, rangeInt a.model "col_idx_idx" (first p initial+k) (last p initial) 1
        (loopBody native p) current=some final ∧
        Runtime a p initial (accumulation a native p initial (visitCount p initial)) final := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases less : k<visitCount p initial
      · obtain ⟨next,he,hi⟩ := visitExecution a native p initial current
          (accumulation a native p initial k) (first p initial+k) h integers
        rw [← accumulation_succ] at hi
        obtain ⟨final,hf,hfi⟩ := ih (visitCount p initial-(k+1)) (by omega)
          (k+1) next rfl (by omega) hi
        refine ⟨final,?_,hfi⟩
        have active : first p initial+(k : Int)<last p initial := by
          have positive : 0<last p initial-first p initial := by
            by_contra nonpositive
            have empty : visitCount p initial=0 :=
              Int.toNat_eq_zero.mpr (le_of_not_gt nonpositive)
            omega
          have cast : ((visitCount p initial:Nat):Int)=last p initial-first p initial :=
            Int.toNat_of_nonneg (le_of_lt positive)
          have lt : (k:Int)<(visitCount p initial:Int) := Int.ofNat_lt.mpr less
          rw [cast] at lt
          simpa only [Int.add_comm] using (lt_sub_iff_add_lt.mp lt)
        rw [rangeInt,if_neg (by decide),if_pos active,he]
        simpa only [Option.bind_some,Int.natCast_add,Int.natCast_one,Int.add_assoc] using hf
      · have equal : k=visitCount p initial := by omega
        subst k
        refine ⟨current,?_,h⟩
        have inactive : ¬first p initial+(visitCount p initial : Int)<last p initial := by
          have cast : last p initial-first p initial≤(visitCount p initial:Int) :=
            Int.self_le_toNat _
          exact not_lt.mpr (by simpa only [Int.add_comm] using (sub_le_iff_le_add.mp cast))
        rw [rangeInt,if_neg (by decide),if_neg inactive]
  exact aux _ _ _ rfl bound h

#axiomsClean accumulation_succ
#axiomsClean outerLoopExecution
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
