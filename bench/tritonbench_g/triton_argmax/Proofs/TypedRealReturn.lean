import bench.tritonbench_g.triton_argmax.Proofs.TypedReal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section

theorem next_scalar (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Value × Nat) (i : Fin BM) (active : row+i.val<M)
    (input : Nat → ℝ)
    (reads : ∀ col,col<N → (s.mem X ((row+i.val)*N*K+col*K+inner)).read .real=(input col : Value)) :
    next s X M N K BM BN row inner cur positive current i=
      Math.strictStep N cur BN input (Math.enumeration BN positive) (current i) := by
  have loads : loaded s X M N K BM BN row inner cur i=Math.padded N cur BN input := by
    funext j
    by_cases valid : cur+j.val<N
    · simp only [loaded,active,valid,and_self,if_true,Math.padded,reads _ valid]
    · simp [loaded,Math.padded,valid]
  unfold next
  rw [loads]
  simp only [best,Math.strictStep,Math.localPair,Math.blockSelected]
  have value : (if row+i.val<M ∧ cur+((Math.enumeration BN positive).pick orderedGt orderedEq (Math.padded N cur BN input)).val<N then
      (s.mem X ((row+i.val)*N*K+(cur+((Math.enumeration BN positive).pick orderedGt orderedEq (Math.padded N cur BN input)).val)*K+inner)).read .real else ⊥)=
      Math.padded N cur BN input ((Math.enumeration BN positive).pick orderedGt orderedEq (Math.padded N cur BN input)) := by
    by_cases valid : cur+((Math.enumeration BN positive).pick orderedGt orderedEq (Math.padded N cur BN input)).val<N
    · simp [Math.padded,active,valid,reads _ valid]
    · simp [Math.padded,active,valid]
  rw [value]

theorem vector_scalar (s : State Value) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Value × Nat) (i : Fin BM) (active : row+i.val<M)
    (input : Nat → ℝ)
    (reads : ∀ col,col<N → (s.mem X ((row+i.val)*N*K+col*K+inner)).read .real=(input col : Value)) :
    vectorLoop s X M N K BM BN row inner positive cur current i=
      Math.dimensionLoop N BN positive input (fun _ => Math.enumeration BN positive) cur (current i) := by
  by_cases running : cur<N
  · rw [vectorLoop,dif_pos running,Math.dimensionLoop,dif_pos running]
    rw [vector_scalar s X M N K BM BN row inner (cur+BN) positive _ i active input reads,
      next_scalar s X M N K BM BN row inner cur positive current i active input reads]
  · rw [vectorLoop,dif_neg running,Math.dimensionLoop,dif_neg running]
termination_by N-cur
decreasing_by omega

theorem vector_reference (s : State Value) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (nonempty : 0<N) (i : Fin BM) (active : row+i.val<M)
    (input : Nat → ℝ)
    (reads : ∀ col,col<N → (s.mem X ((row+i.val)*N*K+col*K+inner)).read .real=(input col : Value)) :
    (vectorLoop s X M N K BM BN row inner positive 0 (fun _ => (⊥,0)) i).2=
      (Math.argmax N nonempty (fun j : Fin N => input j.val)).val := by
  rw [vector_scalar s X M N K BM BN row inner 0 positive _ i active input reads]
  obtain ⟨bound,correct⟩ := Math.dimension_mathematical N BN nonempty positive input
    (fun _ => Math.enumeration BN positive) (fun _ _ => Math.enumeration_complete BN positive)
  exact congrArg Fin.val (least_maximum_unique _ _ _ correct (Math.argmax_spec N nonempty (fun j => input j.val)))

/-- Complete execution of the shared quotation returns the independent Real
argmax for arbitrary logical Real inputs, with exact output stores and frame. -/
theorem source_reference (rest : Algebra Value) (s : State Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) (nonempty : 0<N) (innerPositive : 0<K)
    (input : Nat → Nat → Nat → ℝ)
    (reads : ∀ row col inner,row<M → col<N → inner<K →
      (s.mem X (row*N*K+col*K+inner)).read .real=(input row inner col : Value)) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.dimension X O M N K BM BN base index64) s=some t ∧
      (∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
        t.mem (Region.cast O) ((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)=
          .mk .int (Int.ofNat ((Math.argmax N nonempty (fun j : Fin N =>
            input (((s.pids 0+base)/K)*BM+i.val) ((s.pids 0+base)%K) j.val)).val))) ∧
      (∀ region offset,(region≠Region.cast O ∨
        ∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
          offset≠((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t,run,mem⟩ := source_run rest s X O M N K BM BN base index64 positive
  refine ⟨t,run,?_,?_⟩
  · intro i active
    rw [mem]
    unfold written
    rw [State.masked_scatter_readback (Region.cast O)
      (fun j : TileIndex [BM] => (((s.pids 0+base)/K)*BM+j.1.val)*K+(s.pids 0+base)%K) _
      (fun j : TileIndex [BM] => ((s.pids 0+base)/K)*BM+j.1.val<M) s (i,PUnit.unit) active]
    · rw [vector_reference s X M N K BM BN _ _ positive nonempty i active
        (input _ _) (fun col valid => reads _ col _ active valid (Nat.mod_lt _ innerPositive))]
    · rintro ⟨i,_⟩ ⟨j,_⟩ same
      have sameVal : i.val=j.val := by nlinarith
      have sameFin : i=j := Fin.ext sameVal
      cases sameFin
      rfl
  · intro region offset outside
    rw [mem]
    unfold written
    apply State.masked_scatter_frame
    rcases outside with region | offset
    · exact Or.inl region
    · exact Or.inr (fun i _ hi => offset i.1 hi)

#axiomsClean vector_reference
#axiomsClean source_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReal
