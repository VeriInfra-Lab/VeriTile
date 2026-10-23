import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedWhole

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

def inputRow (s : State Word) (X : RegionName) (N K row inner : Nat) : Nat → Word :=
  fun column => (s.mem X (row*N*K+column*K+inner)).read .real

theorem next_scalar (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Word × Nat) (i : Fin BM) (active : row+i.val<M)
    (tree : Tree BN) (complete : tree.Complete) :
    next s X M N K BM BN row inner cur positive current i=
      strictStep N cur BN (inputRow s X N K (row+i.val) inner) tree (current i) := by
  have loads : loaded s X M N K BM BN row inner cur i=
      padded N cur BN (inputRow s X N K (row+i.val) inner) := by
    funext j
    simp [loaded,padded,inputRow,active]
  unfold next
  rw [TypedReduction.selected_eq positive _ tree complete,loads]
  simp only [strictStep,localPair,TritonArgmaxNanFirst.selected,padded,inputRow,active,true_and]

theorem vector_scalar (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Word × Nat) (i : Fin BM) (active : row+i.val<M)
    (tree : Nat → Tree BN) (complete : ∀ col,(tree col).Complete) :
    vectorLoop s X M N K BM BN row inner positive cur current i=
      TritonArgmaxNanFirst.loop N BN positive (inputRow s X N K (row+i.val) inner) tree cur (current i) := by
  by_cases running : cur<N
  · rw [vectorLoop,dif_pos running,TritonArgmaxNanFirst.loop,dif_pos running]
    rw [vector_scalar s X M N K BM BN row inner (cur+BN) positive _ i active tree complete,
      next_scalar s X M N K BM BN row inner cur positive current i active (tree cur) (complete cur)]
  · rw [vectorLoop,dif_neg running,TritonArgmaxNanFirst.loop,dif_neg running]
termination_by N-cur
decreasing_by omega

theorem vector_reference (s : State Word) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (nonempty : 0<N) (i : Fin BM) (active : row+i.val<M) :
    (vectorLoop s X M N K BM BN row inner positive 0 (fun _ => (0xff800000,0)) i).2=
      (reference N nonempty (fun j : Fin N => inputRow s X N K (row+i.val) inner j.val)).val := by
  let tree := Math.enumeration BN positive
  have complete : tree.Complete := Math.enumeration_complete BN positive
  rw [vector_scalar s X M N K BM BN row inner 0 positive _ i active (fun _ => tree) (fun _ => complete)]
  obtain ⟨bound,correct⟩ := loop_correct N BN nonempty positive (inputRow s X N K (row+i.val) inner)
    (fun _ => tree) (fun _ _ => complete)
  exact congrArg Fin.val (least_maximum_unique _ _ _ correct
    (reference_spec N nonempty (fun j => inputRow s X N K (row+i.val) inner j.val)))

theorem source_reference (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0<BN) (nonempty : 0<N) (innerPositive : 0<K) :
    ∃ t, FP.Structural.exec (backend rest) (TypedSource.dimension X O M N K BM BN base index64) s=some t ∧
      (∀ i : Fin BM,((s.pids 0+base)/K)*BM+i.val<M →
        t.mem (Region.cast O) ((((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K)=
          .mk .int (Int.ofNat ((reference N nonempty (fun j : Fin N =>
            inputRow s X N K (((s.pids 0+base)/K)*BM+i.val) ((s.pids 0+base)%K) j.val)).val))) ∧
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
    · rw [vector_reference s X M N K BM BN _ _ positive nonempty i active]
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

#axiomsClean next_scalar
#axiomsClean vector_reference
#axiomsClean source_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
