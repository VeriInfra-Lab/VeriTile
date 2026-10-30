import bench.tritonbench_g.max_reduction.Proofs.Source

namespace VeriTile.Bench.Repairs.MaxReduction.Stores
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem frame (e : Source.Engine α) (inp ov oi : RegionName) (M N K BM BN : Nat)
    (s : State α) (r : RegionName) (o : Nat)
    (hv : r≠ov ∨ ∀ i : TileIndex [BM],s.pids 0*BM+i.1.val<M → o≠(s.pids 0*BM+i.1.val)*K+s.pids 1)
    (hx : r≠oi ∨ ∀ i : TileIndex [BM],s.pids 0*BM+i.1.val<M → o≠(s.pids 0*BM+i.1.val)*K+s.pids 1) :
    (Source.dimensionStores e inp ov oi M N K BM BN s).mem r o=s.mem r o := by
  unfold Source.dimensionStores
  rw [State.masked_scatter_frame]
  · apply State.masked_scatter_frame
    exact hv.imp id (fun h i _ active => h i active)
  · exact hx.imp id (fun h i _ active => h i active)

theorem offset_injective (BM K p k : Nat) (hK : 0<K) :
    Function.Injective (fun i : TileIndex [BM] => (p*BM+i.1.val)*K+k) := by
  intro i j same
  have eq : i.1.val=j.1.val := Nat.add_left_cancel
    (Nat.eq_of_mul_eq_mul_right hK (Nat.add_right_cancel same))
  apply Prod.ext
  · exact Fin.ext eq
  · exact Subsingleton.elim _ _

theorem values (e : Source.Engine α) (inp ov oi : RegionName) (M N K BM BN : Nat)
    (regions : ov≠oi) (hK : 0<K) (s : State α) (i : TileIndex [BM])
    (active : s.pids 0*BM+i.1.val<M) :
    let answer := Source.rowValue e (fun j => (s.mem inp j).read .real) M N K BM BN (s.pids 0) (s.pids 1) i.1.val
    (Source.dimensionStores e inp ov oi M N K BM BN s).mem ov ((s.pids 0*BM+i.1.val)*K+s.pids 1)=.mk .real answer.1 ∧
    (Source.dimensionStores e inp ov oi M N K BM BN s).mem oi ((s.pids 0*BM+i.1.val)*K+s.pids 1)=.mk .int (answer.2 : Int) := by
  dsimp only
  constructor
  · unfold Source.dimensionStores
    rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl regions)]
    exact State.masked_scatter_readback _ _ _ _ _ i active (offset_injective BM K _ _ hK)
  · unfold Source.dimensionStores
    exact State.masked_scatter_readback _ _ _ _ _ i active (offset_injective BM K _ _ hK)

#axiomsClean frame
#axiomsClean values
end
end VeriTile.Bench.Repairs.MaxReduction.Stores
