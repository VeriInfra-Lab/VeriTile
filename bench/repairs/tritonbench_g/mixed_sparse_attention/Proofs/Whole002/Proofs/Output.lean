import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Whole
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Inactive
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Scatter
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def active (c : Config) (s : State α) (i : TileIndex [c.BM,c.BD]) : Prop :=
  rowStart c s+i.1.val<sequenceLength c s
instance (c : Config) (s : State α) : DecidablePred (active c s) := fun _=>inferInstanceAs (Decidable (_<_))
def address (c : Config) (s : State α) (i : TileIndex [c.BM,c.BD]) : Nat :=
  queryBase c s+(rowStart c s+i.1.val)*c.som+i.2.1.val*c.sok

def outputValue (M : Algebra α) (D : Dot α) (c : Config) (a : Accumulator α c.BM c.BD)
    (i : TileIndex [c.BM,c.BD]) : α :=
  M.cast (some .fp32) c.storage.float .real (M.cast (some .fp32) .real c.storage.float
    (if D.equal (a.normalizer (i.1,())) (M.literal (some .fp32) .real 0) then M.literal (some .fp32) .real 0
     else M.binary (some .fp32) .real .div (a.output i) (a.normalizer (i.1,()))))

theorem source_memory (M : Algebra α) (D : Dot α) (c : Config) (s : State α) (hBN : 0<c.BN) :
    ∃ final,FP.Structural.exec (backend M D) (kernel c) s=some final ∧
      final.mem=(outputMemory M D c s (finalValues M D c s)).mem := by
  by_cases hactive : rowStart c s<sequenceLength c s
  · exact whole_typed M D c s hBN hactive
  · obtain ⟨final,hf,hm⟩:=inactive_run M D c s (by omega)
    exact ⟨final,hf,by rw [inactive_outputMemory M D c s _ (by omega)];exact hm⟩

theorem readback (M : Algebra α) (D : Dot α) (c : Config) (s : State α) (a : Accumulator α c.BM c.BD)
    (i : TileIndex [c.BM,c.BD]) (hi : active c s i)
    (unique : ∀ j k,active c s j→active c s k→address c s j=address c s k→j=k) :
    (outputMemory M D c s a).mem c.out (address c s i)=
      .mk .real (ofFloat .real (outputValue M D c a i)) :=
  Scatter.masked_readback_active _ _ _ _ _ i hi unique

theorem frame (M : Algebra α) (D : Dot α) (c : Config) (s : State α) (a : Accumulator α c.BM c.BD)
    (r : RegionName) (off : Nat) (h : r≠c.out ∨ ∀ i,active c s i → off≠address c s i) :
    (outputMemory M D c s a).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right;intro i _ hi;exact ho i hi

#axiomsClean source_memory
#axiomsClean readback
#axiomsClean frame
#specNonCircular outputValue avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
