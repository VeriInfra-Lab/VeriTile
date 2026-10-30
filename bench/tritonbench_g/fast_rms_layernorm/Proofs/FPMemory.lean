import bench.tritonbench_g.fast_rms_layernorm.Proofs.FPSource

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPMemory
open VeriTile.Triton FP.Structural FPSource
noncomputable section
variable {α : Type} [Inhabited α]

theorem commit_readback (r : RegionName) (off : TileIndex shape → Nat)
    (val : TileIndex shape → Cell α) (active : TileIndex shape → Prop) [DecidablePred active]
    (u : State α) (i : TileIndex shape) (hi : active i) (injective : Function.Injective off) :
    commit r off val active (TileShape.allIndices shape) u.mem r (off i)=val i := by
  have h:=State.masked_scatter_readback r off val active u i hi injective
  rw [fold_mem] at h
  exact h

theorem commit_frame {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (indices : List ι)
    (u : State α) (q : RegionName) (o : Nat)
    (outside : q≠r ∨ ∀ i∈indices,active i → o≠off i) :
    commit r off val active indices u.mem q o=u.mem q o := by
  have h:=State.masked_scatter_frame r off val active indices q o outside u
  rw [fold_mem] at h
  exact h

theorem row_injective (base B : Nat) : Function.Injective (fun i : TileIndex [B] => base+i.1.val) := by
  intro a b eq
  change base+a.1.val=base+b.1.val at eq
  have first : a.1=b.1 := Fin.ext (by omega)
  exact Prod.ext first (Subsingleton.elim _ _)

def raw (h : Call) (u : State α) (r : RegionName) (stride : Nat) (j : Fin h.block) : α :=
  (u.mem r (u.pids 0*stride+j.val)).read .real

def forwardFootprint (h : Call) (u : State α) (q : RegionName) (o : Nat) : Prop :=
  (q="y" ∧ ∃ j : Fin h.cols,o=u.pids 0*h.cols+j.val) ∨ (q="r" ∧ o=u.pids 0)

theorem forward_complete (M : Algebra α) (h : Call) (u : State α) :
    ∃ v, FP.Structural.exec (FP.ProfiledRewrite.engine M) (Kernels.forward h) u=some v ∧
      (∀ j : Fin h.cols, v.mem "y" (u.pids 0*h.cols+j.val)=
        FPReference.output M h (raw h u "x" h.cols) (raw h u "w" 0) ⟨j.val,j.isLt.trans_le h.covered⟩) ∧
      v.mem "r" (u.pids 0)=FPReference.savedInv M h (raw h u "x" h.cols) ∧
      (∀ q o,¬forwardFootprint h u q o → v.mem q o=u.mem q o) := by
  obtain ⟨v,run,mem⟩:=forward_memory M h u
  refine ⟨v,run,?_,?_,?_⟩
  · intro j
    rw [mem]
    let initial:=u.write "r" (u.pids 0) (FPReference.savedInv M h (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem _ _=_
    have readback := commit_readback "y" (fun i : TileIndex [h.block] => u.pids 0*h.cols+i.1.val)
      (fun i => FPReference.output M h (raw h u "x" h.cols) (raw h u "w" 0) i.1)
      (fun i => i.1.val<h.cols) initial (⟨j.val,j.isLt.trans_le h.covered⟩,PUnit.unit) j.isLt (row_injective _ _)
    have weightZero : raw h u "w" 0=(fun k : Fin h.block => (u.mem "w" k.val).read .real) := by
      funext k
      simp [raw]
    rw [weightZero] at readback ⊢
    exact readback
  · rw [mem]
    let initial:=u.write "r" (u.pids 0) (FPReference.savedInv M h (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem _ _=_
    rw [commit_frame _ _ _ _ _ initial _ _ (Or.inl (by decide))]
    exact State.write_same _ _ _ _
  · intro q o outside
    rw [mem]
    let initial:=u.write "r" (u.pids 0) (FPReference.savedInv M h (raw h u "x" h.cols))
    change commit "y" _ _ _ _ initial.mem q o=_
    rw [commit_frame]
    · apply State.write_other
      by_cases same : q="r"
      · right
        intro eq
        exact outside (Or.inr ⟨same,eq⟩)
      · exact Or.inl same
    · by_cases same : q="y"
      · right
        intro i _ active eq
        exact outside (Or.inl ⟨same,⟨⟨i.1.val,active⟩,eq⟩⟩)
      · exact Or.inl same

#axiomsClean forward_complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPMemory
