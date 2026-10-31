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

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPMemory
