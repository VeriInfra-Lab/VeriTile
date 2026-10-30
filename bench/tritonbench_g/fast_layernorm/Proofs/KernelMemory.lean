import bench.tritonbench_g.fast_layernorm.Proofs.KernelReference
namespace VeriTile.Bench.Repairs.FastLayernorm.KernelMemory
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem masked_regs {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i <;> simp [hi,ih]
def update (r : RegionName) (o : Nat) (v : Cell α) (heap : RegionName → Nat → Cell α) :=
  fun q n => if q=r ∧ n=o then v else heap q n

def commit {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (heap : RegionName → Nat → Cell α) :=
  indices.foldl (fun mem i => if active i then update r (off i) (val i) mem else mem) heap

theorem fold_mem {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → Cell α)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (u : State α) :
    (indices.foldl (fun s i => if active i then s.write r (off i) (val i) else s) u).mem=
      commit r off val active indices u.mem := by
  induction indices generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,hi,ite_true,ite_false]
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl
    · rw [ih]
      simp [commit,List.foldl_cons,hi,State.write,update] <;> rfl

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

end
end VeriTile.Bench.Repairs.FastLayernorm.KernelMemory
