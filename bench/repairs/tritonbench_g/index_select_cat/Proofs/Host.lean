import bench.repairs.tritonbench_g.index_select_cat.Proofs.Native

namespace VeriTile.Bench.Repairs.IndexSelectCat.Host
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent logical gather formula, with no grid, execution, or addresses. -/
def target {α : Type} (indices : Nat → Int) (source : Nat → Nat → α) (i j : Nat) : α :=
  source (indices i).toNat j

structure Result (α : Type) where
  output : Nat → α
  returnedObject : Nat
  device : VeriTile.Torch.Device

/-- Source and index roles are read-only and have disjoint backing from output.
The only state changed by successive typed programs is the output role. -/
def run {α : Type} [Inhabited α] (rest : Algebra α) (r : Request)
    (source output : Nat → α) (object : Nat) (incoming : VeriTile.Torch.Device) : Result α :=
  ⟨Program.run rest source (Grid.plans r.N r.D r.row) output,object,incoming⟩

theorem whole_call {α : Type} [Inhabited α] (rest : Algebra α) (r : Request)
    (checked : checkInputs r=Bool.true) (source output : Nat → α) (object : Nat)
    (incoming : VeriTile.Torch.Device) :
    let result := run rest r source output object incoming
    (∀ i j,i<r.N → j<r.D → result.output (i*r.D+j)=
      target r.indices (fun row col=>source (row*r.D+col)) i j) ∧
    (∀ q,r.N*r.D ≤ q → result.output q=output q) ∧
    (∀ i j,i<r.N → j<r.D → r.row i<r.M) ∧
    result.returnedObject=object ∧ result.device=incoming := by
  have h := (Native.checkInputs_iff r).mp checked
  refine ⟨?_,?_,?_,rfl,rfl⟩
  · intro i j hi hj
    exact Grid.readback rest r.N r.D r.row source output i j hi hj
  · intro q outside
    apply Grid.frame
    intro i j hi hj equal
    have bound := Native.flat_bound r.N r.D i j hi hj
    change q=i*r.D+j at equal
    omega
  · intro i j hi hj
    exact (Native.index_valid r h i hi (by omega)).2.1

theorem empty_call {α : Type} [Inhabited α] (rest : Algebra α) (r : Request)
    (source output : Nat → α) (object : Nat) (incoming : VeriTile.Torch.Device)
    (empty : r.N=0 ∨ r.D=0) : (run rest r source output object incoming).output=output := by
  simp only [run,Grid.empty r.N r.D r.row empty,Program.run,List.foldl_nil]

#axiomsClean whole_call
#axiomsClean empty_call
#specNonCircular target avoiding [run,Program.run,TypedGather.execution]
end
end VeriTile.Bench.Repairs.IndexSelectCat.Host
