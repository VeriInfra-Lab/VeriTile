import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.Interface

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

theorem native_implements (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (dst leftPtr rightPtr : BitVec 64) (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr) :
    (nativeIO rest platform op B Q K D dst leftPtr rightPtr code).Implements
      (nativeFormula rest platform op B Q K D dst leftPtr rightPtr code) :=
  addressed_implements _ op B Q K D positive covered (selectedRows B Q K)
    (selectedRows_complete B Q K) _ _

theorem reverse_equivalent {A : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (covered : K≤D) :
    (programIO M op B Q K D (programs B Q)).Equivalent (reversed M op B Q K D) := by
  apply ProgramIO.EquivalentOn.of_implementsOn
    (program_implements M op B Q K D positive covered (programs B Q) (fun b r _ => programs_mem B Q b r))
  exact program_implements M op B Q K D positive covered (programs B Q).reverse
    (fun b r _ => by simpa only [List.mem_reverse] using programs_mem B Q b r)

/-- Empty selection requires neither positive padded depth nor input values.
Its output footprint is empty, and the entire caller memory is framed. -/
theorem empty_implements {A : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (empty : B*Q*K=0) :
    (programIO M op B Q K D (selectedRows B Q K)).Implements
      (orderedTarget M op B Q K D (fun _ _ => id) id) := by
  intro v _ bounds _ initial _
  refine ⟨?_,initial,?_,?_,?_⟩
  · simp [programIO,selectedRows,empty,ScheduleSafe]
  · simp [programIO,selectedRows,empty,runSchedule]
  · intro p j active
    have hp : p.val<1 := p.isLt
    have pz : p=⟨0,by simp [programIO,memory,outputs]⟩ := Fin.ext (show p.val=0 by omega)
    subst p
    have hi : j.2.2.1.val<K := active
    have bound := Native.active_bound j.1.isLt j.2.1.isLt hi
    omega
  · exact fun _ _ _ => rfl

theorem old_inputs (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (dst leftPtr rightPtr : BitVec 64)
    (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr)
    (s : State Graph.Expr) (x y : Nat → Graph.Word) (h : RawKernel.InputWords op B Q K D s x y) :
    (nativeIO rest platform op B Q K D dst leftPtr rightPtr code).InputBinding (words op B Q K D x y) s := by
  intro p j active
  have hp : p.val<2 := p.isLt
  interval_cases hv : p.val
  · have eq : p=⟨0,by simp [nativeIO,addressedIO,memory,inputs]⟩ := Fin.ext hv
    subst p
    simpa only [nativeIO,addressedIO,memory,inputs,inputCoord,words,nativeEncode,
      mapValue,if_pos rfl] using h.1 j.1 j.2.1 j.2.2.1 active.2
  · have eq : p=⟨1,by simp [nativeIO,addressedIO,memory,inputs]⟩ := Fin.ext hv
    subst p
    have hr : op.readsRight=Bool.true := active.1.resolve_left (by change ¬(1:Nat)=0; decide)
    simpa only [nativeIO,addressedIO,memory,inputs,inputCoord,words,nativeEncode,mapValue,
      show ¬(1:Nat)=0 by decide,show ("right" : RegionName)≠"left" by decide,if_false] using
      h.2 j.1 j.2.1 j.2.2.1 hr active.2

theorem old_formula (rest : Algebra Graph.Expr) (platform : KernelCompiler.Platform)
    (op : Operation) (B Q K D : Nat) (dst leftPtr rightPtr : BitVec 64)
    (code : RawKernel.Selected platform op Q K D dst leftPtr rightPtr)
    (x y : Nat → Graph.Word) (b : Fin B) (r : Fin Q) (i : Fin D) :
    nativeFormula rest platform op B Q K D dst leftPtr rightPtr code (words op B Q K D x y)
      ⟨0,by simp [outputs]⟩ (b,r,i,PUnit.unit) =
      Graph.reference rest code.lowering.profile (platform.instructions code.object.primitives)
        op Q K b r 0 1 x y i := by
  unfold Graph.reference gridValue
  rw [value_cell]
  rfl

#axiomsClean native_implements
#axiomsClean reverse_equivalent
#axiomsClean empty_implements
#axiomsClean old_inputs
#axiomsClean old_formula
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
