import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.GridResults

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

theorem implements_of_values (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (geometry : Geometry p)
    (f : Inputs α p batches visits→Outputs α p batches)
    (values : ∀ xs,Admissible p batches visits xs → ∀ s,
      (io a native encode observe p batches visits).InputBinding xs s →
      ∀ j : OutIndex p batches,live j →
      observe (KernelOnly.output a native p (prepared s (programOf j)) (laneOf j)
        (owner j.2.2.2.2.1.val))=f xs (0 : Fin 1) j) :
    (io a native encode observe p batches visits).ImplementsOn (Admissible p batches visits) f := by
  intro xs admitted bounds bounded s binding
  obtain ⟨safe,final,ran,results,frame⟩ := grid_results a native encode observe p batches visits geometry xs admitted
    (fun j=>f xs (0 : Fin 1) j) (values xs admitted) bounds bounded
    (PhysicalKernel.programs p batches) (fun id member=>(PhysicalKernel.programs_mem p batches id).mp member) s binding
  refine ⟨safe,final,ran,?_,?_⟩
  · intro port j active
    rcases port with ⟨port,limit⟩
    change port<1 at limit
    have equal : port=0 := by omega
    subst port
    exact results j active ((PhysicalKernel.programs_mem p batches (programOf j)).mpr (program_valid j))
  · intro region offset outside
    apply frame region offset
    by_cases other : region≠"out"
    · exact Or.inl other
    · have regionEq : region="out" := not_ne_iff.mp other
      subst region
      apply Or.inr
      intro j active _
      have out := outside.1 (0 : Fin 1) j active
      exact out.resolve_left (not_not.mpr rfl)

theorem real_implements (p : Parameters) (batches visits : Nat) (geometry : Geometry p) :
    (realIO p batches visits).ImplementsOn (Admissible p batches visits) (realFormula p batches visits) := by
  apply implements_of_values mathArithmetic Bool.false some (WithBot.unbotD 0) p batches visits geometry
  intro xs admitted s binding j active
  have value := real_output p batches visits xs (prepared s (programOf j)) binding
    j.1 j.2.1 j.2.2.1 rfl admitted (laneOf j) active
  exact congrArg (WithBot.unbotD 0) value

theorem native_implements (ops : PhysicalKernel.NativePrimitives) (p : Parameters)
    (batches visits : Nat) (geometry : Geometry p) :
    (nativeIO ops p batches visits).ImplementsOn (Admissible p batches visits)
      (nativeFormula ops p batches visits) := by
  apply implements_of_values ops.arithmetic Bool.true id id p batches visits geometry
  intro xs admitted s binding j _
  exact native_output ops p batches visits xs (prepared s (programOf j)) binding
    j.1 j.2.1 j.2.2.1 rfl admitted (laneOf j)

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
