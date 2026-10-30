import bench.tritonbench_g.block_sparse_attn.Proofs.IOExecution
import bench.tritonbench_g.block_sparse_attn.Proofs.GridGeometry

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

def observed (observe : β→α) (p : Parameters) (batches : Nat)
    (s : State (Replicas β)) (j : OutIndex p batches) : Cell α :=
  mapCell (fun x=>observe (x (owner j.2.2.2.2.1.val))) (s.mem "out" (outputOffset p batches j))

def OutsidePrograms (p : Parameters) (batches : Nat) (ids : List PhysicalKernel.Program)
    (region : RegionName) (offset : Nat) : Prop :=
  region≠"out" ∨ ∀ j : OutIndex p batches,live j → programOf j∈ids → offset≠outputOffset p batches j

theorem OutsidePrograms.mono (p : Parameters) (batches : Nat) (small big : List PhysicalKernel.Program)
    (subset : ∀ id∈small,id∈big) (region : RegionName) (offset : Nat)
    (outside : OutsidePrograms p batches big region offset) : OutsidePrograms p batches small region offset := by
  rcases outside with outside|outside
  · exact Or.inl outside
  · exact Or.inr (fun j hj member=>outside j hj (subset _ member))

theorem source_frame (p : Parameters) (batches : Nat) (id : PhysicalKernel.Program)
    (valid : id.Valid p batches) (s : State (Replicas β)) (values : Matrix β)
    (region : RegionName) (offset : Nat) (outside : OutsidePrograms p batches [id] region offset) :
    outputMemory p (prepared s id) values region offset=s.mem region offset := by
  apply outputMemory_frame
  rcases outside with outside|outside
  · exact Or.inl outside
  · apply Or.inr
    intro i active
    let j : OutIndex p batches := (⟨id.row,valid.1⟩,⟨id.head,valid.2.1⟩,⟨id.channel,valid.2.2⟩,i.1,i.2.1,PUnit.unit)
    have hj : live j := active
    have result := outside j hj (by simp [j,programOf])
    simpa only [j,outputOffset,KernelOnly.outputOffset,prepared,KernelOnly.batch,KernelOnly.head,
      KernelOnly.row,PhysicalKernel.Program.pids,batch,head] using result

/-- Complete finite launch-list composition. The mathematical/native value
connection supplied here is a proved helper, never a public input premise. -/
theorem grid_results (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (geometry : Geometry p)
    (xs : Inputs α p batches visits) (admitted : Admissible p batches visits xs)
    (f : OutIndex p batches→α)
    (values : ∀ s,(io a native encode observe p batches visits).InputBinding xs s →
      ∀ j : OutIndex p batches,live j →
      observe (KernelOnly.output a native p (prepared s (programOf j)) (laneOf j)
        (owner j.2.2.2.2.1.val))=f j)
    (bounds : RegionName→Nat) (bounded : (io a native encode observe p batches visits).InBounds bounds xs)
    (ids : List PhysicalKernel.Program) (valid : ∀ id∈ids,id.Valid p batches)
    (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s) :
    ScheduleSafe bounds (ids.map (invocation a native p)) s ∧
    ∃ final,runSchedule (ids.map (invocation a native p)) s=some final ∧
      (∀ j : OutIndex p batches,live j → programOf j∈ids → observed observe p batches final j=Cell.mk .real (f j)) ∧
      (∀ region offset,OutsidePrograms p batches ids region offset → final.mem region offset=s.mem region offset) := by
  classical
  induction ids generalizing s with
  | nil =>
    refine ⟨trivial,s,rfl,?_,?_⟩
    · intro j _ member; exact False.elim (List.not_mem_nil member)
    · intro _ _ _; rfl
  | cons id rest ih =>
    have idValid := valid id (List.mem_cons_self ..)
    obtain ⟨next,stepRun,memory⟩ := invocation_complete a native encode observe p batches visits xs s binding admitted id idValid
    have boundNext := inputs_preserved a native encode observe p batches visits xs s next binding admitted id idValid stepRun
    obtain ⟨tailSafe,final,tailRun,tailOutput,tailFrame⟩ :=
      ih (fun j hj=>valid j (List.mem_cons_of_mem _ hj)) next boundNext
    have ownOutput (j : OutIndex p batches) (hj : live j) (equal : programOf j=id) :
        observed observe p batches next j=Cell.mk .real (f j) := by
      subst id
      unfold observed
      rw [memory,← source_offset p batches (prepared s (programOf j)) j rfl,
        outputMemory_readback p (prepared s (programOf j)) _ geometry.2.1 (laneOf j) hj]
      change Cell.mk .real (observe (KernelOnly.output a native p (prepared s (programOf j))
        (laneOf j) (owner j.2.2.2.2.1.val)))=Cell.mk .real (f j)
      rw [values s binding j hj]
    refine ⟨?_,final,?_,?_,?_⟩
    · refine ⟨invocation_safe a native encode observe p batches visits xs s binding admitted id idValid bounds bounded,?_⟩
      intro t run
      rw [stepRun] at run
      cases run
      exact tailSafe
    · simp only [List.map_cons,runSchedule,stepRun,Option.bind_some,tailRun]
    · intro j hj member
      by_cases remaining : programOf j∈rest
      · exact tailOutput j hj remaining
      · have equal : programOf j=id := (List.mem_cons.mp member).resolve_right remaining
        have outside : OutsidePrograms p batches rest "out" (outputOffset p batches j) := by
          apply Or.inr
          intro k hk member same
          have eq := output_injective p batches geometry j k hj hk same
          exact remaining (eq.symm ▸ member)
        unfold observed
        rw [tailFrame _ _ outside]
        exact ownOutput j hj equal
    · intro region offset outside
      rw [tailFrame region offset (OutsidePrograms.mono p batches rest (id::rest)
        (fun j hj=>List.mem_cons_of_mem _ hj) region offset outside),memory]
      exact source_frame p batches id idValid s _ region offset
        (OutsidePrograms.mono p batches [id] (id::rest)
          (fun j hj=>by rcases List.mem_singleton.mp hj with rfl; exact List.mem_cons_self ..) region offset outside)

#axiomsClean grid_results
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
