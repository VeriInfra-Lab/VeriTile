import bench.tritonbench_g.block_sparse_attn.Proofs.SourceSafety
import bench.tritonbench_g.block_sparse_attn.Proofs.IOBounds
import bench.tritonbench_g.block_sparse_attn.Proofs.FormulaTransport

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

/-- The original launch preparation before binding its scalar ABI register. -/
def prepared (s : State (Replicas β)) (id : PhysicalKernel.Program) : State (Replicas β) :=
  {s with regs:=fun _ _ _=>none,pids:=id.pids,numPids:=fun _=>0}

theorem invocation_run (a : Arithmetic β) (native : Bool) (p : Parameters)
    (id : PhysicalKernel.Program) (s : State (Replicas β)) :
    (invocation a native p id).run s=execution a native p (prepared s id) := by
  cases native <;> rfl

theorem invocation_complete (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
    (admitted : Admissible p batches visits xs) (id : PhysicalKernel.Program)
    (valid : id.Valid p batches) :
    ∃ final,(invocation a native p id).run s=some final ∧
      final.mem=outputMemory p (prepared s id) (KernelOnly.output a native p (prepared s id)) := by
  have ints := metadata_reads a native encode observe p batches visits xs (prepared s id) binding
    ⟨id.row,valid.1⟩ ⟨id.head,valid.2.1⟩ ⟨id.channel,valid.2.2⟩ rfl admitted
  rw [invocation_run]
  exact complete_local a native p (prepared s id) ints

theorem invocation_safe (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (s : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
    (admitted : Admissible p batches visits xs) (id : PhysicalKernel.Program)
    (valid : id.Valid p batches) (bounds : RegionName→Nat)
    (bounded : (io a native encode observe p batches visits).InBounds bounds xs) :
    (invocation a native p id).Safe bounds s := by
  have ints := metadata_reads a native encode observe p batches visits xs (prepared s id) binding
    ⟨id.row,valid.1⟩ ⟨id.head,valid.2.1⟩ ⟨id.channel,valid.2.2⟩ rfl admitted
  have backed := access_bounds a native encode observe p batches visits xs (prepared s id) binding
    ⟨id.row,valid.1⟩ ⟨id.head,valid.2.1⟩ ⟨id.channel,valid.2.2⟩ rfl admitted bounds bounded
  have result := SourceSafety.source_safe a native p (prepared s id) ints bounds backed
  cases native <;> exact result

theorem inputs_preserved (a : Arithmetic β) (native : Bool) (encode : α→β) (observe : β→α)
    (p : Parameters) (batches visits : Nat) (xs : Inputs α p batches visits)
    (s t : State (Replicas β)) (binding : (io a native encode observe p batches visits).InputBinding xs s)
    (admitted : Admissible p batches visits xs) (id : PhysicalKernel.Program)
    (valid : id.Valid p batches) (ran : (invocation a native p id).run s=some t) :
    (io a native encode observe p batches visits).InputBinding xs t := by
  obtain ⟨final,known,memory⟩ := invocation_complete a native encode observe p batches visits xs s binding admitted id valid
  rw [known] at ran
  cases ran
  intro port j live
  have outside : (inputs p batches visits)[port].buf≠"out" := by
    rcases port with ⟨port,bound⟩
    change port<5 at bound
    interval_cases port <;> simp [inputs]
  dsimp only [io]
  have frame := outputMemory_frame p (prepared s id) (KernelOnly.output a native p (prepared s id))
    (inputs p batches visits)[port].buf (inputOffset p batches visits port xs j) (Or.inl outside)
  have cell := (congrFun (congrFun memory (inputs p batches visits)[port].buf)
    (inputOffset p batches visits port xs j)).trans frame
  exact (congrArg (fun cell : Cell (Replicas β)=>cell.read (inputs p batches visits)[port].dtype) cell).trans
    (binding port j live)

#axiomsClean invocation_complete
#axiomsClean invocation_safe
#axiomsClean inputs_preserved
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
