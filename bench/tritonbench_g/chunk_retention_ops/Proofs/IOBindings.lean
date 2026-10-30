import bench.tritonbench_g.chunk_retention_ops.Proofs.IOInputValues

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
open KernelNative
noncomputable section

/-- Every former raw Real input state supplies finite logical inputs. No new
input-value or typed-cell restriction is imposed by the public binding. -/
def inputsOfState (a : Call) (s : State ℝ) : Inputs ℝ a.shape :=
  fun port j=>(s.mem (inputs a.shape)[port].buf (inputOffset a port j)).read (inputs a.shape)[port].dtype

theorem inputsOfState_binding (a : Call) (s : State ℝ) :
    (realIO a).InputBinding (inputsOfState a s) s := by
  intro port j _
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port <;> rfl

theorem real_reads (a : Call) (xs : Inputs ℝ a.shape) (s : State ℝ)
    (binding : (realIO a).InputBinding xs s) :
    Reads a xs (fun r off=>(s.mem r off).read .real) := by
  intro port j enabled
  have h := binding port j enabled
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port <;> exact h

theorem native_reads (isa : ISA) (literal : ℝ→Word) (a : Call)
    (xs : Inputs Word a.shape) (s : State Term)
    (binding : (nativeIO isa literal a).InputBinding xs s) :
    Reads a xs (fun r off=>S.lower (sharedOps (wordOps isa)) literal ((s.mem r off).read .real)) := by
  intro port j enabled
  have h := binding port j enabled
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  all_goals exact congrArg (S.lower (sharedOps (wordOps isa)) literal) h

/-- Only actual read addresses are normalized to the former address-labelled
word memory; unrelated symbolic cells remain unconstrained. -/
theorem native_readEq (isa : ISA) (literal : ℝ→Word) (a : Call)
    (xs : Inputs Word a.shape) (s : State Term)
    (binding : (nativeIO isa literal a).InputBinding xs s) :
    ReadEq a s.mem (input (fun r off=>S.lower (sharedOps (wordOps isa)) literal ((s.mem r off).read .real))) := by
  intro port j enabled
  have h := binding port j enabled
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  all_goals
    change _=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.Term.input _ _ (S.lower (sharedOps (wordOps isa)) literal _) 
    change (s.mem _ _).read .real=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.Term.input _ _ _ at h
    dsimp only [nativeIO,layout] at h
    apply Eq.trans h
    exact congrArg (fun v=>VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.Term.input _ _
      (S.lower (sharedOps (wordOps isa)) literal v)) h.symm

#axiomsClean inputsOfState_binding
#axiomsClean real_reads
#axiomsClean native_reads
#axiomsClean native_readEq
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
