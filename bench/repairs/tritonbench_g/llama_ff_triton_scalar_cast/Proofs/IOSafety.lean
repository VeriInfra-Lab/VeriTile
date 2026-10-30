import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.NativeSafety
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.IOBounds

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem schedule_safe (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (bounds : RegionName → Nat)
    (hb : (programIO native a g p).InBounds bounds xs)
    (pids : List Nat) (valid : ∀ pid∈pids, pid<Grid.count p)
    (s : State α) (bytes : ByteValues s) :
    ScheduleSafe bounds (pids.flatMap (jobs native a g p)) s := by
  induction pids generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    have hpid := valid pid (by simp)
    rw [List.flatMap_cons,scheduleSafe_append]
    refine ⟨SourceSafety.jobs_safe native a g p s pid bytes bounds
      (source_bounds native a g p xs bounds hb (by omega) _),?_⟩
    intro t ht
    rw [jobs_run] at ht
    obtain ⟨final,he,_,hf⟩ := Grid.single native a g p s s bytes (fun _ _ _ => rfl) pid
    rw [he] at ht
    have hft : final=t := Option.some.inj ht
    subst t
    apply ih (fun q hq => valid q (by simp [hq])) final
    apply byteValues_preserved s final _ bytes
    intro r o hr
    exact hf r o (Or.inl hr)

theorem program_safe (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (xs : Inputs α) (bounds : RegionName → Nat)
    (hb : (programIO native a g p).InBounds bounds xs) (s : State α) (bytes : ByteValues s) :
    ScheduleSafe bounds ((programIO native a g p).schedule xs) s :=
  schedule_safe native a g p xs bounds hb _ (fun _ h => List.mem_range.mp h) s bytes

#axiomsClean program_safe
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
