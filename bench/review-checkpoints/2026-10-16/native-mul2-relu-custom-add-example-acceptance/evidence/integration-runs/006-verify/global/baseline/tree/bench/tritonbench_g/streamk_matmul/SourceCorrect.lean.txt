import bench.tritonbench_g.streamk_matmul.Proofs.Software

/-! Full pinned source contracts under the explicit SC lane serialization.
Every finite schedule is covered, including blocked and faulted prefixes.
There is no assumed ready lock, completion, zero input, numerical identity,
single-block restriction, or supplied final value. A GPU refinement and an
unconditional termination theorem are not established by this source model. -/
namespace VeriTile.Bench.TritonBenchG.StreamK.SourceCorrect
open VeriTile.Triton

def ValueContract (source : List Command) (grid : Nat) (c : Config)
 (b : Backend V) (schedule : List Nat) (mem : Memory V) : Prop :=
 let events:=controlTrace source c grid schedule mem.locks
 let reference:=replay c events (referenceInitial mem.locks)
 ∃ final, run c b grid schedule (initial source b mem)= (events,final) ∧
   (∀ address, final.mem.data address=denote b mem.data (reference.data address)) ∧
   final.mem.locks=reference.locks ∧
   (∀ p name lane,final.regs p name lane=denote b mem.data (reference.regs p name lane)) ∧
   (∀ address,(∀ e∈events,writeAddress e≠some address)→final.mem.data address=mem.data address) ∧
   (∀ address,(∀ e∈events,lockWriteAddress e≠some address)→final.mem.locks address=mem.locks address)

theorem every_source_contract (source : List Command) (grid : Nat) (c : Config)
 (b : Backend V) (schedule : List Nat) (mem : Memory V) :
 ValueContract source grid c b schedule mem := by
 have ht:=trace_independent source c b grid schedule mem
 have hr:=run_related c b mem.data grid schedule _ _ (related_initial source b mem)
 rw [ht] at hr
 refine ⟨(run c b grid schedule (initial source b mem)).2,?_,hr.data,hr.locks,hr.regs,?_,?_⟩
 · exact (Prod.ext ht rfl)
 · intro address ha
   rw [hr.data,replay_frame c _ _ address ha]
   rfl
 · intro address ha
   rw [hr.locks,replay_lock_frame c _ _ address ha]
   rfl

specification first_wave_source_backend (c : Config) (b : Backend V)
 (schedule : List Nat) (mem : Memory V) :
 ValueContract originalSource.first c.firstGrid c b schedule mem :=
 every_source_contract _ _ c b schedule mem

specification full_tiles_source_backend (c : Config) (b : Backend V)
 (schedule : List Nat) (mem : Memory V) :
 ValueContract originalSource.fullWave c.fullGrid c b schedule mem :=
 every_source_contract _ _ c b schedule mem

specification first_wave_source_fp32 (c : Config) (dot : Dot32)
 (schedule : List Nat) (mem : Memory Word) :
 ValueContract originalSource.first c.firstGrid c (fp32 dot) schedule mem :=
 first_wave_source_backend c (fp32 dot) schedule mem

specification full_tiles_source_fp32 (c : Config) (dot : Dot32)
 (schedule : List Nat) (mem : Memory Word) :
 ValueContract originalSource.fullWave c.fullGrid c (fp32 dot) schedule mem :=
 full_tiles_source_backend c (fp32 dot) schedule mem

def LaunchContract (c : Config) (b : Backend V)
 (first second : List Nat) (mem : Memory V) : Prop :=
 match hostRun c b first second mem,hostReference c first second mem.locks with
 | none,none => True
 | some s,some r => Related b mem.data s r
 | _,_ => False

specification ordered_launches_source_backend (c : Config) (b : Backend V)
 (first second : List Nat) (mem : Memory V) :
 LaunchContract c b first second mem := host_value_contract c b first second mem

specification ordered_launches_source_software (h : HostInput) (bases : Region→Address)
 (first second : List Nat) (mem : Memory Word) :
 LaunchContract (hostConfig h bases) software first second mem :=
 ordered_launches_source_backend (hostConfig h bases) software first second mem

specification host_allocated_locks_source_software (h : HostInput) (bases : Region→Address)
 (first second : List Nat) (mem : Memory Word) :
 LaunchContract (hostConfig h bases) software first second (allocateLocks h bases mem) :=
 ordered_launches_source_backend (hostConfig h bases) software first second (allocateLocks h bases mem)

#axiomsClean host_allocated_locks_source_software
#axiomsClean every_source_contract
#axiomsClean first_wave_source_backend
#axiomsClean full_tiles_source_backend
#axiomsClean first_wave_source_fp32
#axiomsClean full_tiles_source_fp32
#axiomsClean ordered_launches_source_backend
#axiomsClean ordered_launches_source_software
#specNonCircular ValueContract avoiding [first_wave_source_backend,full_tiles_source_backend]
end VeriTile.Bench.TritonBenchG.StreamK.SourceCorrect
