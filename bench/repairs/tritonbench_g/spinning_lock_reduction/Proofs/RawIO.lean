import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.IOBoundary

/-! Every old arbitrary raw memory is representable at the new finite boundary.
The original global publication condition implies the finite input domain;
no old exterior word, register scratch or active value is silently excluded. -/
namespace VeriTile.Bench.Repairs.SpinningLock.RawIO
open VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock

def inputs (h : HostArgs) (count : Nat) (stage : Phased.Stage) (mem : Memory V) :
    (IO.io V h count stage).Inputs where
  scalars := fun port => match port with
    | ⟨0,_⟩ => fun i => mem.partials i.val
    | ⟨1,_⟩ => fun i => mem.output i.val
    | ⟨n+2,hp⟩ => False.elim (by simp [IO.io] at hp)
  words := fun port => match port with
    | ⟨0,_⟩ => fun i => mem.locks i.val
    | ⟨n+1,hp⟩ => False.elim (by simp [IO.io] at hp)

theorem snapshot (h : HostArgs) (count : Nat) (stage : Phased.Stage) (mem : Memory V) :
    IO.inputs h count stage (inputs h count stage mem) = FiniteBoundary.snapshot h count mem := rfl

theorem admitted (b : Backend V) (h : HostArgs) (count : Nat) (stage : Phased.Stage)
    (mem : Memory V) (published : Wide.PublishedReady b h mem)
    (ready : stage = .consumers → KernelProof.ConsumerReady h mem) :
    IO.domain b.zero h count stage (inputs h count stage mem) := by
  change Formula.Domain b.zero h stage (FiniteBoundary.snapshot h count mem)
  refine ⟨FiniteBoundary.snapshot_published b h count mem published,?_⟩
  intro consumer p role
  exact ready consumer p.val p.isLt role

theorem binding (h : HostArgs) (count : Nat) (stage : Phased.Stage) (s : Concurrent.State V) :
    (IO.io V h count stage).InputBinding
      (inputs h count stage (ConcurrentSource.values s).mem) s := by
  constructor
  · rintro ⟨port,hp⟩ lane active
    have bound : port < 2 := hp
    interval_cases port <;> simp [IO.io,inputs,IO.partials,IO.output,
      Concurrent.Window.address,ConcurrentSource.values]
  · rintro ⟨port,hp⟩ lane active
    have he : port = 0 := by change port < 1 at hp; omega
    subst port
    simp [IO.io,inputs,IO.locks,Concurrent.Window.address,ConcurrentSource.values]

#axiomsClean admitted
#axiomsClean binding
end VeriTile.Bench.Repairs.SpinningLock.RawIO
