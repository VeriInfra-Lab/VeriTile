import bench.tritonbench_g.context_attn_bloom.Proofs.Preparation
import bench.tritonbench_g.context_attn_bloom.Proofs.RealStream
import bench.tritonbench_g.context_attn_bloom.Proofs.ABI

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.HostRun
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host (View3)
noncomputable section

/-- Chosen ordinary layout domain. Logical negative Q/K/V inputs are accepted
and normalized. The supplied output must already be a resolved view and use
separate storage; no arbitrary alias/layout/API extension is claimed. -/
structure LayoutDomain (a : Host.Arguments) : Prop where
  query : a.q.rowStride=a.q.heads*a.q.dims ∧ a.q.headStride=a.q.dims ∧ a.q.dimStride=1
  key : a.k.rowStride=a.k.heads*a.k.dims ∧ a.k.headStride=a.k.dims ∧ a.k.dimStride=1
  value : a.v.rowStride=a.v.heads*a.v.dims ∧ a.v.headStride=a.v.dims ∧ a.v.dimStride=1
  output : a.o.rowStride=a.o.heads*a.o.dims ∧ a.o.headStride=a.o.dims ∧ a.o.dimStride=1
  page : a.requestRowStride=a.requestColumns ∧ a.requestTokenStride=1
  normalized : a.q.negative=Bool.false ∧ a.k.negative=Bool.false ∧ a.v.negative=Bool.false ∧ a.o.negative=Bool.false

structure Returned (S : Type) where
  memory : S
  outputOwner : Nat
  descriptor : View3
  pythonReturn : Option Unit

def finish (a : Host.Arguments) (owner : Nat) (s : S) : Returned S :=
  ⟨s,owner,a.o,none⟩

def real (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime) (owner : Nat)
    (jobs : List (GridIndex (Host.grid a))) : Option (Returned BlockState) :=
  (RealStream.run a r jobs (Preparation.prepare api initial).real).map (finish a owner)

def native (a : Host.Arguments) (r : Regions) (d : Preparation.Data)
    (api : Preparation.LocalAPI r d) (initial : Preparation.Runtime) (owner : Nat)
    (jobs : List (GridIndex (Host.grid a))) : Option (Returned (State Native.Expr)) :=
  (FPSuccess.runGrid ⟨a,r,d.inputs⟩ Native.scalar Native.dots jobs
    (Preparation.prepare api initial).native).map (finish a owner)

def allPrograms (a : Host.Arguments) : List (GridIndex (Host.grid a)) := Finset.univ.toList

def CompleteSchedule (a : Host.Arguments) (jobs : List (GridIndex (Host.grid a))) : Prop :=
  jobs.Perm (allPrograms a)

theorem covered (a : Host.Arguments) (jobs : List (GridIndex (Host.grid a)))
    (complete : CompleteSchedule a jobs) (idx : GridIndex (Host.grid a)) : idx∈jobs :=
  complete.mem_iff.mpr (by simp [allPrograms])

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.HostRun
