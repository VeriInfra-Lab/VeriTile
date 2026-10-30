import bench.tritonbench_g.chunk_retention.Spec

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket KernelStaging
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 1000000

/-- Finite logical reads, derived from the shared input binding. This helper
states no execution, dot product, or result assumption. -/
structure Reads (a : Call) (xs : Inputs α a.shape) (read : RegionName→Nat→α) : Prop where
  q : inputEnabled a 0 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.K),
    read "q" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)=xs 0 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  k : inputEnabled a 1 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.K),
    read "k" (h*(a.shape.T*a.shape.K)+t*a.shape.K+e)=xs 1 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  v : inputEnabled a 2 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.V),
    read "v" (h*(a.shape.T*a.shape.V)+t*a.shape.V+e)=xs 2 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  incoming : inputEnabled a 3 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.T) (he : e<a.shape.V),
    read "incoming" (h*(a.shape.T*a.shape.V)+t*a.shape.V+e)=xs 3 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  h0 : inputEnabled a 4 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.K) (he : e<a.shape.V),
    read "h0" (h*a.shape.K*a.shape.V+t*a.shape.V+e)=xs 4 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  terminal : inputEnabled a 5 → ∀ h t e (hh : h<a.shape.B*a.shape.H) (ht : t<a.shape.K) (he : e<a.shape.V),
    read "terminal" (h*a.shape.K*a.shape.V+t*a.shape.V+e)=xs 5 (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,())
  h : inputEnabled a 6 → ∀ h n e j (hh : h<a.shape.B*a.shape.H) (hn : n<a.shape.chunks)
    (he : e<a.shape.K) (hj : j<a.shape.V),
    read "h" (h*(a.shape.chunks*a.shape.K*a.shape.V)+n*a.shape.K*a.shape.V+e*a.shape.V+j)=
      xs 6 (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,())
  dh : inputEnabled a 7 → ∀ h n e j (hh : h<a.shape.B*a.shape.H) (hn : n<a.shape.chunks)
    (he : e<a.shape.K) (hj : j<a.shape.V),
    read "dh" (h*(a.shape.chunks*a.shape.K*a.shape.V)+n*a.shape.K*a.shape.V+e*a.shape.V+j)=
      xs 7 (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,())

theorem real_reads (a : Call) (p grid : Nat→Nat) (xs : Inputs ℝ a.shape) (s : State ℝ)
    (binding : (realIO a p grid).InputBinding xs s) :
    Reads a xs (fun r off=>(s.mem r off).read .real) := by
  constructor
  · intro enabled h t e hh ht he
    exact binding ⟨0,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h t e hh ht he
    exact binding ⟨1,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h t e hh ht he
    exact binding ⟨2,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h t e hh ht he
    exact binding ⟨3,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h t e hh ht he
    exact binding ⟨4,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h t e hh ht he
    exact binding ⟨5,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  · intro enabled h n e j hh hn he hj
    exact binding ⟨6,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled
  · intro enabled h n e j hh hn he hj
    exact binding ⟨7,by simp [realIO,io,layout,inputs]⟩ (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled

theorem native_reads (ops : Ops α) (literal : ℝ→α) (a : Call) (p grid : Nat→Nat)
    (xs : Inputs α a.shape) (s : State (Term α))
    (binding : (nativeIO ops literal a p grid).InputBinding xs s) :
    Reads a xs (fun r off=>lower ops literal ((s.mem r off).read .real)) := by
  constructor
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨0,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨1,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨2,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨3,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨4,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h t e hh ht he
    have hb := congrArg (lower ops literal) (binding ⟨5,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled)
    exact hb
  · intro enabled h n e j hh hn he hj
    have hb := congrArg (lower ops literal) (binding ⟨6,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled)
    exact hb
  · intro enabled h n e j hh hn he hj
    have hb := congrArg (lower ops literal) (binding ⟨7,by simp [nativeIO,layout,inputs]⟩ (⟨h,hh⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled)
    exact hb

#axiomsClean real_reads
#axiomsClean native_reads
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
