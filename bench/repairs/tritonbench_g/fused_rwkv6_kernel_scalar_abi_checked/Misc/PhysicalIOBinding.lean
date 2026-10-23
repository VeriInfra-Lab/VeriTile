import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.PhysicalIOBinding
open VeriTile Triton FP.Structural FusedRwkv6Kernel Interface
noncomputable section
attribute [local instance] Classical.propDecidable
variable {α : Type} [Inhabited α]

/-- The same reader supplies aliased logical coordinates; no copy or added
separation premise is introduced by the shaped IO interface. -/
def readerInputs (c : Shape) (m : FusedRwkv6Kernel.Memory α) : Interface.Inputs α c
  | ⟨0,_⟩,j => m .q (inputKey c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨1,_⟩,j => m .k (inputKey c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨2,_⟩,j => m .v (inputValue c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨3,_⟩,j => m .w (inputKey c j.1.val j.2.1.val j.2.2.1.val)
  | ⟨4,_⟩,j => m .u (j.1.val*c.K+j.2.1.val)
  | ⟨5,_⟩,j => m .h0 (stateAddress c j.1.val j.2.1.val j.2.2.1.val)

def initial (m : FusedRwkv6Kernel.Memory α) : State α where
  mem := Source.cells m
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem reader_binding (M : Algebra α) (c : Shape) (scale : α) (m : FusedRwkv6Kernel.Memory α) :
    (programIO M c scale).InputBinding (readerInputs c m) (initial m) := by
  intro i j _
  change Fin 6 at i
  fin_cases i <;> rfl

/-- Enclosing logical bounds remain available for every old Sizes-only shape,
including overlapping head pitches. Actual admitted physical capacities remain
in the unchanged original contract/byte theorems. -/
def bounds (c : Shape) (r : RegionName) : Nat :=
  if r="q" ∨ r="k" ∨ r="w" then c.Z*c.keyPitch+c.T*c.K
  else if r="v" then c.Z*c.valuePitch+c.T*c.V
  else if r="u" then c.H*c.K
  else if r="h0" ∨ r="ht" then c.Z*c.K*c.V
  else if r="o" then c.NK*c.Z*c.T*c.V else 0

theorem key_bound (c : Shape) (z : Fin c.Z) (t : Fin c.T) (e : Fin c.K) :
    inputKey c z.val t.val e.val<c.Z*c.keyPitch+c.T*c.K := by
  have hz := Nat.mul_le_mul_right c.keyPitch (Nat.le_of_lt z.isLt)
  have ht : t.val*c.K+e.val<c.T*c.K := by nlinarith [t.isLt,e.isLt]
  unfold inputKey
  omega

theorem value_bound (c : Shape) (z : Fin c.Z) (t : Fin c.T) (f : Fin c.V) :
    inputValue c z.val t.val f.val<c.Z*c.valuePitch+c.T*c.V := by
  have hz := Nat.mul_le_mul_right c.valuePitch (Nat.le_of_lt z.isLt)
  have ht : t.val*c.V+f.val<c.T*c.V := by nlinarith [t.isLt,f.isLt]
  unfold inputValue
  omega

theorem io_bounds (M : Algebra α) (c : Shape) (scale : α) (xs : Interface.Inputs α c) :
    (programIO M c scale).InBounds (bounds c) xs := by
  constructor
  · intro r outside
    simp only [programIO,List.mem_cons,List.not_mem_nil,or_false,not_or] at outside
    simp_all [bounds]
  constructor
  · intro i j _
    change Fin 6 at i
    fin_cases i
    · exact key_bound c j.1 j.2.1 j.2.2.1
    · exact key_bound c j.1 j.2.1 j.2.2.1
    · exact value_bound c j.1 j.2.1 j.2.2.1
    · exact key_bound c j.1 j.2.1 j.2.2.1
    · change j.1.val*c.K+j.2.1.val<c.H*c.K
      nlinarith [j.1.isLt,j.2.1.isLt]
    · exact state_address_bound c _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt
  constructor
  · intro o j _
    change Fin 2 at o
    fin_cases o
    · have h := partial_output c _ _ _ _ j.2.1.isLt j.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt
      exact (h.resolve_right (by simp)).2
    · exact state_address_bound c _ _ _ j.1.isLt j.2.1.isLt j.2.2.1.isLt
  · intro i
    exact Fin.elim0 i

def nativeInputs (c : Shape) (layout : Physical.Layout c) (memory : Bytes) : Interface.Inputs Native.Term c :=
  readerInputs c (FusedRwkv6Kernel.Native.entry (Physical.loadWords c layout memory))

theorem native_binding (p : Native.ReductionPlan) (binding : Native.ScalarBinding)
    (raw : PythonNumber) (c : Shape) (layout : Physical.Layout c) (memory : Bytes) :
    (programIO (Native.algebra p) c (Native.scalar binding raw)).InputBinding
      (nativeInputs c layout memory)
      (initial (FusedRwkv6Kernel.Native.entry (Physical.loadWords c layout memory))) :=
  reader_binding _ _ _ _

#axiomsClean reader_binding
#axiomsClean io_bounds
#axiomsClean native_binding
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.PhysicalIOBinding
