import bench.repairs.tritonbench_g.decay_cumsum.Proofs.HostOutcome

namespace VeriTile.Bench.Repairs.DecayCumsum.Host
open VeriTile.Triton PackedStorage LogicalStorage
noncomputable section
variable {β γ : Type}

/-- Original caller bytes in logical row-major order. The generic codec can
read a lazy-negative flag; the public input contract fixes that flag to false. -/
def logicalData (h : Call) (tensors : RegionName → Target)
    (decode : Bool → (Fin 4 → β) → γ) (before : Copyback.Heap β) : Data γ :=
  fun r row time col => decode (tensors r).negative
    (fun byte => before ((tensors r).address (row*(h.T*h.DK)+time*h.DK+col)+byte.val))

/-- This ordinary API contract has only input-copy, logical-shape and fresh
allocation facts. In particular it supplies no grid result, output value,
source execution, numerical identity, or kernel address equation. -/
structure PreparationLaws (api : API β γ) (h : Call) (kind : Kind) (storage : Placement h kind)
    (tensors : RegionName → Target) (decode : Bool → (Fin 4 → β) → γ)
    (before : Copyback.Heap β) (live : Nat → Prop) : Prop where
  snapshots : ∀ packed, api.prepare=.ok packed →
    Snapshot h kind storage packed (logicalData h tensors decode before)
  reshapes : ∀ packed, api.prepare=.ok packed → ∀ r∈Roles.inputs kind,
    (tensors r).shape.prod=Logical.cells h
  fresh : ∀ r∈Roles.buffers kind, ∀ off, off<h.cells → ∀ byte : Fin 4,
    ¬live (address storage r off+byte.val)

structure Arguments where
  B : Int
  H : Int
  T : Int
  DK : Int
  BT : Int
  BK : Int

def validate (a : Arguments) : Except String Call :=
  if dims : 0≤a.B ∧ 0≤a.H ∧ 0≤a.T ∧ 0≤a.DK then
    if tiles : 0<a.BT ∧ 0<a.BK then
      .ok ⟨a.B.toNat,a.H.toNat,a.T.toNat,a.DK.toNat,a.BT.toNat,a.BK.toNat,by omega,by omega⟩
    else .error "tile dimensions must be positive"
  else .error "tensor dimensions must be nonnegative"

def checked (a : Arguments) (body : Call → Except (Failure β) (Returned β))
    (before : Copyback.Heap β) : Except (Failure β) (Returned β) :=
  match validate a with
  | .error message => .error ⟨.validation,message,before⟩
  | .ok h => body h

theorem validation_failure (a : Arguments) (body : Call → Except (Failure β) (Returned β))
    (before : Copyback.Heap β) (message : String) (bad : validate a=.error message) :
    checked a body before=.error ⟨.validation,message,before⟩ := by simp [checked,bad]

theorem negative_dimension (a : Arguments) (bad : a.B<0 ∨ a.H<0 ∨ a.T<0 ∨ a.DK<0) :
    validate a=.error "tensor dimensions must be nonnegative" := by
  have no : ¬(0≤a.B ∧ 0≤a.H ∧ 0≤a.T ∧ 0≤a.DK) := by omega
  simp [validate,no]

theorem invalid_tile (a : Arguments) (dims : 0≤a.B ∧ 0≤a.H ∧ 0≤a.T ∧ 0≤a.DK)
    (bad : a.BT≤0 ∨ a.BK≤0) : validate a=.error "tile dimensions must be positive" := by
  have no : ¬(0<a.BT ∧ 0<a.BK) := by omega
  simp [validate,dims,no]

#axiomsClean negative_dimension
#axiomsClean invalid_tile
end
end VeriTile.Bench.Repairs.DecayCumsum.Host
