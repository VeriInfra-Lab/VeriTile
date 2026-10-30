import bench.tritonbench_g.int4_matmul.Proofs.Grid
namespace VeriTile.Bench.Repairs.Int4Matmul.Native
/-- Exactly the bounded instruction families present in the pinned SM90 profiles. -/
inductive ISA where
  | mma16x8 | wgmma64x8 | wgmma64x16 | wgmma64x32 | wgmma64x64 | wgmma64x128
  deriving DecidableEq, Repr
/-- The fixed-instruction projection consumes eight operands per side and one
live FP32 accumulator for an output component. It has no task dimensions,
addresses, memory, K loop, tile or output tensor in its numerical port. -/
structure Instructions where
  convert : Int → Word
  multiply : Word → Word → Word
  atomicAdd : Word → Word → Word
  mma8 : ISA → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def ops (p : Instructions) (isa : ISA) : Ops Word :=
  ⟨0,p.convert,p.multiply,p.atomicAdd,p.mma8 isa⟩

/-- Only local arithmetic and fixed ISA correspondences remain deferred.
Logical operand indexing and the complete source reduction are proved separately.
Binding physical register fragments to these logical instruction components remains
an ordinary per-compilation lowering/ABI boundary; finite offline profiles are
evidence, not a certified compiler translation. -/
structure Correspondence (actual reference : Instructions) : Prop where
  convert : ∀ z,actual.convert z=reference.convert z
  multiply : ∀ a b,actual.multiply a b=reference.multiply a b
  atomicAdd : ∀ a b,actual.atomicAdd a b=reference.atomicAdd a b
  mma8 : ∀ isa a b seed,actual.mma8 isa a b seed=reference.mma8 isa a b seed

theorem correspondence_eq (actual reference : Instructions) (h : Correspondence actual reference) :
    actual=reference := by
  cases actual;cases reference
  have hc := funext h.convert
  have hm := funext (fun a => funext (h.multiply a))
  have ha := funext (fun a => funext (h.atomicAdd a))
  have hd := funext (fun isa => funext (fun a => funext (fun b => funext (h.mma8 isa a b))))
  congr

theorem whole_congruence (s : Shape) (t : Tile) (ws : List Work) (initial : State Word)
    (actual reference : Instructions) (numeric : Correspondence actual reference) (isa : ISA) :
    call s t (ops actual isa) ws initial=call s t (ops reference isa) ws initial := by
  rw [correspondence_eq actual reference numeric]

theorem reference_consistent (reference : Instructions) : Correspondence reference reference :=
  ⟨fun _ => rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ _ _ _ => rfl⟩
#axiomsClean correspondence_eq
#axiomsClean whole_congruence
#axiomsClean reference_consistent
end VeriTile.Bench.Repairs.Int4Matmul.Native
