import bench.tritonbench_g.fifth_order_sph_harmonics.IO
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOSafety
open VeriTile.Triton FP.Structural HarmonicIO
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

theorem schedule (M : Algebra α) (d : Grid.Direction) (p : Parameters)
    (bounds : RegionName → Nat)
    (safe : ∀ s : State α, TraceSafe M bounds (Grid.source d p) s) (s : State α) :
    ScheduleSafe bounds (launches M d p) s := by
  unfold launches
  generalize List.range (Grid.blocks p) = order
  induction order generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    constructor
    · cases d <;> exact safe _
    · intro t _
      exact ih t

theorem forward (M : Algebra α) (p : Parameters)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (vals : (forwardIO M p).Inputs) (bounds : RegionName → Nat)
    (inside : (forwardIO M p).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds ((forwardIO M p).schedule vals) s := by
  apply schedule M .forward p bounds
  intro t
  apply KernelSafety.forward M p bounds t fit
  · intro r hr c
    exact inside.2.1 (0 : Fin 1) (⟨r, hr⟩, c, ()) trivial
  · intro r hr c
    exact inside.2.2.1 (0 : Fin 1) (⟨r, hr⟩, c, ()) trivial

theorem backward (M : Algebra α) (p : Parameters)
    (fit : p.columnOffset + 11 ≤ p.outputWidth)
    (vals : (backwardIO M p).Inputs) (bounds : RegionName → Nat)
    (inside : (backwardIO M p).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds ((backwardIO M p).schedule vals) s := by
  apply schedule M .backward p bounds
  intro t
  apply KernelSafety.backward M p bounds t fit
  · intro r hr c
    exact inside.2.1 (0 : Fin 3) (⟨r, hr⟩, c, ()) trivial
  · intro r hr c
    exact inside.2.1 (1 : Fin 3) (⟨r, hr⟩, c, ()) trivial
  · intro r hr c
    exact inside.2.1 (2 : Fin 3) (⟨r, hr⟩, c, ()) trivial

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOSafety
