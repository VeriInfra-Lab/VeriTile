import VeriTile.Triton.Memory.RegionalKernelIO

/-! Regional kernel IO restricted by an explicit logical-input domain.
The domain constrains channel values, never execution states or a supplied
runner. Source safety, successful execution, outputs and the exact raw-cell
frame are the same obligations as in `RegionalImplements3D`. -/
namespace VeriTile.Triton.UKernelIO

/-- Regional source correctness for logical inputs in `domain`. Declared
window bounds imply safety of the actual source; execution and outputs do not
require a finite capacity assignment. -/
def RegionalImplementsOn3D (io : UKernelIO) (domain : io.Ctx → Prop)
    (f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier) : Prop :=
  ∀ (s : BlockState) (vals : io.Ctx), domain vals → io.RegionalInputs s vals →
    (∀ bounds, io.RegionalBounds bounds s vals →
      Kernel.TraceSafe bounds io.kernel.toAlgKernel s) ∧
    ∃ final, exec io.kernel.toAlgKernel s = some final ∧
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        (io.oty o).read final (io.obuf o)
          (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
          f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
      io.RegionalFrame s final vals

/-- Compose proved source safety and execution at the same logical domain.
Neither proof is stored as an unchecked field in the IO description. -/
theorem RegionalImplementsOn3D.intro (io : UKernelIO)
    {domain : io.Ctx → Prop}
    {f : Nat → Nat → Nat → io.Ctx → (o : Fin io.nOut) →
      Fin (io.oarity o) → (io.oty o).carrier}
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (vals : io.Ctx),
      domain vals → io.RegionalInputs s vals → io.RegionalBounds bounds s vals →
      Kernel.TraceSafe bounds io.kernel.toAlgKernel s)
    (hrun : ∀ (s : BlockState) (vals : io.Ctx), domain vals →
      io.RegionalInputs s vals →
      ∃ final, exec io.kernel.toAlgKernel s = some final ∧
        (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
          (io.oty o).read final (io.obuf o)
            (io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j) =
            f (s.pids 0) (s.pids 1) (s.pids 2) vals o j) ∧
        io.RegionalFrame s final vals) : io.RegionalImplementsOn3D domain f := by
  intro s vals hd hpins
  exact ⟨fun bounds hb => hts bounds s vals hd hpins hb, hrun s vals hd hpins⟩

end VeriTile.Triton.UKernelIO
