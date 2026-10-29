import bench.tritonbench_g.matmul_tma.Proofs.Layout
import bench.tritonbench_g.matmul_tma.Proofs.FPContract

/-! Whole single-program execution composed with the source wrapper metadata. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.Host
open VeriTile.Triton FP.Structural FPValue
noncomputable section
/-- This is the successful-allocation path: `s` is the state returned by
Torch randn/empty. No initial value is assumed for either empty output buffer. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (admit : Admission) (h : Shape) (f : Flags) (r : Regions) (cfg : Config)
    (s : State α) : Except String (Tensor × State α) := do
  let _ ← admit (request h f r cfg)
  match FP.Structural.exec (backend M D) (source (parameters h f r) f.half) s with
  | none => .error "source interpreter rejected the admitted program"
  | some t => .ok (returnedTensor h f r, t)

/-- Independent observable contract, using initial input memory and the chosen
primitive. It also frames the unused first output in the replacement branch. -/
def Returned {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (h : Shape) (f : Flags) (r : Regions) (s t : State α) : Prop :=
  Post (parameters h f r) f.half s t (outputValues M D (parameters h f r) f.half s)

theorem run_correct {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (admit : Admission) (h : Shape) (f : Flags) (r : Regions) (cfg : Config)
    (s : State α) (ready : admit (request h f r cfg) = .ok ()) :
    ∃ t, run M D admit h f r cfg s = .ok (returnedTensor h f r, t) ∧
      Returned M D h f r s t := by
  obtain ⟨t,he,hp⟩ := correctness M D (parameters h f r) f.half s (c_injective h f r)
  exact ⟨t, by simp [run,ready,he,Except.bind,bind], hp⟩

theorem setup_error {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (admit : Admission) (h : Shape) (f : Flags) (r : Regions) (cfg : Config)
    (s : State α) (message : String) (failed : admit (request h f r cfg) = .error message) :
    run M D admit h f r cfg s = .error message := by simp [run,failed,Except.bind,bind]

theorem replacement_framed {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (h : Shape) (f : Flags) (r : Regions) (a : FlatAlloc) (s t : State α)
    (binding : AllocationBinding h f r a) (half : f.half = Bool.true)
    (post : Returned M D h f r s t) : ∀ off, t.mem r.first off = s.mem r.first off := by
  intro off
  apply post.2 r.first off
  left
  simpa [parameters,returnedTensor,half,contiguous] using (binding.replacement_fresh half).2.2.symm

#axiomsClean run_correct
#axiomsClean replacement_framed
#axiomsClean setup_error
#specNonCircular Returned avoiding [run]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Host
