import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Host
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Machine
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Native

namespace VeriTile.Bench.Repairs.LogsumexpFwd.WrapperFPEquiv
open VeriTile.Triton FP.Structural Native
noncomputable section
open Classical

def library (torchLSE : (n : Nat) → (Fin n → Word) → Word) (cast64 : Word → BitVec 64) :
    Host.Library Word (BitVec 64) := ⟨BitVec.ofNat 32 0xff800000,torchLSE,cast64⟩

def profile (ops : Ops) (plans : Nat → Plans) (warps : Nat) (choice : warps∈Host.warpChoices) : Host.Profile Word :=
  ⟨warps,choice,fun j => Native.engine ops (plans j)⟩

/-- Source-precision whole-call result. The target expands the selected
max/sum trees and fixes each local FP32 primitive; it retains the wrapper's
Torch aggregation and explicit FP64 conversion as the same library calls.
The seven scalar equalities specialize an optional reference backend; they are
ordinary primitive-model premises, not deferred numerical experiments. -/
theorem logsumexp_fp32 (actual : Ops) (target : Target) (ports : LocalObligations actual target)
    (torchLSE : (n : Nat) → (Fin n → Word) → Word) (cast64 : Word → BitVec 64) (a : Arguments) (h : InputContract a) (scale : Word)
    (probes : List (Host.Profile Word)) (plans : Nat → Plans)
    (warps : Nat) (choice : warps∈Host.warpChoices) (s : State Word) :
    ∃ t out,Host.run (library torchLSE cast64) a scale probes (profile actual plans warps choice) s=some (t,out) ∧
      out.shape=a.input.shape.dropLast ∧ out.dtype=a.outputDType ∧ out.device=a.input.device ∧
      (∀ row,row<a.rows → out.value row=Host.castOutput (library torchLSE cast64) a
        (Host.target (library torchLSE cast64) a (fun j => Native.engine (reference target) (plans j)) scale
          (fun i => (s.mem "input" i).read .real) row)) ∧
      (∀ (r : RegionName) (i : Nat),(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,out,he,hs,hd,hdev,hv,hf⟩ := Host.complete (library torchLSE cast64) a h scale probes
    (profile actual plans warps choice) s
  rw [ops_eq actual target ports] at hv
  exact ⟨t,out,he,hs,hd,hdev,hv,hf⟩

#axiomsClean logsumexp_fp32
end
end VeriTile.Bench.Repairs.LogsumexpFwd.WrapperFPEquiv
