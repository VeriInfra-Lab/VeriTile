import bench.repairs.tritonbench_g.kldiv_compute.Kernels
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.FP

namespace VeriTile.Bench.Repairs.KldivCompute.FPEquiv
open VeriTile.Triton FP.Structural Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor World)
open Physical (Word)
noncomputable section

/-- Program/storage theorem has no numerical premise. The data carries the
actual original allocation, normalization results, source grid, copyback and
returned descriptor; raw bytes are read through that same descriptor/layout. -/
theorem whole_program {σ : Type} (neg : Word → Word) (rest : Algebra Word)
    (n : Floating.Numerics) (rt : Runtime Word σ) (lib : Host.Library neg rt)
    (mem : Physical.Memory σ) (physical : Physical.Library rt mem)
    (x y : Tensor) (initial : World Word σ)
    (_precisionX : x.dtype="torch.float32") (_precisionY : y.dtype="torch.float32")
    (hx : Physical.Backed mem initial x) (hy : Physical.Backed mem initial y) :
    Ensures (fun output final =>
      ∃ e : ReturnedData neg (Floating.backend rest n) rt x y initial output final,
      Physical.ReturnedBytes mem neg (Floating.backend rest n) x y output initial final ∧
      Physical.NativeData mem e)
      (Host.run (Floating.backend rest n) rt x y initial) := by
  have h := whole neg (Floating.backend rest n) rt lib x y initial hx.1 hy.1
  cases run : Host.run (Floating.backend rest n) rt x y initial with
  | error => trivial
  | ok output final =>
    rw [run] at h
    obtain ⟨e⟩ := h
    exact ⟨e,Physical.returned_bytes mem neg _ rt physical x y output initial final e hx hy,
      Physical.actual_native mem physical e⟩

/-- Only division, tl.log and multiplication are deferred against the named
FP32 target. Domain: all raw input/intermediate binary32 words; source order
and exceptional values remain explicit. No whole-output or execution axiom. -/
theorem whole_fp {σ : Type} (neg : Word → Word) (rest target : Algebra Word)
    (n : Floating.Numerics) (rt : Runtime Word σ) (lib : Host.Library neg rt)
    (mem : Physical.Memory σ) (physical : Physical.Library rt mem)
    (x y : Tensor) (initial : World Word σ)
    (precisionX : x.dtype="torch.float32") (precisionY : y.dtype="torch.float32")
    (hx : Physical.Backed mem initial x) (hy : Physical.Backed mem initial y)
    (numeric : Floating.Obligations n target) :
    Ensures (fun output final =>
      Physical.ReturnedBytes mem neg (Floating.backend rest n) x y output initial final ∧
      ∀ i, i<x.numel → mem.word final (mem.buffer output.region).device
        ((mem.buffer output.region).base+4*output.index i)=
          Floating.reference target (Physical.logical mem neg initial x i)
            (Physical.logical mem neg initial y i))
      (Host.run (Floating.backend rest n) rt x y initial) := by
  have h := whole_program neg rest n rt lib mem physical x y initial precisionX precisionY hx hy
  cases run : Host.run (Floating.backend rest n) rt x y initial with
  | error => trivial
  | ok output final =>
    rw [run] at h
    obtain ⟨e,bytes,native⟩ := h
    exact ⟨bytes,fun i hi => (bytes.1 i hi).trans (Floating.scalar_equiv rest n target numeric _ _)⟩

#axiomsClean whole_program
#axiomsClean whole_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.FPEquiv
