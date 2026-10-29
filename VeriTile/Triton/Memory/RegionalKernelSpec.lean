import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Triton.Float.Erasure

/-! Named mixed-channel Real IO for masked kernels. The typed source stays
visible; only the mathematical interpretation erases floating dtypes. Regional
contracts admit read-only physical aliases and certify the actual source trace.
-/
namespace VeriTile.Triton

/-- Two independently typed inputs and one typed output, with explicit
one-dimensional windows and masks. The source carries its storage precision;
channel types describe its mathematical observations. There is no private
scratch in this interface. -/
structure MixedMaskedKernelIO₂ where
  source : ComputeKernel
  projection : source.eraseDType.toAlgorithm? = .ok source.eraseDType.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  in1 : RegionName
  in2 : RegionName
  out : RegionName
  type1 : ChanTy
  type2 : ChanTy
  outType : ChanTy
  B1 : Nat
  B2 : Nat
  Bout : Nat
  read1 : Nat → Fin B1 → Nat
  read2 : Nat → Fin B2 → Nat
  write : Nat → Fin Bout → Nat
  mask1 : Nat → Fin B1 → Prop
  mask2 : Nat → Fin B2 → Prop
  outMask : Nat → Fin Bout → Prop

namespace MixedMaskedKernelIO₂

/-- The common regional IO core supplies execution, access safety, typed
input/output binding and the exact memory frame. -/
def toU (io : MixedMaskedKernelIO₂) : UKernelIO where
  kernel := io.source.eraseDType
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.in1,io.in2,io.out]
  ity := Fin.cases io.type1 (fun _ => io.type2)
  iarity := Fin.cases io.B1 (fun _ => io.B2)
  ibuf := Fin.cases io.in1 (fun _ => io.in2)
  oty := fun _ => io.outType
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := Fin.cases (fun _ pid _ _ => io.read1 pid) (fun _ _ pid _ _ => io.read2 pid)
  imask := Fin.cases (fun _ pid _ _ => io.mask1 pid) (fun _ _ pid _ _ => io.mask2 pid)
  owin := fun _ _ pid _ _ => io.write pid
  omask := fun _ _ pid _ _ => io.outMask pid
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- The public mathematical formula consumes logical input arrays, never a
memory state. Source execution and access safety come from the shared core. -/
def Implements (io : MixedMaskedKernelIO₂)
    (f : (Fin io.B1 → io.type1.carrier) → (Fin io.B2 → io.type2.carrier) →
      Fin io.Bout → io.outType.carrier) : Prop :=
  io.toU.RegionalImplements (fun _ _ vals _ => f (vals (0 : Fin 2)) (vals (1 : Fin 2)))

@[inherit_doc] scoped infix:25 " ⊨ " => Implements

/-- Reuse established trace-safety and regional execution proofs without
introducing physical separation, undefined-value or logical-input premises. -/
theorem Implements.intro (io : MixedMaskedKernelIO₂)
    {f : (Fin io.B1 → io.type1.carrier) → (Fin io.B2 → io.type2.carrier) →
      Fin io.Bout → io.outType.carrier}
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ j, io.mask1 (s.pids 0) j → io.read1 (s.pids 0) j < bounds io.in1) →
      (∀ j, io.mask2 (s.pids 0) j → io.read2 (s.pids 0) j < bounds io.in2) →
      (∀ j, io.outMask (s.pids 0) j → io.write (s.pids 0) j < bounds io.out) →
      Kernel.TraceSafe bounds io.source.eraseDType.toAlgKernel s)
    (hrun : ∀ (s : BlockState)
      (xs : Fin io.B1 → io.type1.carrier) (ys : Fin io.B2 → io.type2.carrier),
      (∀ j, io.mask1 (s.pids 0) j →
        io.type1.read s io.in1 (io.read1 (s.pids 0) j) = xs j) →
      (∀ j, io.mask2 (s.pids 0) j →
        io.type2.read s io.in2 (io.read2 (s.pids 0) j) = ys j) →
      ∃ t, exec io.source.eraseDType.toAlgKernel s = some t ∧
        (∀ j, io.outMask (s.pids 0) j →
          io.outType.read t io.out (io.write (s.pids 0) j) = f xs ys j) ∧
        (∀ r address, (r ≠ io.out ∨ ∀ j, io.outMask (s.pids 0) j →
          address ≠ io.write (s.pids 0) j) → t.mem r address = s.mem r address)) :
    io.Implements f := by
  unfold Implements UKernelIO.RegionalImplements
  apply UKernelIO.RegionalImplements3D.intro io.toU
  · intro bounds s vals _ hi ho _
    exact hts bounds s (fun j hj => hi (0 : Fin 2) j hj) (fun j hj => hi (1 : Fin 2) j hj)
      (fun j hj => ho (0 : Fin 1) j hj)
  · intro s vals hpins
    obtain ⟨t,ran,values,frame⟩ := hrun s (vals (0 : Fin 2)) (vals (1 : Fin 2))
      (fun j hj => hpins (0 : Fin 2) j hj) (fun j hj => hpins (1 : Fin 2) j hj)
    refine ⟨t,ran,fun _ j hj => values j hj,?_⟩
    intro r address outside _
    apply frame r address
    by_cases hr : r=io.out
    · right
      intro j hj
      exact (outside (0 : Fin 1) j hj).resolve_left (not_not.mpr hr)
    · exact Or.inl hr

end MixedMaskedKernelIO₂
end VeriTile.Triton
