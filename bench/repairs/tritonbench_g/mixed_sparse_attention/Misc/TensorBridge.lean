import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOScope
import bench.repairs.tritonbench_g.mixed_sparse_attention.Correct
import bench.repairs.tritonbench_g.mixed_sparse_attention.Misc.NativeCorrect
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelFrames

namespace VeriTile.Bench.Repairs.MixedSparseAttention.TensorBridge
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelGrid KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

section
open KernelReal
theorem kernel_real {Store Byte : Type} (api : Memory.API ℝ Store Byte)
    (c : Config) (request : Request) (layout : Reify.Layout c) (b : Buffers request)
    (integer : ℝ → Int) (store : Store)
    (accepted : Admission c request (observed c request (api.read store) 0 integer))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    let s:=initial c request layout (api.read store) integer
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) c (grid c) work s=some final ∧
      Result c request layout (api.read store) integer final ∧
      Shared.TensorImage api (placement b accepted s) (fun _ _=>observe) final store ∧
      (∀ q : Raw.Coord c,live c s q → api.read
        (Shared.commit api (placement b accepted s) (fun _ _=>observe) final store)
        ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q)))=
          target c request layout (api.read store) integer q) :=
  KernelReal.complete api c request layout b integer store accepted work order

#axiomsClean kernel_real
theorem kernel_real_source (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → ℝ) (integer : ℝ → Int)
    (accepted : Admission c request (observed c request read 0 integer))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    ∃ final,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel) c (grid c) work
      (initial c request layout read integer)=some final ∧ Result c request layout read integer final :=
  KernelReal.source c request layout read integer accepted work order

#axiomsClean kernel_real_source
end
noncomputable abbrev kernel_real_target := KernelReal.target

section
open KernelNative
theorem kernel_native {Store Byte : Type} (api : Memory.API Word Store Byte)
    (platform : KernelCompiler.Platform) (c : Config) (request : Request) (layout : Reify.Layout c)
    (b : Buffers request) (binding : KernelCompiler.Binding platform c request)
    (conversion : KernelCompiler.ScaleConversion) (argument : KernelCompiler.ScaleArgument)
    (_launchScale : KernelCompiler.LaunchScale conversion argument request) (store : Store)
    (accepted : Admission c request (observed c request (api.read store) 0 BitVec.toInt))
    (work : List Program) (order : work.Perm (programs (grid c))) :
    let I:=platform.instructions binding.object
    let hbm : 0<c.BM := by rw [accepted.schema.selected.2.1];decide
    let s:=initial c request layout (api.read store)
    ∃ final,runQuoted binding.decoded.source (runtime request) I c work s=some final ∧
      Result I c hbm request layout (api.read store) final ∧
      Shared.TensorImage api (placement b accepted s) (observe I c) final store ∧
      (∀ q : Raw.Coord c,live c s q →
        (Torch.FP32.logicalValue (fun a=>⟨api.read (Shared.commit api (placement b accepted s)
          (observe I c) final store) a⟩) (b.tensor .output) (fpLaws b accepted .output rfl) (Raw.offset q)).bits=
          target I c hbm request layout (api.read store) q) :=
  KernelNative.complete api platform c request layout b binding conversion argument _launchScale store accepted work order

#axiomsClean kernel_native
theorem kernel_native_inputs (I : Native.CompareISA Word) (bits : ∀ w,I.bits w=w)
    (c : Config) (request : Request) (layout : Reify.Layout c) (b : Buffers request) (read : Nat → Word)
    (accepted : Admission c request (observed c request read 0 BitVec.toInt))
    (r : Role) (hr : r∈[Role.query,.key,.value]) (i : Nat) (hi : i<r.size c) (owner : Fin 4) :
    (Native.lower (((initial c request layout read).mem (r.region c) i).read .real owner)).eval I.toISA=
      (Torch.FP32.logicalValue (fun a=>⟨read a⟩) (b.tensor r) (fpLaws b accepted r (by
        simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
        rcases hr with rfl|rfl|rfl <;> rfl)) i).bits :=
  KernelNative.input_words I bits c request layout b read accepted r hr i hi owner

#axiomsClean kernel_native_inputs
end

section
open KernelFrames
theorem kernel_input_frame {α β Store Byte : Type} [Inhabited α]
    (api : Memory.API β Store Byte) (b : Buffers request) (accepted : Admission c request values)
    (s final : State α) (observe : RegionName → Nat → Cell α → β) (store : Store)
    (r : Role) (hr : r∈inputRoles) (i : Nat) (hi : i<r.size c) (j : Fin 4) :
    api.byte (Shared.commit api (placement b accepted s) observe final store)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)+j.val)=
    api.byte store ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)+j.val) :=
  KernelFrames.input_bytes api b accepted s final observe store r hr i hi j

#axiomsClean kernel_input_frame
theorem kernel_inactive_frame {α β Store Byte : Type} [Inhabited α]
    (api : Memory.API β Store Byte) (b : Buffers request) (accepted : Admission c request values)
    (s final : State α) (observe : RegionName → Nat → Cell α → β) (store : Store)
    (q : Raw.Coord c) (hq : ¬live c s q) (j : Fin 4) :
    api.byte (Shared.commit api (placement b accepted s) observe final store)
      ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q))+j.val)=
    api.byte store ((b.tensor .output).storage.address ((b.tensor .output).physicalIndex (Raw.offset q))+j.val) :=
  KernelFrames.inactive_bytes api b accepted s final observe store q hq j

#axiomsClean kernel_inactive_frame
theorem kernel_empty {α Store Byte : Type} [Inhabited α]
    (M : Program → Algebra α) (D : Program → Dot α) (c : Config) (request : Request)
    (accepted : EmptyAdmission c request) (s : State α) (work : List Program)
    (order : work.Perm (programs (grid c))) (api : Memory.API β Store Byte)
    (observe : RegionName → Nat → Cell α → β) (store : Store) :
    runPrograms M D c (grid c) work s=some s ∧
      commitSource api c request s observe s store=store ∧
      (∀ byte,api.byte (commitSource api c request s observe s store) byte=api.byte store byte) :=
  KernelFrames.empty M D c request accepted s work order api observe store

#axiomsClean kernel_empty
end
end
end VeriTile.Bench.Repairs.MixedSparseAttention.TensorBridge
