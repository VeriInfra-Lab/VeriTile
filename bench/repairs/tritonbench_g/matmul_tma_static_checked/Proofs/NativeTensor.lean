import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.PhysicalExecution

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.NativeTensor
open VeriTile.Triton FP.Structural FPValue Host TensorABI Native Bytes
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure TensorPost (I : Instructions) (profile : Profile) (read final : Read)
    (h : Shape) (f : Flags) (a b c : TensorABI.Tensor) : Prop where
  values : ∀ i : CIndex (params h f),
    logical (readWord final f.half) (width f.half) c i.1.val i.2.1.val =
      expected I profile (readWord read Bool.true) h f a b i
  frame : ∀ ptr, ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns)≤ptr → final ptr=read ptr
  inputA : ∀ (i : AIndex (params h f)) j, j<2 →
    final (physicalAddress 2 a i.1.val i.2.1.val+j)=read (physicalAddress 2 a i.1.val i.2.1.val+j)
  inputB : ∀ (i : BIndex (params h f)) j, j<2 →
    final (physicalAddress 2 b i.1.val i.2.1.val+j)=read (physicalAddress 2 b i.1.val i.2.1.val+j)

theorem tensor_post (I : Instructions) (profile : Profile) (read : Read) (h : Shape) (f : Flags)
    (a b c : TensorABI.Tensor) (abi : Binding h f a b c) (s t : State Word)
    (post : Post (params h f) f.half s t (expected I profile (readWord read Bool.true) h f a b)) :
    TensorPost I profile read (render read f.half c (h.rows*h.columns) t) h f a b c := by
  refine ⟨?_,frame read f.half c (h.rows*h.columns) t,?_,?_⟩
  · intro i
    rw [logical,c_address h f a b c abi i,
      render_read read f.half c (h.rows*h.columns) t (cOffset (params h f) i) (c_bound h f regions i)]
    have hp := post.1 i
    rw [regionC] at hp
    rw [hp]
    change pack f.half (value (outputDType f.half) (stored (outputDType f.half) _)) = _
    rw [read_stored]
    exact pack_native I f.half _
  · intro i j hj
    apply frame
    rw [a_address h f a b c abi i]
    have bound : aOffset (params h f) i<h.rows*h.inner := a_bound h f regions i
    have disjoint := abi.a_c
    unfold separated at disjoint
    omega
  · intro i j hj
    apply frame
    rw [b_address h f a b c abi i]
    have bound : bOffset (params h f) i<h.inner*h.columns := b_bound h f regions i
    have disjoint := abi.b_c
    unfold separated at disjoint
    omega

theorem correct (rest : Algebra Word) (I : Instructions) (profile : Profile) (read : Read)
    (h : Shape) (f : Flags) (dim : Addresses.Dimensions h) (a b c : TensorABI.Tensor) (abi : Binding h f a b c)
    (s : State Word) (input : Inputs (readWord read Bool.true) h a b s)
    (output : Bound read f.half c (h.rows*h.columns) s) :
    ∃ typed actual bytes,
      FP.Structural.exec (backend (Native.algebra rest I) (sourceDot I profile)) (kernel h f) s=some typed ∧
      PhysicalExecution.program (Native.algebra rest I) (sourceDot I profile) h f a b c (s,read)=(actual,bytes) ∧
      typed.mem=actual.mem ∧
      Post (params h f) f.half s typed (expected I profile (readWord read Bool.true) h f a b) ∧
      TensorPost I profile read bytes h f a b c := by
  obtain ⟨typed,actual,bytes,he,hb,hm,rendered,post⟩ :=
    ByteExecution.refines (Native.algebra rest I) (sourceDot I profile) h f dim c abi.c_window read s output
  rw [output_expected rest I profile (readWord read Bool.true) h f a b c abi s input] at post
  have physical : PhysicalExecution.program (Native.algebra rest I) (sourceDot I profile) h f a b c (s,read)=(actual,bytes) := by
    rw [PhysicalExecution.program_eq _ _ read h f dim a b c abi s input]
    exact hb
  refine ⟨typed,actual,bytes,he,physical,hm,post,?_⟩
  rw [rendered]
  exact tensor_post I profile read h f a b c abi s typed post

#axiomsClean tensor_post
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.NativeTensor
