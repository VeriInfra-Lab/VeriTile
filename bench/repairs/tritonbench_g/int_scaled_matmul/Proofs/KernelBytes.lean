import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.IntScaledMatmul.KernelBytes
open VeriTile.Triton FP.Structural
open Revision001 Contract TensorBinding InputAdapter FiniteExecution FiniteIndex
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def suffix (scaled : Bool) (undefined : Word → BitVec 32)
    (a b : TensorBinding.Tensor) (p : Contract.Config) (s : State Word) : Suffix (launch a b p) :=
  if scaled then CheckedNative.suffix (KernelModel.ports undefined) (launch a b p) scales s
  else fun carry i => carry i

/-- Physical execution of the prepared grid, retaining each actual four-byte
store beside its finite native logical update. No wrapper is executed. -/
def run (scaled : Bool) (undefined : Word → BitVec 32) (a b c : TensorBinding.Tensor)
    (p : Contract.Config) (initial : ByteExecution.Machine) : Option ByteExecution.Machine :=
  if Empty a.metadata b.metadata then some initial
  else ByteExecution.sequence (ByteExecution.one (launch a b p) c scaled (suffix scaled undefined a b p))
    (List.range (count (rows a.metadata) (columns b.metadata) p)) initial

theorem run_render (scaled : Bool) (undefined : Word → BitVec 32)
    (a b c scale : TensorBinding.Tensor) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (read : Read) (initial : State Word) (bound : OutputBytes.Bound read c initial) :
    run scaled undefined a b c p (initial,read)=
      (KernelExecution.native scaled undefined a b p initial).map
        (fun final => (final,OutputBytes.render read c final)) := by
  by_cases empty : Empty a.metadata b.metadata
  · simp only [run,KernelExecution.native,if_pos empty,Option.map_some,
      OutputBytes.render_initial read c initial bound]
  · have env := (execution_of_nonempty scaled a.metadata b.metadata c.metadata scale.metadata p h empty).1
    have seq := ByteExecution.sequence_render (launch a b p) c scaled (suffix scaled undefined a b p)
      h.config env rfl (ByteHost.size_eq scaled a b c scale p h) read
      (List.range (count (rows a.metadata) (columns b.metadata) p))
      (fun _ hi => List.mem_range.mp hi) initial
    rw [OutputBytes.render_initial read c initial bound] at seq
    simp only [run,KernelExecution.native,if_neg empty,seq]
    cases scaled <;> rfl

theorem expected_range (scaled : Bool) (undefined : Word → BitVec 32) (read : Read)
    (a b scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b) (i : Index a b) :
    (BitVec.ofInt 32 (KernelExecution.expected scaled undefined read a b scale la lb i)).toInt=
      KernelExecution.expected scaled undefined read a b scale la lb i := by
  cases scaled <;> simp only [KernelExecution.expected,Bool.false_eq_true,↓reduceIte,
    LogicalInputs.integerExpected,BitVec.ofInt_toInt]

theorem physical_post (scaled : Bool) (undefined : Word → BitVec 32)
    (read : Read) (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (lc : Laws .int32 c) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial final : State Word)
    (post : KernelExecution.Post a b initial final (KernelExecution.expected scaled undefined read a b scale la lb)) :
    ByteHost.Post read (OutputBytes.render read c final) a b c
      (KernelExecution.expected scaled undefined read a b scale la lb) :=
  ByteHost.post scaled a b c scale lc p h read initial final _
    ⟨output_metadata scaled a b c scale p h,post⟩ (expected_range scaled undefined read a b scale la lb)

theorem complete (rest : Algebra Word) (scaled : Bool) (undefined : Word → BitVec 32)
    (read : Read) (a b c scale : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (lc : Laws .int32 c) (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (inputs : Inputs read a b scale scaled initial)
    (output : OutputBytes.Bound read c initial) :
    ∃ original lowered bytes,
      KernelExecution.source rest scaled undefined a b p initial=some original ∧
      KernelExecution.native scaled undefined a b p initial=some lowered ∧
      run scaled undefined a b c p (initial,read)=some (lowered,bytes) ∧
      original.mem=lowered.mem ∧
      KernelExecution.Post a b initial original (KernelExecution.expected scaled undefined read a b scale la lb) ∧
      ByteHost.Post read bytes a b c (KernelExecution.expected scaled undefined read a b scale la lb) := by
  obtain ⟨original,lowered,hs,hn,same,post⟩ :=
    KernelExecution.equivalent rest scaled undefined read a b c scale la lb ls p h initial inputs
  refine ⟨original,lowered,OutputBytes.render read c lowered,hs,hn,?_,same,post,?_⟩
  · rw [run_render scaled undefined a b c scale p h read initial output,hn]
    rfl
  · apply physical_post scaled undefined read a b c scale la lb lc p h initial lowered
    simpa only [KernelExecution.Post,same] using post

export ByteHost (output_addresses input_frame)
export TensorBinding (pointer_index direct_int8 direct_fp32)

#axiomsClean run_render
#axiomsClean expected_range
#axiomsClean physical_post
#axiomsClean complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul.KernelBytes
