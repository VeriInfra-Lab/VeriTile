import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatCandidate

/-! Actual complete flat executions of both source versions, independent
exponent-bit classification and the full tagged flat-memory frame. Input bits
include infinities and NaNs; unspecified padding has an explicit compatibility
condition and is never replaced with zero. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open Parametric TypedBits BitClassification
noncomputable section
variable {α : Type} [Inhabited α]

def FlatPost (A : FlatAlloc) (classify : α → Bool) (p : Parameters) (bits : Nat → α)
    (initial final : State α) : Prop :=
  (∀ x, SourceCovers p (initial.pids 0) (initial.numPids 0) x →
    final.mem A.flat (A.addr p.output (x * p.outputStride)) = .mk .bool (classify (bits x))) ∧
  (∀ r o, ¬FlatWritten A (Written p (initial.pids 0) (initial.numPids 0)) r o →
    final.mem r o = initial.mem r o)

theorem flat_input_bits (A : FlatAlloc) (hd : A.Disjoint) (background flatUndef)
    (p : Parameters) (bits : Nat → α) (initial : State α)
    (hr : p.input ∈ A.regions)
    (hb : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hi : InputBits p bits initial) (x : Nat) (hx : x < p.size) :
    (flattenState A background flatUndef initial).mem A.flat
      (A.addr p.input (x * p.inputStride)) = .mk .real (bits x) := by
  rw [mem_addr A hd background flatUndef initial hr (hb x hx), hi x hx]
  rfl

theorem flat_post (A : FlatAlloc) (hd : A.Disjoint) (background flatUndef)
    (classify : α → Bool) (p : Parameters) (bits : Nat → α) (initial final : State α)
    (hr : p.output ∈ A.regions)
    (hb : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (hp : Post classify p bits initial final) :
    FlatPost A classify p bits (flattenState A background flatUndef initial)
      (flattenState A background flatUndef final) := by
  constructor
  · intro x hx
    rw [mem_addr A hd background flatUndef final hr
      (hb x (sourceCovers_lt p _ _ x hx)), hp.1 x hx]
    rfl
  · intro r o hmiss
    exact flatten_frame A background flatUndef initial final
      (Written p (initial.pids 0) (initial.numPids 0)) hp.2 r o hmiss

theorem pair32_flat_correct (A : FlatAlloc) (hd : A.Disjoint)
    (background : RegionName → Nat → Cell Float32Bits)
    (flatUndef : (d : TileDType) → RegionName → Nat → Value Float32Bits d)
    (rest : Algebra Float32Bits) (p : Parameters) (bits : Nat → Float32Bits)
    (initial : State Float32Bits) (hu : UndefCompatible A flatUndef initial .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hbits : InputBits p bits initial) :
    ∃ final,
      FP.Structural.exec (engine (backend32 rest)) (flattenComputeKernel A (original32 p))
        (flattenState A background flatUndef initial) = some final ∧
      FP.Structural.exec (engine (backend32 rest)) (flattenComputeKernel A (candidate32 p))
        (flattenState A background flatUndef initial) = some final ∧
      FlatPost A finite32 p bits (flattenState A background flatUndef initial) final := by
  obtain ⟨final, he, hp⟩ := original32_correct rest p bits initial hne hstride hbits
  refine ⟨flattenState A background flatUndef final, ?_, ?_, ?_⟩
  · exact original32_flat_exec A (engine (backend32 rest)) background flatUndef hd p
      initial final hu hinput houtput hin hout he
  · exact candidate32_flat_exec A (engine (backend32 rest)) background flatUndef hd p
      initial final hu hinput houtput hin hout ((whole32 p _ initial).symm.trans he)
  · exact flat_post A hd background flatUndef finite32 p bits initial final houtput hout hp

theorem pair64_flat_correct (A : FlatAlloc) (hd : A.Disjoint)
    (background : RegionName → Nat → Cell Float64Bits)
    (flatUndef : (d : TileDType) → RegionName → Nat → Value Float64Bits d)
    (rest : Algebra Float64Bits) (p : Parameters) (bits : Nat → Float64Bits)
    (initial : State Float64Bits) (hu : UndefCompatible A flatUndef initial .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output)
    (hne : p.input ≠ p.output) (hstride : 0 < p.outputStride)
    (hbits : InputBits p bits initial) :
    ∃ final,
      FP.Structural.exec (engine (backend64 rest)) (flattenComputeKernel A (original64 p))
        (flattenState A background flatUndef initial) = some final ∧
      FP.Structural.exec (engine (backend64 rest)) (flattenComputeKernel A (candidate64 p))
        (flattenState A background flatUndef initial) = some final ∧
      FlatPost A finite64 p bits (flattenState A background flatUndef initial) final := by
  obtain ⟨final, he, hp⟩ := original64_correct rest p bits initial hne hstride hbits
  refine ⟨flattenState A background flatUndef final, ?_, ?_, ?_⟩
  · exact original64_flat_exec A (engine (backend64 rest)) background flatUndef hd p
      initial final hu hinput houtput hin hout he
  · exact candidate64_flat_exec A (engine (backend64 rest)) background flatUndef hd p
      initial final hu hinput houtput hin hout ((whole64 p _ initial).symm.trans he)
  · exact flat_post A hd background flatUndef finite64 p bits initial final houtput hout hp

#axiomsClean flat_input_bits
#axiomsClean flat_post
#axiomsClean pair32_flat_correct
#axiomsClean pair64_flat_correct
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
