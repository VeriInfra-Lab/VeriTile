import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPComplete
import bench.tritonbench_g.int4_matmul.Proofs.FPMixed
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Actual single-to-half rounding; half values occupy the low sixteen bits.
Only same-format carrier conversions are identities. -/
def software (rest : Algebra Word) : Algebra Word :=
  {FP.Software.scalarFP32 rest .ieeeValue with
    cast := fun precision src dst x =>
      match src,dst with
      | .fp32,.fp16 => QuantizeKvTransform.MixedBits.narrow .ieeeValue x
      | .fp32,.real | .real,.fp32 => x
      | src,dst => (FP.Software.scalarFP32 rest .ieeeValue).cast precision src dst x}

/-- An explicit software profile: binary16 operands widened exactly to fp32,
rounded multiplication and ascending-lane rounded addition, positive-zero seed.
Both tf32 flags are retained by the source theorem; this software interpretation
chooses the same schedule for both. It is not a native tensor-core claim. -/
def sequentialDot : Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b =>
    Int4Matmul.FPSoftware.dot (batch := batch) (m := m) (k := k) (n := n) a b⟩

def bitInput (p : Parameters) (s : State Word) (h w c : Nat) (i : AIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (if inputMask p s h w c i then (s.mem p.Input (inputOffset p s h w c i)).read .fp32
     else s.undef .fp32 p.Input (inputOffset p s h w c i))
def bitWeight (p : Parameters) (s : State Word) (h w c : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue
    (if weightMask p s c i then (s.mem p.Weight (weightOffset p s h w c i)).read .fp32
     else s.undef .fp32 p.Weight (weightOffset p s h w c i))
def bitUpdate (p : Parameters) (s : State Word) (h w c : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  fun i => FP.add .ieeeValue (a i)
    (Int4Matmul.FPSoftware.dot (batch := []) (bitInput p s h w c) (bitWeight p s h w c) i)
def bitChannels (p : Parameters) (s : State Word) (h w : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  foldRange (bitUpdate p s h w) 0 (igd p) (BIN p) a
def bitWidths (p : Parameters) (s : State Word) (h : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  foldRange (bitChannels p s h) 0 p.kernel_width 1 a
def bitOutput (p : Parameters) (s : State Word) : CIndex p → Word :=
  foldRange (bitWidths p s) 0 p.kernel_height 1 (fun _ => FP.Value.zero .fp32)

/-- Closure of every numerical callback used by the complete triple nest.
The result is independent of `rest`; no arbitrary real sum or ideal half cast
remains in this software specification. -/
theorem software_reference (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (software rest) sequentialDot p s = bitOutput p s := by
  unfold outputValue accumulation widths channels update inputValue weightValue
    bitOutput bitWidths bitChannels bitUpdate bitInput bitWeight
  simp [software, sequentialDot, FP.Software.scalarFP32, FP.Software.fp32Context]

theorem software_correctness (rest : Algebra Word) (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend (software rest) sequentialDot) (source p) initial = some final ∧
      Post p initial final (bitOutput p initial) := by
  obtain ⟨t,he,hm,hp⟩ := correctness (software rest) sequentialDot p initial hw
  exact ⟨t,he,by simpa [software_reference] using hp⟩

#axiomsClean software_reference
#axiomsClean software_correctness
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPValue
