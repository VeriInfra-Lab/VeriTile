import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultComplete
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- fp32 source scalar operations. No operand half cast occurs in the original
default branch; same-format carrier conversion preserves the payload. -/
def software (rest : Algebra Word) : Algebra Word :=
  {FP.Software.scalarFP32 rest .ieeeValue with
    cast := fun precision src dst x =>
      match src,dst with
      | .fp32,.real | .real,.fp32 => x
      | src,dst => (FP.Software.scalarFP32 rest .ieeeValue).cast precision src dst x}

/-- Ascending fp32 software dot with no half narrowing or widening. Native
TF32/tensor-core scheduling is a separate numerical graph binding. -/
def dot {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → Word)
    (b : TileIndex (batch ++ [k,n]) → Word) : TileIndex (batch ++ [m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc t =>
      FP.add .ieeeValue acc (FP.mul .ieeeValue
        (a (i.1, t, PUnit.unit)) (b (t, i.2.1, PUnit.unit)))) (FP.Value.zero .fp32)
  | _ :: tail => fun i => dot (batch := tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length

def sequentialDot : Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => dot (batch := batch) (m := m) (k := k) (n := n) a b⟩

def bitInput (p : Parameters) (s : State Word) (h w c : Nat) (i : AIndex p) : Word :=
  (if inputMask p s h w c i then (s.mem p.Input (inputOffset p s h w c i)).read .fp32
     else s.undef .fp32 p.Input (inputOffset p s h w c i))
def bitWeight (p : Parameters) (s : State Word) (h w c : Nat) (i : BIndex p) : Word :=
  (if weightMask p s c i then (s.mem p.Weight (weightOffset p s h w c i)).read .fp32
     else s.undef .fp32 p.Weight (weightOffset p s h w c i))
def bitUpdate (p : Parameters) (s : State Word) (h w c : Nat) (a : CIndex p → Word) : CIndex p → Word :=
  fun i => FP.add .ieeeValue (a i)
    (dot (batch := []) (bitInput p s h w c) (bitWeight p s h w c) i)
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
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
