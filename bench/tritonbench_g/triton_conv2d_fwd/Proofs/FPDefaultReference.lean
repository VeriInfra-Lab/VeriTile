import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultTypedSource
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

def igd (p : Parameters) := p.in_feat_dim / p.groups
def ogd (p : Parameters) := p.out_feat_dim / p.groups

def bhw {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) := s.pids 0 * BHW p + i.val
def bh {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) := bhw p s i / p.out_width
def batch {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) := bh p s i / p.out_height
def oh {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) := bh p s i % p.out_height
def ow {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) := bhw p s i % p.out_width
def feat {α : Type} (p : Parameters) (s : State α) (j : Fin (OF p)) := s.pids 1 * OF p + j.val

def inputBase {α : Type} (p : Parameters) (s : State α) (i : Fin (BHW p)) :=
  p.input_batch_stride * batch p s i + p.input_in_feat_stride * s.pids 2 * igd p

def weightBase {α : Type} (p : Parameters) (s : State α) (j : Fin (OF p)) :=
  p.weight_out_feat_stride * feat p s j + p.weight_out_feat_stride * s.pids 2 * ogd p

def ih {α : Type} (p : Parameters) (s : State α) (h : Nat) (i : Fin (BHW p)) : Int :=
  (h : Int) - p.padding_height + (p.stride_height : Int) * oh p s i

def iw {α : Type} (p : Parameters) (s : State α) (w : Nat) (i : Fin (BHW p)) : Int :=
  (w : Int) - p.padding_width + (p.stride_width : Int) * ow p s i

/-- Exact original pointer arithmetic; no zero-padding replacement. -/
def inputOffset {α : Type} (p : Parameters) (s : State α) (h w c : Nat) (i : AIndex p) : Nat :=
  inputBase p s i.1 + p.input_in_feat_stride * (c + i.2.1.val) +
    ((p.input_height_stride : Int) * ih p s h i.1).toNat +
    ((p.input_width_stride : Int) * iw p s w i.1).toNat

def weightOffset {α : Type} (p : Parameters) (s : State α) (h w c : Nat) (i : BIndex p) : Nat :=
  weightBase p s i.2.1 + p.weight_in_feat_stride * (c+i.1.val) +
    p.weight_height_stride * h + p.weight_width_stride * w

def inputMask {α : Type} (p : Parameters) (s : State α) (h w c : Nat) (i : AIndex p) : Prop :=
  ((((batch p s i.1 < p.batch_dim ∧ c+i.2.1.val < igd p) ∧
    0 ≤ ih p s h i.1) ∧ ih p s h i.1 < p.in_height) ∧
    0 ≤ iw p s w i.1) ∧ iw p s w i.1 < p.in_width
instance {α : Type} (p : Parameters) (s : State α) (h w c : Nat) : DecidablePred (inputMask p s h w c) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def weightMask {α : Type} (p : Parameters) (s : State α) (c : Nat) (i : BIndex p) : Prop :=
  c+i.1.val < igd p ∧ feat p s i.2.1 < ogd p
instance {α : Type} (p : Parameters) (s : State α) (c : Nat) : DecidablePred (weightMask p s c) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Nat :=
  p.output_batch_stride * batch p s i.1 +
    p.output_out_feat_stride * (s.pids 2 * ogd p + feat p s i.2.1) +
    p.output_height_stride * oh p s i.1 + p.output_width_stride * ow p s i.1

def active {α : Type} (p : Parameters) (s : State α) (i : CIndex p) : Prop :=
  ((batch p s i.1 < p.batch_dim ∧ feat p s i.2.1 < ogd p) ∧
    oh p s i.1 < p.out_height) ∧ ow p s i.1 < p.out_width
instance {α : Type} (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

def inputValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (h w c : Nat) (i : AIndex p) : α :=
  (if inputMask p s h w c i then
    (s.mem p.Input (inputOffset p s h w c i)).read .fp32
    else s.undef .fp32 p.Input (inputOffset p s h w c i))

def weightValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (h w c : Nat) (i : BIndex p) : α :=
  (if weightMask p s c i then
    (s.mem p.Weight (weightOffset p s h w c i)).read .fp32
    else s.undef .fp32 p.Weight (weightOffset p s h w c i))

/-- Independent initial-memory update, retaining all BIN lanes, unconverted fp32
operands, dot options and the subsequent rounded add. No associativity is used. -/
def update {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (h w c : Nat) (acc : CIndex p → α) : CIndex p → α :=
  fun i => M.binary (some .fp32) .real .add (acc i)
    (M.cast (some .fp32) .fp32 .real
      (D.apply (some .fp32) {allowTF32 := some p.tf32} (batch := [])
        (inputValue M p s h w c) (weightValue M p s h w c) i))

def channels {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (h w : Nat) (acc : CIndex p → α) : CIndex p → α :=
  foldRange (update M D p s h w) 0 (igd p) (BIN p) acc

def widths {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (h : Nat) (acc : CIndex p → α) : CIndex p → α :=
  foldRange (channels M D p s h) 0 p.kernel_width 1 acc

def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) : CIndex p → α :=
  foldRange (widths M D p s) 0 p.kernel_height 1
    (fun _ => M.literal (some .fp32) .real 0)

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp32 (accumulation M D p s i)

/-- Zero trip dimensions retain the source positive-zero seed, then store-cast. -/
theorem zero_height {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (s : State α) (hz : p.kernel_height=0) (i : CIndex p) :
    outputValue M D p s i = M.cast none .real .fp32 (M.literal (some .fp32) .real 0) := by
  simp [outputValue, accumulation, hz, foldRange]

#axiomsClean zero_height
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
