import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalInputs

namespace VeriTile.Bench.Repairs.Conv2d.ConvIO
open VeriTile Triton FP.Structural KernelNumeric
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

/-- Equality on logical tensor coordinates only. Holes, output cells, and every
other region remain unconstrained. -/
structure ArrayReads (s : Shape) (a b : Memory A) : Prop where
  input : ∀ n c y x, n < s.n → c < s.channels → y < s.h → x < s.w →
    a .input (s.sn*n+s.sc*c+s.sx*y+s.sy*x)=b .input (s.sn*n+s.sc*c+s.sx*y+s.sy*x)
  weight : ∀ o c h w, o < s.outputs → c < s.cg → h < s.kh → w < s.kw →
    a .weight (s.so*o+s.tc*c+s.tx*h+s.ty*w)=b .weight (s.so*o+s.tc*c+s.tx*h+s.ty*w)

theorem array_reads (o : Operations A) (s : Shape) (f : Flags) (vals : Inputs A s)
    (t : State (Payload A)) (bound : (program o s f).InputBinding vals t) :
    ArrayReads s (KernelNumeric.memory o t) (logicalScalars o s vals) := by
  constructor
  · intro n c y x hn hc hy hx
    exact congrArg (scalar o) (read_extension o s f vals t bound _ _ (occurs_input s n c y x hn hc hy hx))
  · intro n c h w hn hc hh hw
    exact congrArg (scalar o) (read_extension o s f vals t bound _ _ (occurs_weight s n c h w hn hc hh hw))

theorem inputArray_eq (s : Shape) (a b : Memory A) (same : ArrayReads s a b)
    (zero : A) (n c : Nat) (y x : Int) : inputArray s zero a n c y x=inputArray s zero b n c y x := by
  unfold inputArray
  split_ifs with active
  · exact same.input n c y.toNat x.toNat active.1 active.2.1 (by omega) (by omega)
  · rfl

theorem refA_eq (s : Shape) (a b : Memory A) (same : ArrayReads s a b)
    (zero : A) (g r h w c : Nat) :
    refA s (arrayISA zero) a g r h w c=refA s (arrayISA zero) b g r h w c := by
  unfold refA
  split_ifs <;> first | exact inputArray_eq s a b same zero _ _ _ _ | rfl

theorem refB_eq (s : Shape) (a b : Memory A) (same : ArrayReads s a b)
    (zero : A) (g f h w c : Nat) (hg : g < s.groups) (hh : h < s.kh) (hw : w < s.kw) :
    refB s (arrayISA zero) a g f h w c=refB s (arrayISA zero) b g f h w c := by
  unfold refB
  split_ifs with active
  · apply same.weight _ _ _ _ _ active.1 hh hw
    unfold Shape.outputs
    nlinarith [active.2]
  · rfl

theorem range_congr {B : Type} (f g : Nat → B → B) (cur stop stride : Nat) (initial : B)
    (same : ∀ q, q < stop → ∀ a, f q a=g q a) :
    foldRange f cur stop stride initial=foldRange g cur stop stride initial := by
  by_cases hz : stride=0
  · simp [foldRange,hz]
  · by_cases active : cur < stop
    · conv_lhs => rw [foldRange]
      conv_rhs => rw [foldRange]
      simp only [hz,active,if_false,if_true]
      rw [same cur active]
      exact range_congr f g (cur+stride) stop stride _ same
    · simp [foldRange,hz,active]
termination_by stop-cur
decreasing_by omega

/-- Source-independent recurrence congruence uses only logical array reads;
the original h/w/channel-block order and every padded operation remain. -/
theorem reference_eq (o : Operations A) (s : Shape) (f : Flags) (a b : Memory A)
    (same : ArrayReads s a b) (x : Conv2d.Cell) (hg : x.group < s.groups) :
    reference o s f a x=reference o s f b x := by
  unfold reference
  apply range_congr
  intro h hh acc
  apply range_congr
  intro w hw acc
  apply range_congr
  intro c hc acc
  congr 1
  · funext e
    rw [refA_eq s a b same]
  · funext e
    rw [refB_eq s a b same _ _ _ _ _ _ hg hh hw]

#axiomsClean reference_eq
end
end VeriTile.Bench.Repairs.Conv2d.ConvIO
