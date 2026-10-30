import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeSupported

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

theorem iterate_supported (c : Call) (a : Reference.Inputs Expr) (cur stop : Int)
    (old : Reference.Carry c Expr) (scale : c.scale=ScaleBinding.marker)
    (good : ∀ rg off,Supported c (a.read rg off)) (carry : SupportedCarry c old) :
    SupportedCarry c (Reference.iterate c backend a cur stop old) := by
  rw [Reference.iterate]
  split_ifs with zero live
  · exact carry
  · exact iterate_supported c a (cur+c.BN) stop _ scale good (advance_supported c a cur old scale good carry)
  · exact carry
termination_by (stop-cur).toNat
decreasing_by omega

theorem result_supported (c : Call) (a : Reference.Inputs Expr)
    (scale : c.scale=ScaleBinding.marker) (good : ∀ rg off,Supported c (a.read rg off)) :
    SupportedCarry c (Reference.result c backend a) :=
  iterate_supported c a _ _ _ scale good (initial_supported c a good)

def SupportedMemory (c : Call) (s : State Expr) : Prop :=
  ∀ rg off,Supported c ((s.mem rg off).read .real)

theorem writes_supported {I : Type} (c : Call) (is : List I) (rg : RegionName) (off : I → Nat)
    (values : I → Expr) (s : State Expr) (good : SupportedMemory c s)
    (valuesGood : ∀ i,Supported c (values i)) :
    SupportedMemory c (is.foldl (fun t i => t.write rg (off i) (.mk .real (values i))) s) := by
  induction is generalizing s with
  | nil => exact good
  | cons i rest ih =>
    apply ih
    intro region offset
    by_cases sameRegion : region=rg
    · subst region
      by_cases sameOffset : offset=off i
      · simpa [State.write,sameOffset] using valuesGood i
      · simpa [State.write,sameOffset] using good rg offset
    · simpa [State.write,sameRegion] using good region offset

theorem scatter_supported (c : Call) (a : Reference.Inputs Expr) (old : Reference.Carry c Expr)
    (s : State Expr) (good : SupportedMemory c s) (carry : SupportedCarry c old) :
    SupportedMemory c (Reference.scatter c backend a old s) := by
  unfold Reference.scatter Reference.writeOutput
  apply writes_supported
  · unfold Reference.writeRows
    apply writes_supported
    · split_ifs
      · exact writes_supported c _ _ _ _ s good carry.mass
      · exact good
    · intro i
      unfold Reference.finalCenter
      split_ifs
      · exact supported_add_log c _ _ (carry.center i) (carry.mass i)
      · exact carry.center i
  · intro i
    unfold Reference.finalOutput
    change Supported c (if _ then consume .div _ _ else _)
    split_ifs
    · exact supported_div c _ _ (carry.weighted i) (carry.mass _)
    · exact carry.weighted i

theorem grid_supported (c : Call) (scale : c.scale=ScaleBinding.marker)
    (jobs : List (GridIndex c.grid)) (s : State Expr) (good : SupportedMemory c s) :
    SupportedMemory c (Reference.referenceSchedule c backend jobs s) := by
  induction jobs generalizing s with
  | nil => exact good
  | cons idx rest ih =>
    apply ih
    exact scatter_supported c _ _ _ good (result_supported c _ scale good)

theorem target_supported (c : Call) (A : FlatAlloc) (before : Bytes.Memory) :
    SupportedMemory (ScaleBinding.template c) (targetState c A before) := by
  apply grid_supported _ rfl
  intro rg off
  exact True.intro

theorem no_unsupported (c : Call) (e : Expr) (good : Supported c e) : e≠.unsupported := by
  intro same
  subst e
  exact good

theorem reduction_width (c : Call) (maximum : Bool) (row n : Nat) (values : Fin n → Expr)
    (good : Supported c (.reduction maximum row n values)) : n=c.BN := good.1

theorem literal_whitelist (c : Call) (x : ℝ) (good : Supported c (.literal x)) :
    x=0 ∨ x=1 ∨ x=1.4426950408889634 := good

#axiomsClean iterate_supported
#axiomsClean grid_supported
#axiomsClean target_supported
#axiomsClean no_unsupported
#axiomsClean reduction_width
#axiomsClean literal_whitelist
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
