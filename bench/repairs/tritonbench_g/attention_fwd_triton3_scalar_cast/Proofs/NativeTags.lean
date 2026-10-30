import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeExpr
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Epilogue
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Grid

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

theorem initial_center (c : Call) (a : Reference.Inputs Expr)
    (tagged : ∀ off,centerLike (a.read "M" off)=Bool.true) (i : TileIndex [c.BM]) :
    centerLike ((Reference.initial c backend a).center i)=Bool.true := by
  by_cases hi : c.init=0
  · simpa [Reference.initial,hi,backend,algebra] using tagged (Reference.carryAddress c a i.1.val)
  · simp [Reference.initial,hi,backend,algebra,literal,Ne.symm ScaleBinding.marker_distinct.1,consume,centerLike]

theorem advance_center (c : Call) (a : Reference.Inputs Expr) (cur : Int)
    (old : Reference.Carry c Expr) (i : TileIndex [c.BM]) :
    centerLike ((Reference.advance c backend a cur old).center i)=Bool.true := rfl

theorem iterate_center (c : Call) (a : Reference.Inputs Expr) (cur stop : Int)
    (old : Reference.Carry c Expr)
    (tagged : ∀ i,centerLike (old.center i)=Bool.true) :
    ∀ i,centerLike ((Reference.iterate c backend a cur stop old).center i)=Bool.true := by
  rw [Reference.iterate]
  split_ifs with hz hc
  · exact tagged
  · exact iterate_center c a (cur+c.BN) stop (Reference.advance c backend a cur old)
      (advance_center c a cur old)
  · exact tagged
termination_by (stop-cur).toNat
decreasing_by omega

theorem result_center (c : Call) (a : Reference.Inputs Expr)
    (tagged : ∀ off,centerLike (a.read "M" off)=Bool.true) :
    ∀ i,centerLike ((Reference.result c backend a).center i)=Bool.true :=
  iterate_center c a _ _ _ (initial_center c a tagged)

theorem normalized_center (a b : Expr) :
    centerLike (consume .add a (unary .log2 b))=centerLike a := by
  cases a <;> rfl

theorem final_center (c : Call) (old : Reference.Carry c Expr)
    (tagged : ∀ i,centerLike (old.center i)=Bool.true) (i : TileIndex [c.BM]) :
    centerLike (Reference.finalCenter c backend old i)=Bool.true := by
  unfold Reference.finalCenter
  split_ifs
  · exact (normalized_center _ _).trans (tagged i)
  · exact tagged i

/-- Every online iteration preserves the provenance required by the delayed
NaN-alpha node; no equality or numerical agreement between owners is used. -/
theorem all_alpha_guards (c : Call) (a : Reference.Inputs Expr) (cur stop : Int)
    (old : Reference.Carry c Expr) (tagged : ∀ i,centerLike (old.center i)=Bool.true)
    (i : TileIndex [c.BM]) (next : Expr) :
    (if !(backend.eq (some .fp32) .real
        (consume .sub ((Reference.iterate c backend a cur stop old).center i) next)
        (consume .sub ((Reference.iterate c backend a cur stop old).center i) next)) then literal 1
      else unary .exp2 (consume .sub ((Reference.iterate c backend a cur stop old).center i) next))=
      .alpha ((Reference.iterate c backend a cur stop old).center i) next :=
  alpha_staged _ _ (iterate_center c a cur stop old tagged i)

def TaggedMemory (s : State Expr) : Prop :=
  ∀ off,centerLike ((s.mem "M" off).read .real)=Bool.true

theorem other_writes_tagged {I : Type} (is : List I) (rg : RegionName) (off : I → Nat)
    (values : I → Cell Expr) (s : State Expr) (tagged : TaggedMemory s) (other : rg≠"M") :
    TaggedMemory (is.foldl (fun t i => t.write rg (off i) (values i)) s) := by
  intro index
  rw [State.scatter_frame rg off values is "M" index (Or.inl (Ne.symm other))]
  exact tagged index

theorem center_writes_tagged {I : Type} (is : List I) (off : I → Nat)
    (values : I → Expr) (s : State Expr) (tagged : TaggedMemory s)
    (good : ∀ i,centerLike (values i)=Bool.true) :
    TaggedMemory (is.foldl (fun t i => t.write "M" (off i) (.mk .real (values i))) s) := by
  induction is generalizing s with
  | nil => exact tagged
  | cons i rest ih =>
    apply ih
    intro index
    by_cases same : index=off i
    · simp [State.write,same,good i]
    · simpa [State.write,same] using tagged index

theorem scatter_tagged (c : Call) (a : Reference.Inputs Expr) (old : Reference.Carry c Expr)
    (s : State Expr) (tagged : TaggedMemory s)
    (centers : ∀ i,centerLike (old.center i)=Bool.true) :
    TaggedMemory (Reference.scatter c backend a old s) := by
  unfold Reference.scatter Reference.writeOutput
  apply other_writes_tagged
  · unfold Reference.writeRows
    apply center_writes_tagged
    · split_ifs
      · exact other_writes_tagged _ _ _ _ _ tagged (by decide)
      · exact tagged
    · exact final_center c old centers
  · simp [Reference.optr]

theorem grid_centers (c : Call) (jobs : List (GridIndex c.grid)) (s : State Expr)
    (tagged : TaggedMemory s) : TaggedMemory (Reference.referenceSchedule c backend jobs s) := by
  induction jobs generalizing s with
  | nil => exact tagged
  | cons idx rest ih =>
    apply ih
    apply scatter_tagged
    · exact tagged
    · exact result_center c _ tagged

#axiomsClean initial_center
#axiomsClean result_center
#axiomsClean final_center
#axiomsClean all_alpha_guards
#axiomsClean grid_centers
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
