import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Execution
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedPhysical

/-! The independent native recurrence reads only its original input windows.
This is the task-local foundation for serializing the existing disjoint grid;
no new source execution or arithmetic identity is proved here. -/
namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model
open AttnFwdTriton.Verified.PhysicalCells (Role)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- All active input-role addresses, including both packed scale vectors. -/
def InputAddress (p : Config) (key : RegionName × Nat) : Prop :=
  ∃ z h n d r, r ≠ Role.output ∧ roleActive p z h n d r ∧
    key = roleKey p z h n d r

/-- Pure address separation. It follows from the pre-existing physical view
width equations: input cells have width four and output cells width two. -/
def Separated (p : Config) : Prop :=
  ∀ key, InputAddress p key → ¬Window p key.1 key.2

theorem views_separated (p : Config) (a : AttnFwdTriton.Verified.PhysicalCells.Layout)
    (base : Role → Nat) (views : Views a p base) : Separated p := by
  rintro key ⟨z,h,n,d,r,hr,active,rfl⟩ ⟨region,c,address⟩
  have he : roleKey p z h n d r =
      roleKey p c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val .output :=
    Prod.ext region address
  have hi := views.widths z h n d r active
  have ho := views.widths c.1.val c.2.1.val c.2.2.1.val c.2.2.2.val .output
    ⟨c.1.isLt,c.2.1.isLt,c.2.2.1.isLt,c.2.2.2.isLt⟩
  rw [he,ho] at hi
  cases r <;> simp_all [AttnFwdTriton.Verified.PhysicalCells.width]

abbrev Memory := RegionName → Nat → Cell Expr

def Agree (p : Config) (left right : Memory) : Prop :=
  ∀ key, InputAddress p key → reads left key = reads right key

theorem frame_agree (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (separate : Separated p) (s t : State Expr)
    (frame : Frame p BD idx.1.val idx.2.val s t) : Agree p t.mem s.mem := by
  intro key input
  unfold reads
  rw [frame key.1 key.2 (fun written => separate key input
    (written_within p BD idx key.1 key.2 written))]

theorem head_positive (p : Config) (idx : ProgramIndex p) : 0 < p.heads := by
  have := idx.2.isLt
  by_contra hn
  simp [Nat.eq_zero_of_not_pos hn] at this

theorem head_valid (p : Config) (idx : ProgramIndex p) :
    idx.2.val/p.heads<p.batches ∧ idx.2.val%p.heads<p.heads :=
  ⟨(Nat.div_lt_iff_lt_mul (head_positive p idx)).mpr idx.2.isLt,
    Nat.mod_lt _ (head_positive p idx)⟩

theorem row_valid (p : Config) (idx : ProgramIndex p) : idx.1.val*128<p.context := by
  have h := idx.1.isLt
  unfold rowCount at h
  split at h
  · omega
  · omega

theorem query_input (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (i : TileIndex [128,BD]) (row : idx.1.val*128+i.1.val<p.context)
    (dim : i.2.1.val<p.headDim) :
    InputAddress p (p.Q,idx.2.val/p.heads*p.queryBatch+idx.2.val%p.heads*p.queryHead+
      (idx.1.val*128+i.1.val)*p.queryRow+i.2.1.val*p.queryDim) := by
  exact ⟨idx.2.val/p.heads,idx.2.val%p.heads,idx.1.val*128+i.1.val,i.2.1.val,
    .query,by decide,⟨(head_valid p idx).1,(head_valid p idx).2,row,dim⟩,rfl⟩

theorem query_scale_input (p : Config) (idx : ProgramIndex p) :
    InputAddress p (p.QS,idx.2.val*((p.context+128-1)/128)+idx.1.val) := by
  refine ⟨0,0,_,0,.queryScale,by decide,⟨rfl,rfl,rfl,?_⟩,rfl⟩
  have hp := idx.2.isLt
  have hr := idx.1.isLt
  unfold rowCount at hr
  split at hr
  · omega
  · have he : (p.context+128-1)/128 = (p.context+127)/128 := by omega
    rw [he]
    nlinarith

theorem fixed_eq (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (left right : Memory) (same : Agree p left right) :
    fixed p BD left idx.1.val idx.2.val = fixed p BD right idx.1.val idx.2.val := by
  have query : (fixed p BD left idx.1.val idx.2.val).query =
      (fixed p BD right idx.1.val idx.2.val).query := by
    funext i
    simp only [fixed]
    split
    · rename_i active
      have hi : idx.1.val*128+i.1.val<p.context ∧ i.2.1.val<p.headDim := by
        simpa using active
      exact same _ (query_input p BD idx i hi.1 hi.2)
    · rfl
  have scale := same _ (query_scale_input p idx)
  simp only [fixed] at query ⊢
  rw [query,scale]

end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
