import bench.tritonbench_g.rotary_transform.Proofs.ProgramGeometry
import bench.tritonbench_g.rotary_transform.Proofs.ProgramScatter

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem not_early (s : Shape) (u : State α) (lane pair : Nat)
    (live : active s u lane pair Bool.false) :
    ¬((u.pids 0*s.blockM : Nat) : Int) ≥ sequenceLength s u := by
  have less := live.1
  unfold row at less
  omega

theorem split_readback (s : Shape) (heads : 0<s.heads) (rotary : 2*s.half≤s.width)
    (p : Ops α) (u : State α) (split : s.interleaved=Bool.false)
    (i : TileIndex [s.blockM,s.blockK/2])
    (live : active s u i.1.val i.2.1.val Bool.false) (second : Bool) :
    (result s p u).mem (KernelAlgebra.regionName s.output)
      (dataOffset s u i.1.val i.2.1.val+(if second then s.half else 0))=
      Cell.mk .real (splitValue s p u i.1.val i.2.1.val second) := by
  unfold result
  rw [if_neg (not_early s u _ _ live),split]
  simp only [Bool.false_eq_true,if_false]
  cases second
  · simp only [Bool.false_eq_true,if_false,Nat.add_zero]
    rw [scatter_frame _ _ _ _ _ _ _ _ _ (Or.inr ?_)]
    · exact scatter_readback s heads rotary u u _ Bool.false 0 _ i live
    · intro j hj eq
      have eq' : dataOffset s u i.1.val i.2.1.val=
          dataOffset s u j.1.val (j.2.1.val+s.half) := by simpa [dataOffset,Nat.add_assoc] using eq
      have colj : j.2.1.val+s.half<s.width := by have h:=hj.2;dsimp at h;omega
      have col := (data_injective s heads u _ _ _ _
        (active_col_bound s rotary u _ _ Bool.false live) colj eq').2
      have hi:=live.2
      dsimp at hi
      omega
  · simp only [if_true]
    exact scatter_readback s heads rotary u _ _ Bool.false s.half _ i live

theorem interleaved_readback (s : Shape) (heads : 0<s.heads) (rotary : 2*s.half≤s.width)
    (p : Ops α) (u : State α) (inter : s.interleaved=Bool.true)
    (i : TileIndex [s.blockM,s.blockK])
    (live : active s u i.1.val i.2.1.val Bool.true) :
    (result s p u).mem (KernelAlgebra.regionName s.output) (dataOffset s u i.1.val i.2.1.val)=
      Cell.mk .real (interleavedValue s p u i.1.val i.2.1.val) := by
  have early : ¬((u.pids 0*s.blockM : Nat) : Int) ≥ sequenceLength s u := by
    have less := live.1
    unfold row at less
    omega
  unfold result
  simp only [if_neg early,inter,↓reduceIte]
  exact scatter_readback s heads rotary u u _ Bool.true 0 _ i live

theorem result_pair (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (lane pair : Nat) (hl : lane<c.shape.blockM)
    (live : active c.shape u lane pair Bool.false) (second : Bool) :
    (result c.shape p u).mem (KernelAlgebra.regionName c.shape.output)
      (c.cell (work c.shape u lane pair) second)=Cell.mk .real
        (component (pairValues c p (work c.shape u lane pair) (KernelAlgebra.memory u)) second) := by
  have hp : pair<c.shape.half := live.2
  cases hi : c.shape.interleaved
  · let i : TileIndex [c.shape.blockM,c.shape.blockK/2] :=
      (⟨lane,hl⟩,⟨pair,lt_of_lt_of_le hp (blockK_covers c valid)⟩,PUnit.unit)
    have read := split_readback c.shape valid.heads valid.rotary p u hi i live second
    have off : dataOffset c.shape u lane pair+(if second then c.shape.half else 0)=
        c.cell (work c.shape u lane pair) second := by
      rw [←offset_dim c u md batch lane pair second]
      simp [Shape.dim,hi,dataOffset,Nat.add_assoc]
    simpa only [i,off,split_value c valid p u md batch head hi lane pair live second] using read
  · have bound : c.shape.dim pair second<c.shape.blockK := by
      have dim := dim_bound c.shape pair hp second
      have covers := blockK_covers c valid
      omega
    let i : TileIndex [c.shape.blockM,c.shape.blockK] :=
      (⟨lane,hl⟩,⟨c.shape.dim pair second,bound⟩,PUnit.unit)
    have activeI : active c.shape u i.1.val i.2.1.val Bool.true :=
      ⟨live.1,dim_bound _ _ hp second⟩
    have read := interleaved_readback c.shape valid.heads valid.rotary p u hi i activeI
    simpa only [i,offset_dim c u md batch,
      interleaved_value c p u md batch hi lane pair live second] using read

#axiomsClean split_readback
#axiomsClean interleaved_readback
#axiomsClean result_pair
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
