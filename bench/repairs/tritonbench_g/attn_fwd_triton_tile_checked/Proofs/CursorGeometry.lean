import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.ReadLocality

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph Model
open AttnFwdTriton.Verified.PhysicalCells (Role)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- The unchanged input pointer bases after exactly `n` chunks. -/
def cursor (f : Fixed BD) (n : Nat) : Cursor BD :=
  ⟨fun i => ((f.keyBase i).1,(f.keyBase i).2+n*64*f.keyStride),
    (f.scaleBase.1,f.scaleBase.2+n),
    fun i => ((f.valueBase i).1,(f.valueBase i).2+n*64*f.valueStride)⟩

theorem advance_cursor (f : Fixed BD) (n : Nat) :
    advance f (cursor f n) = cursor f (n+1) := by
  simp [advance,cursor,Nat.add_mul,Nat.mul_add,Nat.add_assoc]

def startChunk (f : Fixed BD) (second : Bool) := if second then f.startM*2 else 0

theorem start_cursor (f : Fixed BD) (second : Bool) :
    startCursor f second = cursor f (startChunk f second) := by
  cases second
  · simp [startCursor,cursor,startChunk]
  · simp only [startCursor,cursor,startChunk,Bool.true_eq,↓reduceIte]
    congr 1
    · funext i
      congr 1
      ring
    · congr 1
      omega
    · funext i
      congr 1
      ring

theorem lower_chunk (f : Fixed BD) (second : Bool) :
    lower f second = startChunk f second*64 := by
  cases second <;> simp [lower,startChunk] <;> omega

theorem stop_bounded (p : Config) (BD : Nat) (memory : Memory)
    (idx : ProgramIndex p) (second : Bool) :
    upper (fixed p BD memory idx.1.val idx.2.val) second ≤ p.context := by
  cases second
  · exact Nat.le_of_lt (row_valid p idx)
  · exact Nat.min_le_right _ _

theorem key_input (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : Memory) (n : Nat) (i : TileIndex [BD,64])
    (row : n*64+i.2.1.val<p.context) (dim : i.1.val<p.headDim) :
    InputAddress p ((cursor (fixed p BD memory idx.1.val idx.2.val) n).key i) := by
  refine ⟨idx.2.val/p.heads,idx.2.val%p.heads,n*64+i.2.1.val,i.1.val,
    .key,by decide,⟨(head_valid p idx).1,(head_valid p idx).2,row,dim⟩,?_⟩
  apply Prod.ext
  · rfl
  simp only [cursor,fixed,roleKey,roleOffset]
  ring

theorem value_input (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : Memory) (n : Nat) (i : TileIndex [64,BD])
    (row : n*64+i.1.val<p.context) (dim : i.2.1.val<p.headDim) :
    InputAddress p ((cursor (fixed p BD memory idx.1.val idx.2.val) n).value i) := by
  refine ⟨idx.2.val/p.heads,idx.2.val%p.heads,n*64+i.1.val,i.2.1.val,
    .value,by decide,⟨(head_valid p idx).1,(head_valid p idx).2,row,dim⟩,?_⟩
  apply Prod.ext
  · rfl
  simp only [cursor,fixed,roleKey,roleOffset]
  ring

theorem key_scale_input (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : Memory) (n : Nat) (row : n*64<p.context) :
    InputAddress p (cursor (fixed p BD memory idx.1.val idx.2.val) n).scale := by
  refine ⟨0,0,_,0,.keyScale,by decide,⟨rfl,rfl,rfl,?_⟩,rfl⟩
  have hp := idx.2.isLt
  have hn : n<(p.context+63)/64 := by omega
  have he : (p.context+64-1)/64 = (p.context+63)/64 := by omega
  simp only [cursor,fixed]
  rw [he]
  nlinarith

theorem iteration_reads (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (left right : Memory) (same : Agree p left right)
    (n : Nat) (row : n*64<p.context) :
    let f := fixed p BD left idx.1.val idx.2.val
    let ptr := cursor f n
    (∀ i : TileIndex [BD,64], f.keys (i.2.1,PUnit.unit)<f.context-n*64 →
      i.1.val<p.headDim → reads left (ptr.key i)=reads right (ptr.key i)) ∧
    reads left ptr.scale=reads right ptr.scale ∧
    (∀ i : TileIndex [64,BD], f.keys (i.1,PUnit.unit)<f.context-n*64 →
      i.2.1.val<p.headDim → reads left (ptr.value i)=reads right (ptr.value i)) := by
  refine ⟨?_,same _ (key_scale_input p BD idx left n row),?_⟩
  · intro i active dim
    exact same _ (key_input p BD idx left n i
      (by change i.2.1.val<p.context-n*64 at active; omega) dim)
  · intro i active dim
    exact same _ (value_input p BD idx left n i
      (by change i.1.val<p.context-n*64 at active; omega) dim)

end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.ReadLocality
