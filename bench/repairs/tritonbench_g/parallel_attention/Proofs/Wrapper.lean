import bench.repairs.tritonbench_g.parallel_attention.Proofs.Combined
import bench.repairs.tritonbench_g.parallel_attention.Proofs.Types

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

def normalizeReads (M T V o z : Nat) : Set Nat :=
  Window.Cells ⟨o,M*T*V⟩ ∪ Window.Cells ⟨z,M*T⟩

def normalizeValue (a : Ops α) (eps : α) (M T V o z : Nat) (m : Memory α) (i : Nat) : α :=
  if i < M*T*V then a.cast (a.div (m (o+i)) (a.add (m (z+i/V)) eps)) else a.zero

def normalizeRun (a : Ops α) (eps : α) (M T V o z dst : Nat) (m : Memory α) : Memory α :=
  runStores (fun i => dst+i) (normalizeValue a eps M T V o z) (List.range (M*T*V)) m

theorem normalize_local (a : Ops α) (eps : α) (M T V o z : Nat) :
    ReadsOnly (normalizeReads M T V o z) (normalizeValue a eps M T V o z) := by
  intro m m' same i
  unfold normalizeValue
  split_ifs with hi
  · have hV : 0 < V := by
      by_contra h
      have hz : V=0 := by omega
      simp [hz] at hi
    have hd : i/V < M*T := (Nat.div_lt_iff_lt_mul hV).mpr hi
    have ho : o+i∈normalizeReads M T V o z := Or.inl ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hi o⟩
    have hz : z+i/V∈normalizeReads M T V o z := Or.inr ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hd z⟩
    rw [same ho,same hz]
  · rfl

theorem normalize_output (a : Ops α) (eps : α) (M T V o z dst : Nat)
    (m : Memory α) (fresh : ∀ i,i < M*T*V → dst+i∉normalizeReads M T V o z)
    (i : Nat) (hi : i < M*T*V) :
    normalizeRun a eps M T V o z dst m (dst+i)=a.cast (a.div (m (o+i)) (a.add (m (z+i/V)) eps)) := by
  have h := actual_store_value (normalizeReads M T V o z) (fun i => dst+i)
    (normalizeValue a eps M T V o z) (normalize_local a eps M T V o z) (List.range (M*T*V)) m
    (by intro j hj;exact fresh j (List.mem_range.mp hj)) i (List.mem_range.mpr hi)
    (by intro j k h;dsimp only at h;omega)
  simpa [normalizeRun,normalizeValue,hi] using h

theorem normalize_frame (a : Ops α) (eps : α) (M T V o z dst : Nat)
    (m : Memory α) (address : Nat) (outside : address < dst ∨ dst+M*T*V ≤ address) :
    normalizeRun a eps M T V o z dst m address=m address := by
  apply runStores_frame
  intro i hi
  have := List.mem_range.mp hi
  omega

theorem dense_normalizer_index (T V bh t e : Nat) (he : e < V) :
    dense T V bh t e/V=dense T 1 bh t 0 := by
  unfold dense
  rw [(pack_divmod V (bh*T+t) e he).1]
  omega

/-- Public branch selection reads the completed forward buffers. -/
def publicRead (a : Ops α) (eps : α) (s : Shape) (o z : Nat) (m : Memory α)
    (bh : Nat) (returnBoth normalize : Bool) : PublicResult α :=
  let output := fun t e => m (o+dense s.T s.V bh t e)
  let normalizer := fun t => m (z+dense s.T 1 bh t 0)
  if returnBoth then .both output normalizer
  else if normalize then .single (fun t e => a.cast (a.div (output t e) (a.add (normalizer t) eps)))
  else .single (fun t e => a.cast (output t e))

structure SavedContext where
  inputs : Bases
  shape : Shape

structure TensorMetadata where
  batch : Nat
  heads : Nat
  time : Nat
  features : Nat
  fp32 : Bool
  contiguous : Bool
  deriving DecidableEq

def outputMetadata (s : Shape) (kind : OutputKind) : TensorMetadata :=
  ⟨s.B,s.H,s.T,kind.width s,true,true⟩

def backwardResult (q k v : TensorMetadata) : TensorMetadata × TensorMetadata × TensorMetadata × Option Nat :=
  (q,k,v,none)

theorem backward_scale_has_no_gradient (s : Shape) :
    (backwardResult (outputMetadata s .query) (outputMetadata s .key) (outputMetadata s .value)).2.2.2=none := rfl


end
end VeriTile.Bench.Repairs.ParallelAttention
