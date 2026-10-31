import bench.tritonbench_g.rope_transform.Proofs.Settings
import bench.tritonbench_g.rope_transform.Proofs.Mathematical_TypedFrame
import bench.tritonbench_g.rope_transform.Proofs.Mathematical_TypedProof
import bench.tritonbench_g.rope_transform.Proofs.Mathematical_RopeTransform
namespace VeriTile.Bench.Repairs.RopeTransform
open VeriTile Triton FP.Structural
namespace T
export VeriTile.Bench.TritonBenchG.RopeTransform.Typed (Memory Parameters active address kernel readFloat coefficient component laneResult scatter referenceMemory execution execution_memory scatter_frame Outside reference_frame)
end T
noncomputable section
set_option maxHeartbeats 3000000

structure Request where
  (batches sequence qHeads kHeads half : Nat)
  qPositive : 0<qHeads
  kPositive : 0<kHeads
  halfPositive : 0<half

def Request.dimension (r : Request) := 2*r.half
def Request.rows (r : Request) := r.batches*r.sequence
def Request.heads (r : Request) (key : Bool) := if key then r.kHeads else r.qHeads
def region (key : Bool) : RegionName := if key then "K" else "Q"
def Request.paddedHeads (r : Request) (key : Bool) := (r.heads key).nextPowerOfTwo
def Request.paddedDimension (r : Request) := r.dimension.nextPowerOfTwo
def Request.width (r : Request) := r.paddedDimension/2
def Request.stride (r : Request) (key : Bool) := r.heads key*r.dimension
def Request.offset (r : Request) (key : Bool) (a h c : Nat) :=
  a*r.stride key+h*r.dimension+c
def Request.external (r : Request) (key : Bool) (b h s c : Nat) :=
  ((b*r.heads key+h)*r.sequence+s)*r.dimension+c
def Request.column (r : Request) (second : Bool) (j : Nat) := j+if second then r.half else 0
def Request.parameters (r : Request) : T.Parameters :=
  ⟨"Q","K","C","S",r.stride Bool.false,r.stride Bool.true,r.half,r.half,r.sequence,
    r.qHeads,r.kHeads,r.dimension,r.paddedHeads Bool.false,r.paddedHeads Bool.true,r.paddedDimension,Bool.false⟩
def kernel (r : Request) := T.kernel .fp32 .fp32 r.parameters

theorem half_eq (r : Request) : r.dimension/2=r.half := by simp [Request.dimension]
theorem heads_positive (r : Request) (key : Bool) : 0<r.heads key := by
  cases key <;> simp [Request.heads,r.qPositive,r.kPositive]
theorem heads_cover (r : Request) (key : Bool) : r.heads key≤r.paddedHeads key := nextPower_covers _
theorem width_cover (r : Request) : r.half≤r.width := by
  have := nextPower_covers r.dimension
  unfold Request.width Request.paddedDimension Request.dimension at *
  omega
theorem width_le (r : Request) : r.width≤r.dimension := by
  have h := nextPower_lt r.dimension (by have := r.halfPositive; unfold Request.dimension; omega)
  unfold Request.width Request.paddedDimension
  omega
theorem column_lt (r : Request) (second : Bool) (j : Nat) (hj : j<r.half) :
    r.column second j<r.dimension := by
  cases second <;> simp only [Request.column, Bool.false_eq_true, ↓reduceIte]
  all_goals unfold Request.dimension; omega

theorem block_injective (n a b i j : Nat) (hi : i<n) (hj : j<n)
    (he : a*n+i=b*n+j) : a=b ∧ i=j := by
  have hm := congrArg (fun v => v%n) he
  simp [Nat.add_mod,Nat.mod_eq_of_lt hi,Nat.mod_eq_of_lt hj] at hm
  have hn : 0<n := by omega
  have hab : a*n=b*n := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right hn hab,hm⟩

theorem offset_injective (r : Request) (key : Bool) (a b h g c d : Nat)
    (hh : h<r.heads key) (hg : g<r.heads key) (hc : c<r.dimension) (hd : d<r.dimension)
    (he : r.offset key a h c=r.offset key b g d) : a=b ∧ h=g ∧ c=d := by
  have he' : (a*r.heads key+h)*r.dimension+c=(b*r.heads key+g)*r.dimension+d := by
    simpa [Request.offset,Request.stride,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨he2,hcd⟩ := block_injective _ _ _ _ _ hc hd he'
  obtain ⟨hab,hhg⟩ := block_injective _ _ _ _ _ hh hg he2
  exact ⟨hab,hhg,hcd⟩

theorem offset_window (r : Request) (key : Bool) (a h c : Nat)
    (hh : h<r.heads key) (hc : c<r.dimension) :
    a*r.stride key≤r.offset key a h c ∧ r.offset key a h c<(a+1)*r.stride key := by
  have ht := Nat.mul_le_mul_right r.dimension (show h+1≤r.heads key by omega)
  simp only [Nat.add_mul,Nat.one_mul] at ht
  unfold Request.offset Request.stride
  constructor
  · omega
  · rw [Nat.add_mul,Nat.one_mul]; omega

theorem row_bound (r : Request) (b s : Nat) (hb : b<r.batches) (hs : s<r.sequence) :
    b*r.sequence+s<r.rows := by
  have ht := Nat.mul_le_mul_right r.sequence (show b+1≤r.batches by omega)
  simp only [Nat.add_mul,Nat.one_mul] at ht
  unfold Request.rows; omega

def lane (r : Request) (key : Bool) (h j : Nat)
    (hh : h<r.heads key) (hj : j<r.half) : TileIndex [r.paddedHeads key,r.width] :=
  (⟨h,lt_of_lt_of_le hh (heads_cover r key)⟩,
   ⟨j,lt_of_lt_of_le hj (width_cover r)⟩,PUnit.unit)

theorem lane_active (r : Request) (key : Bool) (h j : Nat)
    (hh : h<r.heads key) (hj : j<r.half) :
    T.active (r.heads key) (r.dimension/2) (lane r key h j hh hj) := by
  simp [T.active,lane,half_eq,hh,hj]

theorem address_eq (r : Request) (key : Bool) (a : Nat) (second : Bool)
    (i : TileIndex [r.paddedHeads key,r.width]) :
    T.address a (r.stride key) r.dimension (r.dimension/2) second i=
      r.offset key a i.1.val (r.column second i.2.1.val) := by
  simp [T.address,Request.offset,Request.column,half_eq,Nat.add_assoc]

theorem address_injective (r : Request) (key : Bool) (a : Nat) (second : Bool) :
    Function.Injective (T.address a (r.stride key) r.dimension (r.dimension/2) second :
      TileIndex [r.paddedHeads key,r.width] → Nat) := by
  intro i j he
  have hi : i.2.1.val<r.dimension := lt_of_lt_of_le i.2.1.isLt (width_le r)
  have hj : j.2.1.val<r.dimension := lt_of_lt_of_le j.2.1.isLt (width_le r)
  have he' : i.1.val*r.dimension+i.2.1.val=j.1.val*r.dimension+j.2.1.val := by
    unfold T.address at he; omega
  obtain ⟨eh,ej⟩ := block_injective _ _ _ _ _ hi hj he'
  apply Prod.ext
  · exact Fin.ext eh
  · apply Prod.ext
    · exact Fin.ext ej
    · exact Subsingleton.elim _ _
end
end VeriTile.Bench.Repairs.RopeTransform
