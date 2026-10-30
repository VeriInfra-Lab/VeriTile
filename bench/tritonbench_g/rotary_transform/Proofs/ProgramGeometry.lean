import bench.tritonbench_g.rotary_transform.Proofs.ProgramModel
import bench.tritonbench_g.rotary_transform.Proofs.Source

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- These are exclusively input-memory facts. Both integer widths use exact
signed values; the caller's original Tensor storage supplies these reads. -/
structure MetadataBinding (c : Config) (u : State α) : Prop where
  cumulative : c.shape.varlen = Bool.true → ∀ b, b ≤ c.shape.batches →
    (u.mem "cu" b).read .int = (c.metadata.cumulative b : Int)
  offsets : c.shape.offsetTensor = Bool.true → ∀ b, b < c.shape.batches →
    (u.mem "offsets" b).read .int = (c.metadata.offsets b : Int)

def pid (u : State α) : Program := (u.pids 0,u.pids 1,u.pids 2)
def work (s : Shape) (u : State α) (lane pair : Nat) : Work :=
  ⟨u.pids 1,row s u lane,u.pids 2,pair⟩

theorem length_eq (c : Config) (valid : c.Valid) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1 < c.shape.batches) :
    sequenceLength c.shape u = (c.length (u.pids 1) : Int) := by
  cases hv : c.shape.varlen
  · simp [sequenceLength,Config.length,hv]
  · have first := md.cumulative hv _ (Nat.le_of_lt batch)
    have last := md.cumulative hv _ batch
    have mono := valid.cumulativeMonotone hv (u.pids 1) (u.pids 1+1) (by omega) batch
    simp only [sequenceLength,Config.length,hv,if_true,first,last]
    exact (Int.natCast_sub mono).symm

theorem shift_eq (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) : shift c.shape u=(c.shift (u.pids 1) : Int) := by
  cases ht : c.shape.offsetTensor
  · simp [shift,Config.shift,ht]
  · simp [shift,Config.shift,ht,md.offsets ht _ batch]

theorem base_eq (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) :
    dataBase c.shape u=c.start (u.pids 1)*(c.shape.heads*c.shape.width)+u.pids 2*c.shape.width := by
  cases hv : c.shape.varlen
  · simp [dataBase,Config.start,hv,Nat.mul_assoc]
  · simp only [dataBase,Config.start,hv,if_true,md.cumulative hv _ (Nat.le_of_lt batch),
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem offset_eq (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) (lane col : Nat) :
    dataOffset c.shape u lane col=
      c.shape.address (c.start (u.pids 1)+row c.shape u lane) (u.pids 2) col := by
  rw [dataOffset,base_eq c u md batch]
  simp only [Shape.address]
  ring

theorem coefficient_eq (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) (lane pair : Nat) :
    coefficientOffset c.shape u lane pair=c.coeff (work c.shape u lane pair) := by
  simp only [coefficientOffset,shift_eq c u md batch,Config.coeff,work,
    ←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast]

theorem live_eq (c : Config) (valid : c.Valid) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1 < c.shape.batches)
    (head : u.pids 2 < c.shape.heads) (lane pair : Nat) :
    active c.shape u lane pair Bool.false ↔ c.Active (work c.shape u lane pair) := by
  simp [active,length_eq c valid u md batch,Config.Active,work,batch,head]

theorem hasCoefficient_eq (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) (lane pair : Nat) (hp : pair<c.shape.half) :
    hasCoefficient c.shape u lane pair ↔ c.hasCoeff (work c.shape u lane pair) := by
  simp only [hasCoefficient,shift_eq c u md batch,Config.hasCoeff,work,hp,
    ←Int.natCast_add,Int.ofNat_lt,and_true]

theorem offset_dim (c : Config) (u : State α) (md : MetadataBinding c u)
    (batch : u.pids 1 < c.shape.batches) (lane pair : Nat) (second : Bool) :
    dataOffset c.shape u lane (c.shape.dim pair second)=c.cell (work c.shape u lane pair) second := by
  rw [offset_eq c u md batch]
  rfl

theorem split_value (c : Config) (valid : c.Valid) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (head : u.pids 2<c.shape.heads) (split : c.shape.interleaved=Bool.false)
    (lane pair : Nat) (live : active c.shape u lane pair Bool.false) (second : Bool) :
    splitValue c.shape p u lane pair second=
      component (pairValues c p (work c.shape u lane pair) (KernelAlgebra.memory u)) second := by
  have coeff := hasCoefficient_eq c u md batch lane pair live.2
  have low := offset_dim c u md batch lane pair Bool.false
  have high := offset_dim c u md batch lane pair Bool.true
  simp only [Shape.dim,split,Bool.false_eq_true,if_false,if_true,Nat.add_zero] at low high
  have high' : dataOffset c.shape u lane pair+c.shape.half=
      c.cell (work c.shape u lane pair) Bool.true := by
    rw [←high]
    simp [dataOffset,Nat.add_assoc]
  have high'' : c.cell (work c.shape u lane pair) Bool.false+c.shape.half=
      c.cell (work c.shape u lane pair) Bool.true := by rw [←low];exact high'
  cases second <;>
    simp [splitValue,live,cosine,sine,coeff,coefficient_eq c u md batch,
      low,high'',pairValues,sourcePair,component,split,
      KernelAlgebra.memory,KernelAlgebra.regionName]

theorem interleaved_value (c : Config) (p : Ops α) (u : State α)
    (md : MetadataBinding c u) (batch : u.pids 1<c.shape.batches)
    (inter : c.shape.interleaved=Bool.true) (lane pair : Nat)
    (live : active c.shape u lane pair Bool.false) (second : Bool) :
    interleavedValue c.shape p u lane (c.shape.dim pair second)=
      component (pairValues c p (work c.shape u lane pair) (KernelAlgebra.memory u)) second := by
  have coeff := hasCoefficient_eq c u md batch lane pair live.2
  have repeated : c.shape.dim pair second/2=pair := by
    simpa only [Shape.dim,inter,if_true] using adjacent_repeat pair second
  have swap : swapped (c.shape.dim pair second)=c.shape.dim pair (!second) := by
    cases second
    · simpa only [Shape.dim,inter,if_true,Bool.not_false,Bool.not_true,
        Bool.false_eq_true,if_false,swapped] using adjacent_swap pair Bool.false
    · simpa only [Shape.dim,inter,if_true,Bool.not_false,Bool.not_true,
        Bool.false_eq_true,if_false,swapped] using adjacent_swap pair Bool.true
  have selfLive : active c.shape u lane (c.shape.dim pair second) Bool.true :=
    ⟨live.1,dim_bound _ _ live.2 second⟩
  have otherLive : active c.shape u lane (c.shape.dim pair (!second)) Bool.true :=
    ⟨live.1,dim_bound _ _ live.2 (!second)⟩
  have parity : (c.shape.dim pair second)%2=(if second then 1 else 0) := by
    cases second <;> simp only [Shape.dim,inter,if_true,Bool.false_eq_true,if_false] <;> omega
  simp only [interleavedValue,swap,selfLive,otherLive,if_true,repeated,
    offset_dim c u md batch,cosine,sine,coeff,coefficient_eq c u md batch,parity]
  cases second <;> simp [pairValues,sourcePair,component,inter,
    KernelAlgebra.memory,KernelAlgebra.regionName]

#axiomsClean length_eq
#axiomsClean offset_eq
#axiomsClean split_value
#axiomsClean interleaved_value
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.ProgramModel
