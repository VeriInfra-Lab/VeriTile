import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RealSource
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RealPhysical
open VeriTile Triton
open HostStorage RawStorage
open RealSource (Memory)
noncomputable section
set_option maxHeartbeats 8000000

/-- Ideal Real scalar observations at physical byte addresses. The concrete
four-byte encoding is supplied separately by the native FP32 endpoint. -/
def commit (before : Memory) (base count : Nat) (values : Nat→ℝ) : Memory := fun address=>
  if base≤address ∧ address<base+4*count then values ((address-base)/4) else before address

theorem commit_read (before : Memory) (base count : Nat) (values : Nat→ℝ) (i : Nat) (hi : i<count) :
    commit before base count values (base+4*i)=values i := by
  simp [commit,show base≤base+4*i ∧ base+4*i<base+4*count by omega]
theorem commit_frame (before : Memory) (base count : Nat) (values : Nat→ℝ) (address : Nat)
    (outside : Outside base count address) : commit before base count values address=before address := by
  simp [commit,show ¬(base≤address ∧ address<base+4*count) by have:=outside;unfold Outside at this;omega]

def copyFinal (r : Admission.Inner) (before : Memory) (values : Nat→ℝ) : Memory :=
  match r.final with | none=>before | some tensor=>commit before tensor.byteBase (finalCount r) values
theorem copy_read (r : Admission.Inner) (before : Memory) (values : Nat→ℝ) (tensor : Admission.Tensor)
    (ht : r.final=some tensor) (i : Nat) (hi : i<finalCount r) :
    copyFinal r before values (tensor.byteBase+4*i)=values i := by
  simp only [copyFinal,ht]
  exact commit_read _ _ _ _ i hi
theorem copy_frame (r : Admission.Inner) (before : Memory) (values : Nat→ℝ) (address : Nat)
    (outside : ∀ tensor,r.final=some tensor → Outside tensor.byteBase (finalCount r) address) :
    copyFinal r before values address=before address := by
  cases ht : r.final with
  | none=>simp [copyFinal,ht]
  | some tensor=>simp only [copyFinal,ht];exact commit_frame _ _ _ _ _ (outside tensor ht)

def emptyFinal (r : Admission.Inner) (before : Memory) (i : Nat) : ℝ :=
  match r.seed with | none=>0 | some tensor=>0+before (tensor.byteBase+4*i)

theorem reencode3 (K V i : Nat) : i/(K*V)*(K*V)+(i/V)%K*V+i%V=i := by
  have h1:=Nat.mod_add_div i V
  have h2:=congrArg (fun n=>n*V) (Nat.mod_add_div (i/V) K)
  rw [Nat.div_div_eq_div_mul,Nat.mul_comm V K] at h2
  nlinarith

theorem empty_formula (r : Admission.Inner) (checked : r.check=Bool.true) (empty : ¬r.nonempty)
    (before : Memory) (i : Nat) (hi : i<finalCount r) : emptyFinal r before i=RealSource.Final r before i := by
  have he : r.B*r.H*r.T*r.K*r.V=0:=not_not.mp empty
  have ht : r.T=0 := by
    have hn : r.B*r.H*r.K*r.V≠0:=by change i<r.B*r.H*r.K*r.V at hi;omega
    have eq : r.B*r.H*r.T*r.K*r.V=(r.B*r.H*r.K*r.V)*r.T:=by ring
    rw [eq] at he
    exact (mul_eq_zero.mp he).resolve_left hn
  have bt:=(r.blocks checked).1
  have nt : r.config.NT=0 := by
    simp [Admission.Inner.config,HostFP.Inner.NT,HostFP.chunks,ht,Nat.div_eq_of_lt (show r.BT-1<r.BT by omega)]
  cases seed : r.seed with
  | none=>simp [emptyFinal,RealSource.Final,RealSource.inputs,Host.innerState,seed,nt]
  | some tensor=>simp [emptyFinal,RealSource.Final,RealSource.inputs,RealSource.logical,Host.innerState,seed,nt,reencode3]

def finalValues (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState) (i : Nat) : ℝ :=
  if r.nonempty then (RealSource.innerResult r checked before scratch).readMem "ht" (HostData.finalOffset r i)
  else emptyFinal r before i
def historyValues (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState) (i : Nat) : ℝ :=
  (RealSource.innerResult r checked before scratch).readMem "h" (HostData.historyOffset r i)
def innerResult (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory)
    (scratch : BlockState) (tensor : Admission.Tensor) : Memory :=
  commit (copyFinal r before (finalValues r checked before scratch)) tensor.byteBase (innerCount r)
    (historyValues r checked before scratch)

theorem history (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState)
    (tensor : Admission.Tensor) (i : Nat) (hi : i < innerCount r) :
    innerResult r checked before scratch tensor (tensor.byteBase+4*i)=RealSource.History r before i := by
  rw [innerResult,commit_read _ _ _ _ i hi]
  exact RealSource.history r checked before scratch i hi

theorem final (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState)
    (place : InnerPlace r) (tensor : Admission.Tensor) (ht : r.final=some tensor) (i : Nat) (hi : i<finalCount r) :
    innerResult r checked before scratch place.tensor (tensor.byteBase+4*i)=RealSource.Final r before i := by
  have shape:=(r.final_valid checked tensor ht).2.2.2.2.2.2.2.2
  have fresh:=place.fresh tensor (by simp [Admission.Inner.tensors,ht])
  have sep : Disjoint place.tensor.byteBase (innerCount r) tensor.byteBase (finalCount r) := by
    simpa [innerCount,finalCount,shape,Nat.mul_assoc] using fresh
  unfold innerResult
  rw [commit_frame _ _ _ _ _ (disjoint_byte sep (by omega : 4*i<4*finalCount r)),copy_read _ _ _ tensor ht i hi]
  unfold finalValues
  split
  · exact RealSource.final r checked before scratch (by simp [ht]) i hi
  · rename_i empty
    exact empty_formula r checked empty before i hi

theorem inner_frame (r : Admission.Inner) (checked : r.check=Bool.true) (before : Memory) (scratch : BlockState)
    (tensor : Admission.Tensor) (address : Nat) (output : Outside tensor.byteBase (innerCount r) address)
    (final : ∀ t,r.final=some t → Outside t.byteBase (finalCount r) address) :
    innerResult r checked before scratch tensor address=before address := by
  rw [innerResult,commit_frame _ _ _ _ _ output]
  exact copy_frame _ _ _ _ final

def prefixResult (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState) (tensor : Admission.Tensor) : Memory :=
  commit before tensor.byteBase (prefixCount r)
    (fun i=>(RealSource.prefixResult r checked BS selected before scratch).readMem "o" (HostData.prefixOffset r i))
theorem prefix_value (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState) (tensor : Admission.Tensor)
    (i : Nat) (hi : i<prefixCount r) :
    prefixResult r checked BS selected before scratch tensor (tensor.byteBase+4*i)=RealSource.Prefix r before i := by
  rw [prefixResult,commit_read _ _ _ _ i hi]
  exact RealSource.prefix_value r checked BS selected before scratch i hi
theorem prefix_frame (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : Memory) (scratch : BlockState) (tensor : Admission.Tensor)
    (address : Nat) (outside : Outside tensor.byteBase (prefixCount r) address) :
    prefixResult r checked BS selected before scratch tensor address=before address :=
  commit_frame _ _ _ _ _ outside

#axiomsClean empty_formula
#axiomsClean history
#axiomsClean final
#axiomsClean inner_frame
#axiomsClean prefix_value
#axiomsClean prefix_frame
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RealPhysical
