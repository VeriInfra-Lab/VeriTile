import bench.repairs.tritonbench_g.parallel_attention.Proofs.Physical

namespace VeriTile.Bench.Repairs.ParallelAttention.Physical
noncomputable section
open HostAPI
set_option maxHeartbeats 800000

/-- Logical source addresses for an expanded/strided incoming cotangent. -/
def viewCells (s : Shape) (kind : OutputKind) (v : View) : Set Nat :=
  {a | ∃ bh t e,bh<s.B*s.H ∧ t<s.T ∧ e<kind.width s ∧ a=v.address s bh t e}

def copyValue (a : Ops FP32Word) (s : Shape) (kind : OutputKind) (v : View)
    (m : Memory FP32Word) (i : Nat) : FP32Word :=
  let D:=kind.width s
  viewRead a s D v m (i/D/s.T) (i/D%s.T) (i%D)

/-- The allocation branch of do.contiguous()/dz.contiguous(): each copy reads
its actual incoming view from the evolving heap and stores four bytes. -/
def copy (a : Ops FP32Word) (s : Shape) (kind : OutputKind) (v : View)
    (dst : Nat) (h : Heap) : Heap :=
  cells (fun i=>dst+i) (copyValue a s kind v) (List.range (kind.cells s)) h

theorem copy_materialized (a : Ops FP32Word) (s : Shape) (kind : OutputKind) (v : View)
    (dst : Nat) (h : Heap)
    (fresh : ∀ i,i<kind.cells s → dst+i∉viewCells s kind v) :
    Materialized s kind (observe h) (observe (copy a s kind v dst h)) v (tensor s kind dst) := by
  refine ⟨rfl,?_⟩
  intro bh t e hb ht he
  have hd:=dense_bound s.T (kind.width s) (s.B*s.H) bh t e hb ht he
  have decode:=scratch_decode (s.B*s.H) s.T (kind.width s) 0 bh t e hb ht he
  simp only [zero_mul,zero_add] at decode
  have headBound : dense s.T (kind.width s) bh t e/(kind.width s)/s.T<s.B*s.H := by
    rw [Nat.div_div_eq_div_mul]
    apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos (by omega) (by omega))).mpr
    simpa [Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm] using hd
  have head : dense s.T (kind.width s) bh t e/(kind.width s)/s.T=bh := by
    simpa [Nat.mod_eq_of_lt headBound] using decode.2.2.1
  unfold copy
  rw [observe_cells]
  have result:=actual_store_value (viewCells s kind v) (fun i=>dst+i) (copyValue a s kind v)
    (by
      intro m m' same i
      dsimp only [copyValue,viewRead]
      split_ifs with active
      · exact same ⟨_,_,_,active.1,active.2.1,active.2.2,rfl⟩
      · rfl)
    (List.range (kind.cells s)) (observe h) (by intro i hi;exact fresh i (List.mem_range.mp hi))
    (dense s.T (kind.width s) bh t e) (List.mem_range.mpr hd) (by intro i j eq;dsimp at eq;omega)
  simpa [tensor,copyValue,head,decode.1,decode.2.1,viewRead,hb,ht,he] using result

/-- The no-allocation `.contiguous()` branch preserves identity and bytes.
The equality premise is only the caller's logical address map. -/
theorem alias_materialized (s : Shape) (kind : OutputKind) (v : View) (dst : Nat) (h : Heap)
    (address : ∀ bh t e,bh<s.B*s.H → t<s.T → e<kind.width s →
      v.address s bh t e=dst+dense s.T (kind.width s) bh t e) :
    Materialized s kind (observe h) (observe h) v (tensor s kind dst) := by
  refine ⟨rfl,?_⟩
  intro bh t e hb ht he
  rw [address bh t e hb ht he]
  rfl

#axiomsClean copy_materialized
#axiomsClean alias_materialized
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelAttention.Physical
