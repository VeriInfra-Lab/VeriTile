import bench.tritonbench_g.block_sparse_attn.Proofs.RealBackend
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Real
open VeriTile.Triton
open TritonBenchG.BlockSparseAttn.StreamingMath
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

def state (sc : Nat → Fin n → R) (v : Nat → Nat → ℝ) (k : Nat) (replicas : Nat) : Acc R replicas :=
  ⟨fun _=>maximum sc k,fun _=>some (normalizer sc k),
    some (accumulated sc (fun visit j=>v visit j.val) k)⟩

theorem shifted_some (a b : R) : ops.exp (ops.sub a b)=some (shifted a b) :=
  TritonBenchG.BlockSparseAttn.RealNormalizationValues.shifted_carrier a b

theorem sumReal_fin (f : Nat → ℝ) (n : Nat) :
    FlashAttn.RealSum.sumReal f n=∑ j : Fin n,f j.val := by
  rw [FlashAttn.RealSum.sumReal_list,←List.sum_toFinset f List.nodup_range]
  simp only [List.toFinset_range,Finset.sum_range]

theorem dot_fin (instruction : Instruction) (n : Nat) (aligned : n/8*8=n)
    (a : Fin n → ℝ) (b : Nat → ℝ) (seed : ℝ) :
    dot ops instruction n (fun j=>some (if hj : j<n then a ⟨j,hj⟩ else 0))
      (fun j=>some (b j)) (some seed)=some (seed+∑ j : Fin n,a j*b j.val) := by
  rw [dot_some,aligned,sumReal_fin]
  simp only [Fin.isLt,dif_pos]

theorem normalized_real (plan : Reduction n) (instruction : Instruction)
    (aligned : n/8*8=n) (sc : Nat → Fin n → R) (v : Nat → Nat → ℝ) (col k : Nat) :
    normalizedStep plan instruction ops (sc k) (fun j=>some (v k j)) col (state sc v k plan.replicas)=
      state sc v (k+1) plan.replicas := by
  let bm := blockMax sc k
  let nm := maximum sc (k+1)
  let pr := fun j=>shifted (sc k j) bm
  let al := shifted (maximum sc k) nm
  let be := shifted bm nm
  let ln := normalizer sc (k+1)
  have hm (o : Fin plan.replicas) : (plan.tree o).eval ops.max (sc k)=bm := reduce_max plan o (sc k)
  have hp (j : Fin n) : ops.exp (ops.sub (sc k j) bm)=some (pr j) := shifted_some _ _
  have hl (o : Fin plan.replicas) : (plan.tree o).eval ops.add (fun j=>some (pr j))=some (∑ j,pr j) :=
    reduce_sum plan o pr
  have hnext : ops.max (maximum sc k) bm=nm := rfl
  have ha : ops.exp (ops.sub (maximum sc k) nm)=some al := shifted_some _ _
  have hb : ops.exp (ops.sub bm nm)=some be := shifted_some _ _
  have hden : ops.fma (some (normalizer sc k)) (some al) (ops.mul (some (∑ j,pr j)) (some be))=
      some ln := by
    have he:=block_normalizer sc k
    change al*normalizer sc k+be*(∑ j,pr j)=ln at he
    change some (normalizer sc k*al+(∑ j,pr j)*be)=some ln
    congr 1
    simpa only [mul_comm] using he
  have hmul (a b : ℝ) : ops.mul (some a) (some b)=some (a*b) := rfl
  have hdiv (a b : ℝ) : ops.div (some a) (some b)=some (a/b) := rfl
  have hleft : (fun j=>if hj : j<n then ops.mul (some (pr ⟨j,hj⟩)) (ops.div (some be) (some ln)) else ops.zero)=
      (fun j=>some (if hj : j<n then pr ⟨j,hj⟩*(be/ln) else 0)) := by
    funext j
    split <;> rfl
  simp only [normalizedStep,state,hm,hnext,hp,hl,ha,hb,hden]
  rw [hleft]
  simp only [hmul,hdiv]
  rw [dot_fin instruction n aligned (fun j=>pr j*(be/ln)) (v k)
    (accumulated sc (fun visit j=>v visit j.val) k*(al*(normalizer sc k/ln)))]
  change Acc.mk (fun _=>nm) (fun _=>some ln)
    (some (accumulated sc (fun visit j=>v visit j.val) k*(al*(normalizer sc k/ln))+
      ∑ j : Fin n,(pr j*(be/ln))*v k j.val))=state sc v (k+1) plan.replicas
  unfold state
  congr 2
  rw [accumulated]
  dsimp only
  rw [block_normalizer]
  change accumulated sc (fun visit j=>v visit j.val) k*(al*(normalizer sc k/ln))+
      (∑ j : Fin n,(pr j*(be/ln))*v k j.val)=
    accumulated sc (fun visit j=>v visit j.val) k*((normalizer sc k/ln)*al)+
      (∑ j : Fin n,(pr j*(be/ln))*v k j.val)
  rw [mul_comm al]

#axiomsClean sumReal_fin
#axiomsClean dot_fin
#axiomsClean normalized_real
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Real
