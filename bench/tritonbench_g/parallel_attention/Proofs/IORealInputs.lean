import bench.tritonbench_g.parallel_attention.Proofs.IOInputBinding

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open scoped BigOperators
noncomputable section

structure TensorAgree (direction : Direction) (shape : Shape) (x y : TensorInputs ℝ) (bh : Nat) : Prop where
  query : ∀ t d,t<shape.T → d<shape.K → x.q bh t d=y.q bh t d
  key : ∀ t d,t<shape.T → d<shape.K → x.k bh t d=y.k bh t d
  value : ∀ t d,t<shape.T → d<shape.V → x.v bh t d=y.v bh t d
  outGrad : direction=.backward → ∀ t d,t<shape.T → d<shape.V → x.do_ bh t d=y.do_ bh t d
  normGrad : direction=.backward → ∀ t,t<shape.T → x.dz bh t=y.dz bh t

theorem tensor_agree (direction : Direction) (shape : Shape) (xs : ChannelValues ℝ (inputPorts shape))
    (s : State ℝ) (bh : Nat) (hh : bh<heads shape)
    (h : InputsAgree direction shape (fun r off=>(s.mem r off).read .real) (inputValues shape xs)) :
    TensorAgree direction shape (inputReaders shape s) (tensorValues shape xs) bh := by
  constructor
  · intro t d ht hd
    simpa [inputReaders,tensorValues,dense,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h.query _ (dense_bound shape.T shape.K (heads shape) bh t d hh ht hd)
  · intro t d ht hd
    simpa [inputReaders,tensorValues,dense,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h.key _ (dense_bound shape.T shape.K (heads shape) bh t d hh ht hd)
  · intro t d ht hd
    simpa [inputReaders,tensorValues,dense,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h.value _ (dense_bound shape.T shape.V (heads shape) bh t d hh ht hd)
  · intro dir t d ht hd
    simpa [inputReaders,tensorValues,dense,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using h.outGrad dir _ (dense_bound shape.T shape.V (heads shape) bh t d hh ht hd)
  · intro dir t ht
    apply h.normGrad dir
    nlinarith

theorem score_congr {direction : Direction} {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree direction shape x y bh) (scale : ℝ) (t j : Nat) (ht : t<shape.T) (hj : j<shape.T) :
    score shape.K scale (headInputs x bh) t j=score shape.K scale (headInputs y bh) t j := by
  unfold score
  congr 1
  apply Finset.sum_congr rfl
  intro e he
  simp only [headInputs,h.query t e ht (Finset.mem_range.mp he),h.key j e hj (Finset.mem_range.mp he)]

theorem output_congr {direction : Direction} {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree direction shape x y bh) (scale : ℝ) (t v : Nat) (ht : t<shape.T) (hv : v<shape.V) :
    outputSpec shape.T shape.K scale (headInputs x bh) t v=
      outputSpec shape.T shape.K scale (headInputs y bh) t v := by
  apply Finset.sum_congr rfl
  intro j hj
  have hj:=Finset.mem_range.mp hj
  simp only [weight]
  rw [score_congr h scale t j ht hj]
  simp only [headInputs,h.value j v hj hv]

theorem normalizer_congr {direction : Direction} {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree direction shape x y bh) (scale : ℝ) (t : Nat) (ht : t<shape.T) :
    normalizerSpec shape.T shape.K scale (headInputs x bh) t=
      normalizerSpec shape.T shape.K scale (headInputs y bh) t := by
  apply Finset.sum_congr rfl
  intro j hj
  simp only [weight,score_congr h scale t j ht (Finset.mem_range.mp hj)]

theorem cotangent_congr {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree .backward shape x y bh) (shard t j : Nat) (ht : t<shape.T) (hj : j<shape.T) :
    shardCotangent shape x bh shard t j=shardCotangent shape y bh shard t j := by
  unfold shardCotangent
  rw [h.normGrad rfl t ht]
  congr 1
  apply Finset.sum_congr rfl
  intro v _
  split
  · rename_i active
    rw [h.outGrad rfl t _ ht active,h.value j _ hj active]
  · rfl

theorem query_congr {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree .backward shape x y bh) (scale : ℝ) (shard t e : Nat) (ht : t<shape.T) (he : e<shape.K) :
    queryPartial shape scale x bh shard t e=queryPartial shape scale y bh shard t e := by
  unfold queryPartial
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  have hj:=Finset.mem_range.mp hj
  rw [cotangent_congr h shard t j ht hj,score_congr h scale t j ht hj,h.key j e hj he]

theorem key_congr {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree .backward shape x y bh) (scale : ℝ) (shard j e : Nat) (hj : j<shape.T) (he : e<shape.K) :
    keyPartial shape scale x bh shard j e=keyPartial shape scale y bh shard j e := by
  apply Finset.sum_congr rfl
  intro t ht
  have ht:=Finset.mem_range.mp ht
  rw [cotangent_congr h shard t j ht hj,score_congr h scale t j ht hj,h.query t e ht he]

theorem value_congr {shape : Shape} {x y : TensorInputs ℝ} {bh : Nat}
    (h : TensorAgree .backward shape x y bh) (scale : ℝ) (j v : Nat) (hj : j<shape.T) (hv : v<shape.V) :
    valueGradient shape scale x bh j v=valueGradient shape scale y bh j v := by
  apply Finset.sum_congr rfl
  intro t ht
  have ht:=Finset.mem_range.mp ht
  rw [score_congr h scale t j ht hj,h.outGrad rfl t v ht hv]

#axiomsClean tensor_agree
#axiomsClean output_congr
#axiomsClean normalizer_congr
#axiomsClean query_congr
#axiomsClean key_congr
#axiomsClean value_congr
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
