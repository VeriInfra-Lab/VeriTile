import bench.tritonbench_g.token_attn_mistral.Proofs.Binding

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Binding
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

variable {p : Parameters} {work : List Grid.PID} {counts : Fin work.length → Nat}
  {x : Ports.Logical α work.length (Ports.span p work counts) p.width} {m : Grid.Memory α}

@[simp] theorem state_mem (m : Grid.Memory α) (pid : Grid.PID) : (Grid.state m pid).mem=m := rfl
@[simp] theorem state_pid0 (m : Grid.Memory α) (pid : Grid.PID) : (Grid.state m pid).pids 0=pid.1 := rfl
@[simp] theorem state_pid1 (m : Grid.Memory α) (pid : Grid.PID) : (Grid.state m pid).pids 1=pid.2 := rfl

@[simp] theorem Facts.count (f : Facts p work counts x m) (j : Fin work.length) :
    TokenAttnMistral.count p (Grid.state m work[j])=counts j := by simpa [TokenAttnMistral.count,Grid.state] using f.attention j
@[simp] theorem Facts.sequenceLength (f : Facts p work counts x m) (j : Fin work.length) :
    TokenAttnMistral.sequenceLength p (Grid.state m work[j])=x.sequence j := by simpa [TokenAttnMistral.sequenceLength,Grid.state] using f.sequence j
@[simp] theorem Facts.startIndex (f : Facts p work counts x m) (j : Fin work.length) :
    TokenAttnMistral.startIndex p (Grid.state m work[j])=x.start j := by simpa [TokenAttnMistral.startIndex,Grid.state] using f.start j
@[simp] theorem Facts.requestIndex (f : Facts p work counts x m) (j : Fin work.length) :
    TokenAttnMistral.requestIndex p (Grid.state m work[j])=x.request j := by simpa [TokenAttnMistral.requestIndex,Grid.state] using f.request j
@[simp] theorem Facts.windowStart (f : Facts p work counts x m) (j : Fin work.length) :
    TokenAttnMistral.windowStart p (Grid.state m work[j])=Ports.window p x j := by
  exact congrArg (fun n=>max (n-p.window) 0) (f.sequenceLength j)

theorem Facts.probability_eq (f : Facts p work counts x m) (M : Algebra α)
    (domain : ∀ j,x.attention j=counts j) (j : Fin work.length)
    (n : Nat) (active : n<counts j) (lane : Fin p.block) :
    TokenAttnMistral.probability M p (Grid.state m work[j]) n lane.val=
      if n+lane.val<x.attention j then Reference.probAt (padding M) x j (n+lane.val) else padding M := by
  have hb := lane_bound p work counts j n active lane
  have eq : probOffset p (Grid.state m work[j]) lane.val+n*p.probTokenStride=
      work[j].2*p.probHeadStride+(x.start j+(n+lane.val))*p.probTokenStride := by
    simp only [probOffset,f.startIndex,state_mem,state_pid0,state_pid1]; ring
  simp only [TokenAttnMistral.probability,f.count,domain,Reference.probAt,hb,↓reduceDIte,eq,state_mem,state_pid0,state_pid1]
  split
  · rename_i live
    exact f.probability j ⟨n+lane.val,hb⟩ (by have:=lane.isLt; omega) live
  · rfl

theorem Facts.value_eq (f : Facts p work counts x m) (M : Algebra α)
    (j : Fin work.length) (n : Nat) (active : n<counts j) (aligned : n%p.block=0)
    (lane : Fin p.block) (d : Fin p.width) :
    TokenAttnMistral.value M p (Grid.state m work[j]) n lane.val d.val=
      if n+lane.val+Ports.window p x j<x.sequence j then
        Reference.valueAt (padding M) x j (n+lane.val) d else padding M := by
  have hb := lane_bound p work counts j n active lane
  have visit : Ports.visited p (counts j) (n+lane.val) :=
    ⟨by have:=lane.isLt; omega,by rw [visited_lane p n aligned lane]; exact active⟩
  have eq : gatherOffset p (Grid.state m work[j]) lane.val+n*p.requestTokenStride=
      x.request j*p.requestBatchStride+(Ports.window p x j+(n+lane.val))*p.requestTokenStride := by
    simp only [gatherOffset,f.requestIndex,f.windowStart]; ring
  simp only [TokenAttnMistral.value,gather,f.windowStart,f.sequenceLength,eq,Reference.valueAt,hb,↓reduceDIte]
  split
  · rename_i live
    simp only [live,if_true,state_mem,state_pid0,state_pid1]
    rw [f.tokens j ⟨n+lane.val,hb⟩ visit live]
    simpa only [valueOffset,state_mem,state_pid0,state_pid1] using f.value j ⟨n+lane.val,hb⟩ visit live d
  · rfl

theorem Facts.product_eq (f : Facts p work counts x m) (M : Algebra α)
    (domain : ∀ j,x.attention j=counts j) (j : Fin work.length)
    (n : Nat) (active : n<counts j) (aligned : n%p.block=0) :
    products M p (Grid.state m work[j]) n=Reference.product M p x j n := by
  funext i
  simp only [products,Reference.product,f.probability_eq M domain j n active i.1,
    f.value_eq M j n active aligned i.1 i.2.1,padding]

theorem Facts.fold_eq (f : Facts p work counts x m) (M : Algebra α) (R : SumPrimitive α)
    (domain : ∀ j,x.attention j=counts j) (j : Fin work.length)
    (n : Nat) (aligned : n%p.block=0) (a : TileIndex [p.width] → α) :
    Range.foldRange (update M R p (Grid.state m work[j])) n (counts j) p.block a=
      Range.foldRange (Reference.update M R p x j) n (counts j) p.block a := by
  conv_lhs => rw [Range.foldRange]
  conv_rhs => rw [Range.foldRange]
  by_cases zero : p.block=0
  · simp only [zero,↓reduceDIte]
  · simp only [zero,↓reduceDIte]
    by_cases active : n<counts j
    · simp only [active,↓reduceDIte]
      have eq : update M R p (Grid.state m work[j]) n=Reference.update M R p x j n := by
        funext a i
        simp only [update,Reference.update,f.product_eq M domain j n active aligned]
      rw [eq]
      exact f.fold_eq M R domain j (n+p.block) (by simp [Nat.add_mod,aligned]) _
    · simp only [active,↓reduceDIte]
termination_by counts j-n

theorem Facts.result_eq (f : Facts p work counts x m) (M : Algebra α) (R : SumPrimitive α)
    (domain : ∀ j,x.attention j=counts j) (j : Fin work.length) (d : Fin p.width) :
    outputValues M R p (Grid.state m work[j]) (d,())=Reference.result M R p x j d := by
  simp only [outputValues,accumulated,Reference.result,f.count,domain,
    f.fold_eq M R domain j 0 (by simp)]
  rfl

theorem real_weighted {x : Ports.Logical ℝ work.length (Ports.span p work counts) p.width}
    {m : Grid.Memory ℝ} (f : Facts p work counts x m) (domain : ∀ j,x.attention j=counts j)
    (positive : 0<p.block) (j : Fin work.length) (d : Fin p.width) :
    Real.weighted p (Grid.state m work[j]) d.val=Reference.weighted p x j d := by
  simp only [Real.weighted,Reference.weighted,f.count,domain,f.windowStart,f.sequenceLength]
  apply Finset.sum_congr rfl
  intro n hn
  have live := Finset.mem_range.mp hn
  have hb : n<Ports.span p work counts := by have:=count_le p work counts j; unfold Ports.span; omega
  have visit : Ports.visited p (counts j) n := ⟨positive,(Nat.div_mul_le_self n p.block).trans_lt live⟩
  split
  · rename_i active
    simp only [Reference.probAt,Reference.valueAt,hb,↓reduceDIte,
      probOffset,gatherOffset,valueOffset,f.startIndex,f.requestIndex,f.windowStart,state_mem,state_pid0,state_pid1]
    rw [f.probability j ⟨n,hb⟩ positive live,f.tokens j ⟨n,hb⟩ visit active,f.value j ⟨n,hb⟩ visit active d]
  · rfl

#axiomsClean Facts.result_eq
#axiomsClean real_weighted
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Binding
