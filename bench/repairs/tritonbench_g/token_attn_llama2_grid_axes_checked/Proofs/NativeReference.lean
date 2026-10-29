import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.NativeKernel
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Logical

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.KernelNative
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 8000000

def wordRead (before : Cells Word) (address : Nat) : Word := (before address).read .fp32

def tensorWord (before : Cells Word) (p : Payload) (i : Nat) : Word :=
  (VeriTile.Torch.FP32.logicalValue (fun a => ⟨wordRead before a⟩) p.tensor p.laws i).bits

def queryTerm (t : TensorCall) (before : Cells Word) (b h : Nat) (d : Fin t.call.dim) : Native.Term :=
  .cast (some .fp32) .fp32 .real
    (.leaf "Q" (b*t.call.qBatch+h*t.call.qHead+d.val*t.call.qDim)
      (tensorWord before t.q (queryIndex t b h d.val)))

def keyTerm (t : TensorCall) (before : Cells Word) (b h n : Nat) (d : Fin t.call.dim) : Native.Term :=
  .cast (some .fp32) .fp32 .real
    (.leaf "K" ((metadata t (wordRead before)).page b n*t.call.kBatch+
        (h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d.val*t.call.kDim)
      (tensorWord before t.k (keyIndex t (wordRead before) b h n d.val)))

def productTerm (t : TensorCall) (before : Cells Word) (b h n : Nat)
    (d : Fin t.call.dim) : Native.Term :=
  .binary (some .fp32) .real .mul (queryTerm t before b h d) (keyTerm t before b h n d)

/-- Independent ordered instruction graph on original logical Tensor words.
Tree selection reads shapes, not numerical words; every D leaf is present. -/
def referenceTerm (trees : Program → Native.Reductions) (t : TensorCall)
    (before : Cells Word) (b h n : Nat) : Native.Term :=
  let products := productTerm t before b h n
  let plan := trees (b,h,n/32) (some .fp32) [32,t.call.dim] ⟨1,by simp⟩ Bool.false
    (⟨n%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit) (fun d => Native.shape (products d))
  .cast none .real .fp32 (.binary (some .fp32) .real .mul
    (plan.tree.expand (some .fp32) .real .add products (.literal (some .fp32) .real 0))
    (.literal (some .fp32) .real (hostScale t.call.dim)))

def reference (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (t : TensorCall) (before : Cells Word) (b h n : Nat) : Word :=
  Native.eval I fusion (scaleWord t.call.dim) (referenceTerm trees t before b h n)

theorem query_node (R : Native.Reductions) (t : TensorCall) (before : Cells Word)
    (rest : State Word) (contract : InputContract t (wordRead before)) (launch : active t)
    (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads) (d : Fin t.call.dim) :
    VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.queryValue (Native.algebra R)
      (t.call.parameters (hostScale t.call.dim))
      (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
        (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest))) d=
      queryTerm t before b h d := by
  have domain := valid t (wordRead before) contract launch
  have bound := domain.queryBounds b hb h hh d.val d.isLt
  have ordinary : t.q.tensor.metadata.isNeg=Bool.false :=
    contract.1.ordinary t.q.observation (by simp [TensorCall.observations])
  have words := (query_tensor t (wordRead before) before contract launch b h hb hh d).trans
    (logical_word before t.q ordinary (queryIndex t b h d.val))
  simp only [query,finiteRead,dif_pos bound,buffers] at words
  change Native.Term.cast (some .fp32) .fp32 .real
    (finiteRead (Native.injectBuffers t.call (buffers t before)).q
      (b*t.call.qBatch+h*t.call.qHead+d.val*t.call.qDim))=queryTerm t before b h d
  rw [finiteRead,dif_pos bound]
  exact congrArg (fun w => Native.Term.cast (some .fp32) .fp32 .real
    (.leaf "Q" (b*t.call.qBatch+h*t.call.qHead+d.val*t.call.qDim) w)) words

theorem key_node (R : Native.Reductions) (t : TensorCall) (before : Cells Word)
    (rest : State Word) (contract : InputContract t (wordRead before)) (launch : active t)
    (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads)
    (hn : n<(metadata t (wordRead before)).len b) (d : Fin t.call.dim) :
    VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.keyValue (Native.algebra R)
      (t.call.parameters (hostScale t.call.dim))
      (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
        (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest)))
      (lane t.call (hostScale t.call.dim) n) d=keyTerm t before b h n d := by
  have domain := valid t (wordRead before) contract launch
  have bound := domain.keyBounds _ (domain.pages ⟨b,hb⟩ n hn).2
    _ (grouped_head t.call (metadata t (wordRead before)) domain h hh) d.val d.isLt
  change (metadata t (wordRead before)).page b n*t.call.kBatch+
    (h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d.val*t.call.kDim<t.call.kSize at bound
  have ordinary : t.k.tensor.metadata.isNeg=Bool.false :=
    contract.1.ordinary t.k.observation (by simp [TensorCall.observations])
  have words := (key_tensor t (wordRead before) before contract launch b h n hb hh hn d).trans
    (logical_word before t.k ordinary (keyIndex t (wordRead before) b h n d.val))
  simp only [key,finiteRead,dif_pos bound,buffers] at words
  have split : n/32*32+n%32=n := by omega
  have lengthBound := length_le t (wordRead before) contract b hb
  have live : VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.active
      (t.call.parameters (hostScale t.call.dim))
      (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
        (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest)))
      (lane t.call (hostScale t.call.dim) n) := by
    rw [initial_active]
    change t.maximum-(metadata t (wordRead before)).len b+(n/32*32+n%32)<t.maximum
    omega
  have offset : VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.keyOffset
      (t.call.parameters (hostScale t.call.dim))
      (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
        (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest)))
      (lane t.call (hostScale t.call.dim) n) d=
      (metadata t (wordRead before)).page b n*t.call.kBatch+
        (h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d.val*t.call.kDim := by
    unfold VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.keyOffset
      VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.keyLocation
    rw [if_pos live]
    simp [
      VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.windowIndex,
      VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.length,
      VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.token,
      Call.parameters,start,ids,initial,Whole.Metadata.page,lane,split,Region.cast]
  unfold VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.keyValue
  rw [if_pos live,offset]
  change Native.Term.cast (some .fp32) .fp32 .real
    (finiteRead (Native.injectBuffers t.call (buffers t before)).k
      ((metadata t (wordRead before)).page b n*t.call.kBatch+
        (h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d.val*t.call.kDim))=keyTerm t before b h n d
  rw [finiteRead,dif_pos bound]
  exact congrArg (fun w => Native.Term.cast (some .fp32) .fp32 .real
    (.leaf "K" ((metadata t (wordRead before)).page b n*t.call.kBatch+
      (h/(t.call.heads/t.call.kvHeads))*t.call.kHead+d.val*t.call.kDim) w)) words

theorem source_reference (trees : Program → Native.Reductions) (t : TensorCall)
    (before : Cells Word) (rest : State Word) (contract : InputContract t (wordRead before))
    (launch : active t) (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads)
    (hn : n<(metadata t (wordRead before)).len b) :
    VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.score (algebras trees (b,h,n/32))
      (t.call.parameters (hostScale t.call.dim))
      (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
        (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest)))
      (lane t.call (hostScale t.call.dim) n)=referenceTerm trees t before b h n := by
  have products : (fun d : Fin t.call.dim =>
      VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.products (algebras trees (b,h,n/32))
        (t.call.parameters (hostScale t.call.dim))
        (start t.call.grid (b,h,n/32) (initial t.call (metadata t (wordRead before))
          (Native.injectBuffers t.call (buffers t before)) (Native.injectState rest)))
        (⟨n%32,Nat.mod_lt _ (by decide)⟩,d,PUnit.unit))=productTerm t before b h n := by
    funext d
    unfold VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.products productTerm algebras
    rw [query_node (trees (b,h,n/32)) t before rest contract launch b h n hb hh d]
    rw [show (⟨n%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit)=
        lane t.call (hostScale t.call.dim) n from rfl]
    rw [key_node (trees (b,h,n/32)) t before rest contract launch b h n hb hh hn d]
    rfl
  exact congrArg (fun xs : Fin t.call.dim → Native.Term =>
    let plan := trees (b,h,n/32) (some .fp32) [32,t.call.dim] ⟨1,by simp⟩ Bool.false
      (⟨n%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit) (fun d => Native.shape (xs d))
    Native.Term.cast none .real .fp32 (.binary (some .fp32) .real .mul
      (plan.tree.expand (some .fp32) .real .add xs (.literal (some .fp32) .real 0))
      (.literal (some .fp32) .real (hostScale t.call.dim)))) products

#axiomsClean query_node
#axiomsClean key_node
#axiomsClean source_reference
#specNonCircular reference avoiding [VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,
  VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.score,NativeGrid.runPrograms]

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.KernelNative
