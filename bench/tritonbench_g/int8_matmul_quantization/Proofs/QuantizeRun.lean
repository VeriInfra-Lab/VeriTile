import bench.tritonbench_g.int8_matmul_quantization.Proofs.QuantizeData

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

def quantScaleCode (q : QuantBuffers) (plan : Fin G → ReductionPlan K) (g : Fin G) : Code I F H F := do
  let maximum ← (plan g).tree.code (fun k => q.inputAddress g k)
  let maximum ← if q.initialMaximumZero then do
      let zero ← emit .fzero
      emit (.maximum zero maximum)
    else pure maximum
  emit (.scale127 maximum)
theorem eval_quantScaleCode (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (g : Fin G) (m : Memory F H) :
    (quantScaleCode q plan g).eval a m=(quantScale a q plan m g,m) := by
  unfold quantScaleCode quantScale
  simp only [bind,eval_bind,Reduction.eval_code,Prod.fst,Prod.snd]
  cases q.initialMaximumZero <;> rfl

def secondPassCode (q : QuantBuffers) (g : Fin G) (scale : F) : List (Fin K) → Code I F H Unit
  | [] => pure ()
  | k::ks => do
      let value ← emit (.read32 (q.inputAddress g k))
      let quotient ← emit (.divide value scale)
      let integer ← emit (.integer8 quotient)
      let _ ← emit (.store8 (q.dataAddress (g,k)) integer)
      secondPassCode q g scale ks

theorem eval_secondPassCode (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (g : Fin G) (scale : F) (ks : List (Fin K)) (m before : Memory F H)
    (same : QuantInputsAgree q m before) :
    (secondPassCode q g scale ks).eval a m=((),writeAll (fun k => q.dataAddress (g,k))
      (fun k => .inl (a.integer8 (a.divide (read32 a.toOps before (q.inputAddress g k)) scale))) ks m) := by
  induction ks generalizing m with
  | nil => rfl
  | cons k ks ih =>
    have load : read32 a.toOps m (q.inputAddress g k)=read32 a.toOps before (q.inputAddress g k) := by
      unfold read32 Int8DequantMatmul.read32
      rw [same _ rfl]
    simp only [secondPassCode,bind,eval_bind,eval_emit,Event.eval,Prod.fst,Prod.snd,load]
    simpa only [write8,put,QuantBuffers.writeAddress,writeAll] using
      ih (put m (q.writeAddress (.inl (g,k)))
        (.inl (a.integer8 (a.divide (read32 a.toOps before (q.inputAddress g k)) scale))))
        (quant_put_input q sep (.inl (g,k)) _ m before same)

def quantScaleStoreCode (q : QuantBuffers) (g : Fin G) (value : F) : Code I F H Unit :=
  match q.scalePrecision with
  | .half => do
      let half ← emit (.narrow value)
      emit (.store (q.scaleAddress g) half)
  | .single => emit (.store32 (q.scaleAddress g) value)
theorem eval_quantScaleStoreCode (a : Ops I F H) (q : QuantBuffers) (g : Fin G)
    (value : F) (m : Memory F H) :
    (quantScaleStoreCode q g value).eval a m=
      ((),put m (q.scaleAddress g) (.inr (match q.scalePrecision with
        | .half => .inr (a.narrow value) | .single => .inl value))) := by
  cases mode : q.scalePrecision <;>
    simp [quantScaleStoreCode,mode,bind,eval_bind,eval_emit,Event.eval,
      put,write32,write16,Int8DequantMatmul.write16] <;> rfl

def quantGroupCode (q : QuantBuffers) (plan : Fin G → ReductionPlan K) (g : Fin G) : Code I F H Unit := do
  let scale ← quantScaleCode q plan g
  let _ ← secondPassCode q g scale (List.finRange K)
  quantScaleStoreCode q g scale
def quantGroupIndices (g : Fin G) : List (QuantIndex G K) :=
  (List.finRange K).map (fun k => .inl (g,k))++[.inr g]

theorem eval_quantGroupCode (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (plan : Fin G → ReductionPlan K) (g : Fin G) (m before : Memory F H)
    (same : QuantInputsAgree q m before) :
    (quantGroupCode q plan g).eval a m=
      ((),writeAll q.writeAddress (quantExpected a q plan before) (quantGroupIndices g) m) := by
  simp only [quantGroupCode,bind,eval_bind,eval_quantScaleCode,Prod.fst,Prod.snd,
    quantScale_current a q plan m before same,
    eval_secondPassCode a q sep g _ _ m before same,eval_quantScaleStoreCode,
    quantGroupIndices,writeAll_append,writeAll_map,writeAll,quantExpected,quantValue,
    QuantBuffers.writeAddress]
  rfl

theorem quant_writeAll_input (q : QuantBuffers) (sep : q.Separate)
    (values : QuantIndex G K → Cell F H) (xs : List (QuantIndex G K)) (m before : Memory F H)
    (same : QuantInputsAgree q m before) :
    QuantInputsAgree q (writeAll q.writeAddress values xs m) before := by
  induction xs generalizing m with
  | nil => exact same
  | cons x xs ih => exact ih _ (quant_put_input q sep x _ m before same)

def quantGridCode (q : QuantBuffers) (plan : Fin G → ReductionPlan K) : List (Fin G) → Code I F H Unit
  | [] => pure ()
  | g::gs => do
      let _ ← quantGroupCode q plan g
      quantGridCode q plan gs
theorem eval_quantGridCode (a : Ops I F H) (q : QuantBuffers) (sep : q.Separate)
    (plan : Fin G → ReductionPlan K) (groups : List (Fin G)) (m before : Memory F H)
    (same : QuantInputsAgree q m before) :
    (quantGridCode q plan groups).eval a m=
      ((),writeAll q.writeAddress (quantExpected a q plan before) (groups.flatMap quantGroupIndices) m) := by
  induction groups generalizing m with
  | nil => rfl
  | cons g gs ih =>
    simp only [quantGridCode,bind,eval_bind,eval_quantGroupCode a q sep plan g m before same,Prod.fst,Prod.snd]
    rw [ih _ (quant_writeAll_input q sep _ _ m before same)]
    simp only [List.flatMap_cons,writeAll_append]

#axiomsClean eval_secondPassCode
#axiomsClean eval_quantGroupCode
#axiomsClean eval_quantGridCode
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
