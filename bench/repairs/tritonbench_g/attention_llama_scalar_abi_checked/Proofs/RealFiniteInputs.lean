import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealFiniteContext

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteInputs
open VeriTile Triton Source RealFiniteContext RealLayout
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult MaskCoverage
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem query_read (c : Config) (scale : ℝ) (s : BlockState)
    (i : Index B c.NH c.H c.D) (d : Fin c.D) :
    (contextFor (context c scale s) i).query (HostResult.lane (c:=context c scale s) i,d,())=
      s.readMem c.Q (IOLayout.query c (i.batch,i.head,i.query,d,())) := by
  change (if i.query.val/64*64+i.query.val%64<c.H then
    s.readMem c.Q (alBase (contextFor (context c scale s) i).initial c.NH c.sqz c.sqh+
      (i.query.val/64*64+i.query.val%64)*c.sqm+d.val*c.sqk) else 0)=_
  have hr : i.query.val/64*64+i.query.val%64=i.query.val := by omega
  rw [hr,if_pos i.query.isLt,base_eq c scale s i c.sqz c.sqh]
  rfl

theorem key_read (c : Config) (scale : ℝ) (s : BlockState)
    (i : Index B c.NH c.H c.D) (k : Fin c.N) (d : Fin c.D) :
    (contextFor (context c scale s) i).key k.val d=
      s.readMem c.K (IOLayout.key c (i.batch,i.head,k,d,())) := by
  change s.readMem c.K (alBase (contextFor (context c scale s) i).initial c.NH c.skz c.skh+
    k.val*c.skn+d.val*c.skk)=_
  rw [base_eq c scale s i c.skz c.skh]
  rfl

theorem value_read (c : Config) (scale : ℝ) (s : BlockState)
    (i : Index B c.NH c.H c.D) (k : Fin c.N) (d : Fin c.D) :
    (contextFor (context c scale s) i).value k.val d=
      s.readMem c.V (IOLayout.value c (i.batch,i.head,k,d,())) := by
  change s.readMem c.V (alBase (contextFor (context c scale s) i).initial c.NH c.svz c.svh+
    k.val*c.svk+d.val*c.svn)=_
  rw [base_eq c scale s i c.svz c.svh]
  rfl

theorem input_query (B : Nat) (c : Config) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals) (j : QueryIndex B c) :
    s.readMem c.Q (IOLayout.query c j)=vals 0 (querySlot B c j) := by
  have h:=hin 0 (querySlot B c j) (by simp [memory])
  change s.readMem c.Q (IOLayout.query c (queryIndex B c (querySlot B c j)))=vals 0 (querySlot B c j) at h
  simpa only [queryIndex_slot] using h

theorem input_key (B : Nat) (c : Config) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals) (hf : c.fp8=Bool.false) (j : KeyIndex B c) :
    s.readMem c.K (IOLayout.key c j)=vals 1 (keySlot B c j) := by
  have h:=hin 1 (keySlot B c j) (by simp [memory,hf])
  change s.readMem c.K (IOLayout.key c (keyIndex B c (keySlot B c j)))=vals 1 (keySlot B c j) at h
  simpa only [keyIndex_slot] using h

theorem input_value (B : Nat) (c : Config) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals) (hf : c.fp8=Bool.false) (j : KeyIndex B c) :
    s.readMem c.V (IOLayout.value c j)=vals 2 (keySlot B c j) := by
  have h:=hin 2 (keySlot B c j) (by simp [memory,hf])
  change s.readMem c.V (IOLayout.value c (keyIndex B c (keySlot B c j)))=vals 2 (keySlot B c j) at h
  simpa only [keyIndex_slot] using h

theorem input_byte_key (B : Nat) (c : Config) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals) (hf : c.fp8=Bool.true) (j : KeyIndex B c) :
    s.readMemValue .int c.K (IOLayout.key c j)=vals 3 (keySlot B c j) := by
  have h:=hin 3 (keySlot B c j) (by simp [memory,hf])
  change s.readMemValue .int c.K (IOLayout.key c (keyIndex B c (keySlot B c j)))=vals 3 (keySlot B c j) at h
  simpa only [keyIndex_slot] using h

theorem input_byte_value (B : Nat) (c : Config) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals) (hf : c.fp8=Bool.true) (j : KeyIndex B c) :
    s.readMemValue .int c.V (IOLayout.value c j)=vals 4 (keySlot B c j) := by
  have h:=hin 4 (keySlot B c j) (by simp [memory,hf])
  change s.readMemValue .int c.V (IOLayout.value c (keyIndex B c (keySlot B c j)))=vals 4 (keySlot B c j) at h
  simpa only [keyIndex_slot] using h

/-- Decoding is an input interpretation only. Both branches bind the same
finite arrays at the original strided addresses, including permitted aliases. -/
theorem binding (B : Nat) (c : Config) (scale : ℝ) (vals : (memory B c).Ctx) (s : BlockState)
    (hin : (memory B c).InputBinding s vals)
    (sep : c.fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V) :
    HostResult.Binding (Backing.realView (context c scale s) c.fp8)
      (RealMathematical.inputs B c vals) := by
  let view:=if c.fp8 then FP8RealMemory.decoded c.K c.V s else s
  change HostResult.Binding (context c scale view) _
  constructor
  · intro i d
    rw [query_read]
    change view.readMem c.Q _=vals 0 _
    cases hf:c.fp8
    · simpa only [view,hf,Bool.false_eq_true,↓reduceIte] using input_query B c vals s hin _
    · have hs:=sep hf
      simp only [view,hf,↓reduceIte]
      change (FP8RealMemory.decoded c.K c.V s).readMem c.Q _=_
      unfold BlockState.readMem
      rw [FP8RealMemory.outside_memory c.K c.V s c.Q _ hs.1 hs.2]
      exact input_query B c vals s hin _
  · intro i k d hk
    change k<c.N at hk
    rw [key_read c scale view i ⟨k,hk⟩ d]
    change view.readMem c.K _=(RealMathematical.inputs B c vals).key i.batch i.head k d
    cases hf:c.fp8
    · simpa [RealMathematical.inputs,view,hf,hk] using
        input_key B c vals s hin hf (i.batch,i.head,⟨k,hk⟩,d,())
    · simp only [view,hf,↓reduceIte]
      change (FP8RealMemory.decoded c.K c.V s).readMem c.K _=_
      rw [FP8RealMemory.read_decoded c.K c.V s c.K _ (Or.inl rfl)]
      simpa [RealMathematical.inputs,hk,hf,FP8RealMemory.value,decode] using
        congrArg decode (input_byte_key B c vals s hin hf (i.batch,i.head,⟨k,hk⟩,d,()))
  · intro i k d hk
    change k<c.N at hk
    rw [value_read c scale view i ⟨k,hk⟩ d]
    change view.readMem c.V _=(RealMathematical.inputs B c vals).value i.batch i.head k d
    cases hf:c.fp8
    · simpa [RealMathematical.inputs,view,hf,hk] using
        input_value B c vals s hin hf (i.batch,i.head,⟨k,hk⟩,d,())
    · simp only [view,hf,↓reduceIte]
      change (FP8RealMemory.decoded c.K c.V s).readMem c.V _=_
      rw [FP8RealMemory.read_decoded c.K c.V s c.V _ (Or.inr rfl)]
      simpa [RealMathematical.inputs,hk,hf,FP8RealMemory.value,decode] using
        congrArg decode (input_byte_value B c vals s hin hf (i.batch,i.head,⟨k,hk⟩,d,()))

/-- The finite byte domain is precisely restricted to source-reached masked
blocks. Unvisited keys do not acquire a new numerical premise. -/
theorem byte_domain (B : Nat) (c : Config) (scale : ℝ) (vals : (memory B c).Ctx)
    (s : BlockState) (hin : (memory B c).InputBinding s vals)
    (hd : domain B c vals) (hf : c.fp8=Bool.true) :
    ByteHostResult.Domain (B:=B) (heads:=c.NH) (context c scale s) := by
  have reached (i : Index B c.NH c.H c.D) (k : Nat) (j : Fin 64)
      (hr : k*64<MaskCoverage.stop c.N c.causal c.SP (i.query.val/64))
      (hn : k*64+j.val<c.N) : reachedKey c ⟨k*64+j.val,hn⟩ := by
    refine ⟨⟨i.query.val/64,(query_program_exists c.H i.query.val i.query.isLt).1⟩,?_⟩
    change (k*64+j.val)/64*64<_
    have hj:=j.isLt
    have eq : (k*64+j.val)/64=k := by omega
    rw [eq]
    exact hr
  constructor
  · intro i k hr j d hn
    have hr' : k*64<MaskCoverage.stop c.N c.causal c.SP (i.query.val/64) := by
      simpa only [FP8RealSetup.stop,MaskCoverage.stop,contextFor,context,Nat.zero_add] using hr
    have hn' : k*64+j.val<c.N := by simpa only [Nat.zero_add] using hn
    have h:= (hd hf (i.batch,i.head,⟨k*64+j.val,hn'⟩,d,()) (reached i k j hr' hn')).1
    change FP8RealMemory.FiniteByte (s.readMemValue .int c.K
      (alBase (contextFor (context c scale s) i).initial c.NH c.skz c.skh+
        (0+k*64+j.val)*c.skn+d.val*c.skk))
    rw [base_eq c scale s i c.skz c.skh]
    simpa only [IOLayout.key,IOLayout.offset,Nat.zero_add,
      ←input_byte_key B c vals s hin hf (i.batch,i.head,⟨k*64+j.val,hn'⟩,d,())] using h
  · intro i k hr j d hn
    have hr' : k*64<MaskCoverage.stop c.N c.causal c.SP (i.query.val/64) := by
      simpa only [FP8RealSetup.stop,MaskCoverage.stop,contextFor,context,Nat.zero_add] using hr
    have hn' : k*64+j.val<c.N := by simpa only [Nat.zero_add] using hn
    have h:= (hd hf (i.batch,i.head,⟨k*64+j.val,hn'⟩,d,()) (reached i k j hr' hn')).2
    change FP8RealMemory.FiniteByte (s.readMemValue .int c.V
      (alBase (contextFor (context c scale s) i).initial c.NH c.svz c.svh+
        (0+k*64+j.val)*c.svk+d.val*c.svn))
    rw [base_eq c scale s i c.svz c.svh]
    simpa only [IOLayout.value,IOLayout.offset,Nat.zero_add,
      ←input_byte_value B c vals s hin hf (i.batch,i.head,⟨k*64+j.val,hn'⟩,d,())] using h

#axiomsClean binding
#axiomsClean byte_domain
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.RealFiniteInputs
