import bench.tritonbench_g.chunk_retention_ops.Proofs.IOInputValues

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section
variable {α : Type} [Inhabited α]

/-- Extensionality at masked logical inputs; unused coordinates and unrelated
fields have no bearing on the scalar graph. -/
theorem history_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (h : Nat)
    (hk : kAt ops c x h=kAt ops c y h) (hv : vAt ops c x h=vAt ops c y h)
    (hs : masked ops c.K c.V (x.seed h)=masked ops c.K c.V (y.seed h)) (n : Nat) :
    Reference.history ops c x h n=Reference.history ops c y h n := by
  induction n with
  | zero => simp only [Reference.history,hs]
  | succ n ih =>
    funext e j
    simp only [Reference.history,Body.nextH,hk,hv,ih]

theorem reverse_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (scale : α) (h : Nat)
    (hq : qAt ops c x h=qAt ops c y h) (hg : gAt ops c x h=gAt ops c y h)
    (hs : masked ops c.K c.V (x.finalIncoming h)=masked ops c.K c.V (y.finalIncoming h)) (n : Nat) :
    Reference.reverseCarry ops c x scale h n=Reference.reverseCarry ops c y scale h n := by
  induction n with
  | zero => simp only [Reference.reverseCarry,hs]
  | succ n ih =>
    funext e j
    simp only [Reference.reverseCarry,Body.nextDH,hq,hg,ih]

theorem output_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (scale : α)
    (h n : Nat) (hx hy : Tile α)
    (hq : qAt ops c x h=qAt ops c y h) (hk : kAt ops c x h=kAt ops c y h)
    (hv : vAt ops c x h=vAt ops c y h)
    (hist : masked ops c.K c.V hx=masked ops c.K c.V hy) :
    Body.output ops c x scale h n hx=Body.output ops c y scale h n hy := by
  funext e j
  simp only [Body.output,Body.score,Body.outputCarry,hq,hk,hv,hist]

theorem query_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (scale : α)
    (h n : Nat) (hx hy : Tile α)
    (hk : kAt ops c x h=kAt ops c y h) (hv : vAt ops c x h=vAt ops c y h)
    (hg : gAt ops c x h=gAt ops c y h)
    (hist : masked ops c.K c.V hx=masked ops c.K c.V hy) :
    Body.dq ops c x scale h n hx=Body.dq ops c y scale h n hy := by
  funext e j
  simp only [Body.dq,Body.ds,hk,hv,hg,hist]

theorem key_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (scale : α)
    (h n : Nat) (hx hy : Tile α)
    (hq : qAt ops c x h=qAt ops c y h) (hv : vAt ops c x h=vAt ops c y h)
    (hg : gAt ops c x h=gAt ops c y h)
    (hist : masked ops c.K c.V hx=masked ops c.K c.V hy) :
    Body.dk ops c x scale h n hx=Body.dk ops c y scale h n hy := by
  funext e j
  simp only [Body.dk,Body.ds,hq,hv,hg,hist]

theorem value_congr (ops : Ops α) (c : Shape) (x y : Whole001.Inputs α) (scale : α)
    (block h n : Nat) (hx hy : Tile α)
    (hq : qAt ops c x h=qAt ops c y h) (hk : kAt ops c x h=kAt ops c y h)
    (hg : gAt ops c x h=gAt ops c y h)
    (hist : masked ops c.K c.V hx=masked ops c.K c.V hy) :
    Body.dvPartial ops c x scale block h n hx=Body.dvPartial ops c y scale block h n hy := by
  funext e j
  simp only [Body.dvPartial,Body.partialScore,hq,hk,hg,hist]

/-- Transport the existing scalar reference through the actual finite input
binding. All padded reads remain zero-masked and all optional channels retain
their original guards. -/
theorem scalar_transport (ops : Ops α) (scale : α) (a : Call) (xs : Inputs α a.shape)
    (read : RegionName→Nat→α) (hr : Reads a xs read)
    (port : Fin (outputs a.shape).length) (j : TileIndex (outputs a.shape)[port].shape)
    (active : outputActive a port j) :
    scalarOutputs ops scale a (readInputs ops.zero a read)
      (readHistory a read Bool.false) (readHistory a read Bool.true) port j=
      scalarFormula ops scale a xs port j := by
  rcases port with ⟨port,hp⟩
  have hport : port<8 := hp
  interval_cases port
  · have hs : masked ops a.shape.K a.shape.V ((readInputs ops.zero a read).seed a.pid.z)=
        masked ops a.shape.K a.shape.V ((logicalInputs ops.zero a xs).seed a.pid.z) := by
      funext e v
      unfold masked
      split_ifs with h
      · exact hr.seed_eq ops.zero active.1 e v h.1 h.2
      · rfl
    exact congrArg ops.cast (congrFun (congrFun
      (history_congr ops a.shape _ _ a.pid.z
        (hr.kAt_eq ops (Or.inl active.1)) (hr.vAt_eq ops (Or.inl active.1)) hs j.1.val)
      (a.pid.x*a.shape.BK+j.2.1.val)) (a.pid.y*a.shape.BV+j.2.2.1.val))
  · have hs : masked ops a.shape.K a.shape.V ((readInputs ops.zero a read).seed a.pid.z)=
        masked ops a.shape.K a.shape.V ((logicalInputs ops.zero a xs).seed a.pid.z) := by
      funext e v
      unfold masked
      split_ifs with h
      · exact hr.seed_eq ops.zero active.1 e v h.1 h.2
      · rfl
    exact congrArg ops.cast (congrFun (congrFun
      (history_congr ops a.shape _ _ a.pid.z
        (hr.kAt_eq ops (Or.inl active.1)) (hr.vAt_eq ops (Or.inl active.1)) hs a.shape.chunks)
      (a.pid.x*a.shape.BK+j.1.val)) (a.pid.y*a.shape.BV+j.2.1.val))
  · exact congrFun (congrFun (output_congr ops a.shape _ _ scale a.pid.z a.pid.y _ _
      (hr.qAt_eq ops (Or.inl active.1)) (hr.kAt_eq ops (Or.inr (Or.inl active.1)))
      (hr.vAt_eq ops (Or.inr (Or.inl active.1))) (hr.masked_history_eq ops Bool.false (Or.inl active.1)))
      j.1.val) (a.pid.x*a.shape.BV+j.2.1.val)
  · have hs : masked ops a.shape.K a.shape.V ((readInputs ops.zero a read).finalIncoming a.pid.z)=
        masked ops a.shape.K a.shape.V ((logicalInputs ops.zero a xs).finalIncoming a.pid.z) := by
      funext e v
      unfold masked
      split_ifs with h
      · exact hr.finalIncoming_eq ops.zero active.1 e v h.1 h.2
      · rfl
    exact congrArg ops.cast (congrFun (congrFun
      (reverse_congr ops a.shape _ _ scale a.pid.z
        (hr.qAt_eq ops (Or.inr (Or.inl active.1))) (hr.gAt_eq ops (Or.inl active.1)) hs j.1.val)
      (a.pid.x*a.shape.BK+j.2.1.val)) (a.pid.y*a.shape.BV+j.2.2.1.val))
  · have hs : masked ops a.shape.K a.shape.V ((readInputs ops.zero a read).finalIncoming a.pid.z)=
        masked ops a.shape.K a.shape.V ((logicalInputs ops.zero a xs).finalIncoming a.pid.z) := by
      funext e v
      unfold masked
      split_ifs with h
      · exact hr.finalIncoming_eq ops.zero active.1 e v h.1 h.2
      · rfl
    exact congrArg ops.cast (congrFun (congrFun
      (reverse_congr ops a.shape _ _ scale a.pid.z
        (hr.qAt_eq ops (Or.inr (Or.inl active.1))) (hr.gAt_eq ops (Or.inl active.1)) hs a.shape.chunks)
      (a.pid.x*a.shape.BK+j.1.val)) (a.pid.y*a.shape.BV+j.2.1.val))
  · exact congrFun (congrFun (query_congr ops a.shape _ _ scale a.pid.z a.pid.y _ _
      (hr.kAt_eq ops (Or.inr (Or.inr active.1))) (hr.vAt_eq ops (Or.inr (Or.inr active.1)))
      (hr.gAt_eq ops (Or.inr active.1)) (hr.masked_history_eq ops Bool.false (Or.inr active.1)))
      j.1.val) (a.pid.x*a.shape.BK+j.2.1.val)
  · exact congrFun (congrFun (key_congr ops a.shape _ _ scale a.pid.z a.pid.y _ _
      (hr.qAt_eq ops (Or.inr (Or.inr active.1))) (hr.vAt_eq ops (Or.inr (Or.inr active.1)))
      (hr.gAt_eq ops (Or.inr active.1)) (hr.masked_history_eq ops Bool.true active.1))
      j.1.val) (a.pid.x*a.shape.BK+j.2.1.val)
  · exact congrFun (congrFun (value_congr ops a.shape _ _ scale a.pid.x a.pid.z a.pid.y _ _
      (hr.qAt_eq ops (Or.inr (Or.inr active.1))) (hr.kAt_eq ops (Or.inr (Or.inr active.1)))
      (hr.gAt_eq ops (Or.inr active.1)) (hr.masked_history_eq ops Bool.true active.1))
      j.2.1.val) (j.1.val*a.shape.BV+j.2.2.1.val)

#axiomsClean scalar_transport
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PublicIO
