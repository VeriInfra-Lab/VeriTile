import bench.repairs.tritonbench_g.chunk_retention.Proofs.InputValues

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
open VeriTile Triton FP.Structural KernelPacket
noncomputable section
variable {α : Type}

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

#axiomsClean history_congr
#axiomsClean reverse_congr
#axiomsClean output_congr
#axiomsClean query_congr
#axiomsClean key_congr
#axiomsClean value_congr
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PublicIO
