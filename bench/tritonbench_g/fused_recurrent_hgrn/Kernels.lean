import VeriTile.Triton

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 5000000

def forward (T D BD : Nat) (U S : Bool) : ComputeKernel :=
  let x : RegionName := "x"
  let g : RegionName := "g"
  let h0 : RegionName := "h0"
  let o : RegionName := "o"
  let ht : RegionName := "ht"
  triton {
  tl.ports inputs=[x,g,h0], outputs=[o,ht]
  tl.region x = tl.float32, g = tl.float32, h0 = tl.float32,
    o = tl.float32, ht = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_x = x + i_bh * $(T) * $(D) + o_d
  p_g = g + i_bh * $(T) * $(D) + o_d
  p_o = o + i_bh * $(T) * $(D) + o_d
  b_h = tl.zeros([$(BD)], dtype=tl.float32)
  if U {
    p_h0 = h0 + i_bh * $(D) + o_d
    b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_h = b_g * b_h + b_x
    tl.store(p_o, (b_h).to(p_o.dtype.element_ty), mask=mask)
    p_x += $(D)
    p_g += $(D)
    p_o += $(D)
  }
  if S {
    p_ht = ht + i_bh * $(D) + o_d
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask)
  }
}

/-- Fold the constexpr T*D stride before runtime pointer arithmetic.
The three forward pointers each use one runtime multiply instead of two. -/
def forwardStrideFolded (T D BD : Nat) (U S : Bool) : ComputeKernel :=
  let x : RegionName := "x"
  let g : RegionName := "g"
  let h0 : RegionName := "h0"
  let o : RegionName := "o"
  let ht : RegionName := "ht"
  triton {
  tl.ports inputs=[x,g,h0], outputs=[o,ht]
  tl.region x = tl.float32, g = tl.float32, h0 = tl.float32,
    o = tl.float32, ht = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_x = x + i_bh * $(T*D) + o_d
  p_g = g + i_bh * $(T*D) + o_d
  p_o = o + i_bh * $(T*D) + o_d
  b_h = tl.zeros([$(BD)], dtype=tl.float32)
  if U {
    p_h0 = h0 + i_bh * $(D) + o_d
    b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_h = b_g * b_h + b_x
    tl.store(p_o, (b_h).to(p_o.dtype.element_ty), mask=mask)
    p_x += $(D)
    p_g += $(D)
    p_o += $(D)
  }
  if S {
    p_ht = ht + i_bh * $(D) + o_d
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask)
  }
}

def backward (T D BD : Nat) (U V : Bool) : ComputeKernel :=
  let g : RegionName := "g"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let DO : RegionName := "do"
  let dht : RegionName := "dht"
  let dx : RegionName := "dx"
  let dg : RegionName := "dg"
  triton {
  tl.ports inputs=[g,o,h0,DO,dht], outputs=[dx,dg]
  tl.region g = tl.float32, o = tl.float32, h0 = tl.float32,
    DO = tl.float32, dht = tl.float32, dx = tl.float32, dg = tl.float32
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  b_dh = tl.zeros([$(BD)], dtype=tl.float32)
  if V {
    b_dh = tl.load(dht + i_bh * $(D) + o_d, mask=mask, other=0).to(tl.float32)
  }
  for step in range($(0), $(T), $(1)) {
    i = $(T) - $(1) - step
    p_g = g + (i_bh * $(T) + i) * $(D) + o_d
    p_dx = dx + (i_bh * $(T) + i) * $(D) + o_d
    p_dg = dg + (i_bh * $(T) + i) * $(D) + o_d
    p_do = DO + (i_bh * $(T) + i) * $(D) + o_d
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
    if i > $(0) {
      b_o = tl.load(o + (i_bh * $(T) + i - $(1)) * $(D) + o_d,
        mask=mask, other=0).to(tl.float32)
    } else {
      if U {
        b_o = tl.load(h0 + i_bh * $(D) + o_d, mask=mask, other=0).to(tl.float32)
      } else {
        b_o = tl.zeros([$(BD)], dtype=tl.float32)
      }
    }
    b_dh = b_dh + b_do
    b_dx = b_dh
    b_dg = b_dh * b_o
    b_dh = b_dh * b_g
    tl.store(p_dx, (b_dx).to(p_dx.dtype.element_ty), mask=mask)
    tl.store(p_dg, (b_dg).to(p_dg.dtype.element_ty), mask=mask)
  }
}

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
