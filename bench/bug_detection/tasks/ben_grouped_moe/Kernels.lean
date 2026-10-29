import VeriTile.Triton.DSL

/-! Explicit Triton lowerings. Native source/backend correspondence is an
unsolved goal in Proofs/Backend.lean; compiling these sources does not prove it. -/

namespace VeriTile.Bench.BugDetection.ben_grouped_moe
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
namespace Kernels

def original : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let router_ptr : RegionName := "router"
  let gate_ptr : RegionName := "gate"
  let up_ptr : RegionName := "up"
  let down_ptr : RegionName := "down"
  let shared_gate_ptr : RegionName := "shared_gate"
  let shared_up_ptr : RegionName := "shared_up"
  let shared_down_ptr : RegionName := "shared_down"
  let out_ptr : RegionName := "out"
  let route_index_ptr : RegionName := "route_index"
  let route_score_ptr : RegionName := "route_score"
  let route_prob_ptr : RegionName := "route_prob"
  let packed_token_ptr : RegionName := "packed_token"
  let packed_weight_ptr : RegionName := "packed_weight"
  triton {
    t = tl.arange($(16))
    h = tl.arange($(2048))
    e = tl.arange($(128))
    j = tl.arange($(6))
    k = tl.arange($(640))
    x = tl.load($(x_ptr) + t[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    router = tl.load($(router_ptr) + e[None, :] * $(2048) + h[:, None], dtype=tl.float32)
    logits = tl.dot(x.to(tl.float32), router.to(tl.float32)).to(tl.float32)
    scores = tl.sigmoid(logits)
    remaining = scores
    for slot in range($(6)) {
      index = tl.argmax(remaining, axis=1)
      top = tl.max(remaining, axis=1)
      tl.store($(route_index_ptr) + t * $(6) + slot, index)
      tl.store($(route_score_ptr) + t * $(6) + slot, top)
      remaining = tl.where(e[None, :] == index[:, None], -inf, remaining)
    }
    top = tl.load($(route_score_ptr) + t[:, None] * $(6) + j[None, :], dtype=tl.float32)
    top = top / (tl.sum(top, axis=1, keep_dims=true) + 0.00000000000000000001)
    prob = top * 2.826
    prob = prob.to(tl.bfloat16)
    tl.store($(route_prob_ptr) + t[:, None] * $(6) + j[None, :], prob)
    tl.store($(out_ptr) + t[:, None] * $(2048) + h[None, :], tl.zeros([$(16), $(2048)], dtype=tl.bfloat16))
    for token in range($(16)) {
      for slot in range($(6)) {
        expert = tl.load($(route_index_ptr) + token * $(6) + slot, dtype=tl.uint32)
        weight = tl.load($(route_prob_ptr) + token * $(6) + slot, dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + token * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + token * $(2048) + h[None, :], dtype=tl.bfloat16)
        y = (out + y).to(tl.bfloat16)
        tl.store($(out_ptr) + token * $(2048) + h[None, :], y)
      }
    }
    s = tl.arange($(512))
    sg = tl.load($(shared_gate_ptr) + s[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    su = tl.load($(shared_up_ptr) + s[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    sd = tl.load($(shared_down_ptr) + h[None, :] * $(512) + s[:, None], dtype=tl.bfloat16)
    g = tl.dot(x, sg).to(tl.bfloat16)
    u = tl.dot(x, su).to(tl.bfloat16)
    a = silu(g).to(tl.bfloat16)
    a = (a * u).to(tl.bfloat16)
    shared = tl.dot(a, sd).to(tl.bfloat16)
    out = tl.load($(out_ptr) + t[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    tl.store($(out_ptr) + t[:, None] * $(2048) + h[None, :], (out + shared).to(tl.bfloat16))
  }

def candidate : ComputeKernel :=
  let x_ptr : RegionName := "x"
  let router_ptr : RegionName := "router"
  let gate_ptr : RegionName := "gate"
  let up_ptr : RegionName := "up"
  let down_ptr : RegionName := "down"
  let shared_gate_ptr : RegionName := "shared_gate"
  let shared_up_ptr : RegionName := "shared_up"
  let shared_down_ptr : RegionName := "shared_down"
  let out_ptr : RegionName := "out"
  let route_index_ptr : RegionName := "route_index"
  let route_score_ptr : RegionName := "route_score"
  let route_prob_ptr : RegionName := "route_prob"
  let packed_token_ptr : RegionName := "packed_token"
  let packed_weight_ptr : RegionName := "packed_weight"
  triton {
    t = tl.arange($(16))
    h = tl.arange($(2048))
    e = tl.arange($(128))
    j = tl.arange($(6))
    k = tl.arange($(640))
    x = tl.load($(x_ptr) + t[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    router = tl.load($(router_ptr) + e[None, :] * $(2048) + h[:, None], dtype=tl.float32)
    logits = tl.dot(x.to(tl.float32), router.to(tl.float32)).to(tl.float32)
    scores = tl.sigmoid(logits)
    remaining = scores
    for slot in range($(6)) {
      index = tl.argmax(remaining, axis=1)
      top = tl.max(remaining, axis=1)
      tl.store($(route_index_ptr) + t * $(6) + slot, index)
      tl.store($(route_score_ptr) + t * $(6) + slot, top)
      remaining = tl.where(e[None, :] == index[:, None], -inf, remaining)
    }
    top = tl.load($(route_score_ptr) + t[:, None] * $(6) + j[None, :], dtype=tl.float32)
    top = top / (tl.sum(top, axis=1, keep_dims=true) + 0.00000000000000000001)
    prob = top * 2.826
    prob = prob.to(tl.bfloat16)
    tl.store($(route_prob_ptr) + t[:, None] * $(6) + j[None, :], prob)
    tl.store($(out_ptr) + t[:, None] * $(2048) + h[None, :], tl.zeros([$(16), $(2048)], dtype=tl.bfloat16))
    for expert in range($(128)) {
      count = $(0)
      for token in range($(16)) {
        active = false
        weight = tl.full([], 0.0, dtype=tl.bfloat16)
        for slot in range($(6)) {
          route = tl.load($(route_index_ptr) + token * $(6) + slot, dtype=tl.uint32)
          prob = tl.load($(route_prob_ptr) + token * $(6) + slot, dtype=tl.bfloat16)
          selected = route == expert
          active = active | selected
          weight = weight + tl.where(selected, prob, 0.0)
        }
        if active {
          tl.store($(packed_token_ptr) + count, token)
          tl.store($(packed_weight_ptr) + count, weight)
          count = count + $(1)
        }
      }
      if count == $(1) {
        rr = tl.arange($(1))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(2) {
        rr = tl.arange($(2))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(3) {
        rr = tl.arange($(3))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(4) {
        rr = tl.arange($(4))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(5) {
        rr = tl.arange($(5))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(6) {
        rr = tl.arange($(6))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(7) {
        rr = tl.arange($(7))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(8) {
        rr = tl.arange($(8))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(9) {
        rr = tl.arange($(9))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(10) {
        rr = tl.arange($(10))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(11) {
        rr = tl.arange($(11))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(12) {
        rr = tl.arange($(12))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(13) {
        rr = tl.arange($(13))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(14) {
        rr = tl.arange($(14))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(15) {
        rr = tl.arange($(15))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
      if count == $(16) {
        rr = tl.arange($(16))
        tokens = tl.load($(packed_token_ptr) + rr, dtype=tl.uint32)
        weight = tl.load($(packed_weight_ptr) + rr[:, None], dtype=tl.bfloat16)
        xin = tl.load($(x_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        wg = tl.load($(gate_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        wu = tl.load($(up_ptr) + expert * $(1310720) + k[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
        g = tl.dot(xin, wg).to(tl.bfloat16)
        u = tl.dot(xin, wu).to(tl.bfloat16)
        a = silu(g).to(tl.bfloat16)
        act = (a * u).to(tl.bfloat16)
        wd = tl.load($(down_ptr) + expert * $(1310720) + h[None, :] * $(640) + k[:, None], dtype=tl.bfloat16)
        y = tl.dot(act, wd).to(tl.bfloat16)
        y = (weight * y).to(tl.bfloat16)
        out = tl.load($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
        tl.store($(out_ptr) + tokens[:, None] * $(2048) + h[None, :], (out + y).to(tl.bfloat16))
      }
    }
    s = tl.arange($(512))
    sg = tl.load($(shared_gate_ptr) + s[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    su = tl.load($(shared_up_ptr) + s[None, :] * $(2048) + h[:, None], dtype=tl.bfloat16)
    sd = tl.load($(shared_down_ptr) + h[None, :] * $(512) + s[:, None], dtype=tl.bfloat16)
    g = tl.dot(x, sg).to(tl.bfloat16)
    u = tl.dot(x, su).to(tl.bfloat16)
    a = silu(g).to(tl.bfloat16)
    a = (a * u).to(tl.bfloat16)
    shared = tl.dot(a, sd).to(tl.bfloat16)
    out = tl.load($(out_ptr) + t[:, None] * $(2048) + h[None, :], dtype=tl.bfloat16)
    tl.store($(out_ptr) + t[:, None] * $(2048) + h[None, :], (out + shared).to(tl.bfloat16))
  }

end Kernels
end
end VeriTile.Bench.BugDetection.ben_grouped_moe
