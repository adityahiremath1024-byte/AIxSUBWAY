#!/usr/bin/env python3


import heapq
import json
import math
import sys
import traceback

INF = math.inf

def log(message):
    """Debug message. Goes to stderr, so it never breaks the game."""
    print(message, file=sys.stderr, flush=True)


# ===========================================================================
#  >>> YOUR CODE STARTS HERE <<<
# ===========================================================================

# ----- Tunable constants -----
POLICE_DANGER_MULTIPLIER = 2.2    # how far beyond radius we start penalizing
DANGER_EDGE_SCALE = 0.12          # how much danger affects edge cost
NODE_DANGER_PENALTY = 250.0       # max extra cost for a node in danger zone
DEPOSIT_FRAC_NORMAL = 0.50        # deposit when this full normally
DEPOSIT_FRAC_URGENT = 0.15        # deposit when few ticks left
TICKS_URGENT = 45                 # when to become urgent about depositing
TICKS_CRITICAL = 20               # when to rush to deposit no matter what
PREDICT_TICKS = 4                 # how many ticks ahead to predict police
MIN_DROP_VALUE = 10.0             # ignore drops worth less than this
SAFETY_MARGIN = 1.25              # multiply police radius for safety
HYSTERESIS = 1.4                  # new target must be this much better to switch
TOTAL_TICKS = 300


class BotState:
    """Persistent state across ticks."""
    def __init__(self):
        self.target_drop = {}     # runner_id -> drop_id
        self.graph_built = False
        self.adj = {}             # node_id -> [(neighbor, base_weight), ...]

bot_state = BotState()


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _node_danger(nx, ny, police):
    """How dangerous is a node position? Returns 0..NODE_DANGER_PENALTY+."""
    danger = 0.0
    for cop in police:
        radius = cop["radius"] * SAFETY_MARGIN
        extended = radius * POLICE_DANGER_MULTIPLIER

        # Current position
        d = _dist(nx, ny, cop["x"], cop["y"])
        if d <= radius:
            danger += NODE_DANGER_PENALTY * 2.5
        elif d <= extended:
            closeness = 1.0 - (d - radius) / (extended - radius)
            danger += NODE_DANGER_PENALTY * closeness * closeness

        # Predicted future positions (lighter weight)
        vx, vy = cop["vx"], cop["vy"]
        for t in range(1, PREDICT_TICKS + 1):
            px = max(10, min(490, cop["x"] + vx * t))
            py = max(10, min(490, cop["y"] + vy * t))
            d_fut = _dist(nx, ny, px, py)
            decay = 0.6 ** t
            if d_fut <= radius:
                danger += NODE_DANGER_PENALTY * decay
            elif d_fut <= extended:
                closeness = 1.0 - (d_fut - radius) / (extended - radius)
                danger += NODE_DANGER_PENALTY * 0.25 * closeness * decay

    return danger


def _dijkstra(neighbors, nodes, start, police, use_danger=True):
    """Dijkstra with optional police danger penalties. Returns (cost, first_step, hops)."""
    cost_to = {start: 0.0}
    first_step = {start: start}
    hops = {start: 0}
    pq = [(0.0, start)]

    while pq:
        cost, node = heapq.heappop(pq)
        if cost > cost_to.get(node, INF):
            continue
        h = hops[node]
        for nxt, w in neighbors[node]:
            if w == INF:
                continue
            extra = 0.0
            if use_danger:
                n_info = nodes.get(nxt)
                if n_info:
                    extra = _node_danger(n_info["x"], n_info["y"], police) * DANGER_EDGE_SCALE
            nc = cost + w + extra
            if nc < cost_to.get(nxt, INF):
                cost_to[nxt] = nc
                first_step[nxt] = nxt if node == start else first_step[node]
                hops[nxt] = h + 1
                heapq.heappush(pq, (nc, nxt))

    return cost_to, first_step, hops


def _future_value(drop, hops_to_reach):
    """Estimate drop value when runner arrives."""
    if hops_to_reach >= drop["ttl_remaining"]:
        return 0.0
    return drop["current_value"] * math.exp(-drop["leak_lambda"] * hops_to_reach)


def _score_drop(drop, cost_drop, hops_drop, safehouse_hops_est, ticks_left, drop_node, police):
    """Score a drop: higher = better to chase."""
    if cost_drop >= INF or hops_drop is None:
        return -INF

    fv = _future_value(drop, hops_drop)
    if fv < MIN_DROP_VALUE:
        return -INF

    # Round trip: go to drop + collect + go to safehouse + deposit
    total_hops = hops_drop + 1 + safehouse_hops_est + 1
    if total_hops >= ticks_left:
        return -INF

    # Danger at the drop node
    danger = 0.0
    if drop_node:
        danger = _node_danger(drop_node["x"], drop_node["y"], police)

    # Urgency bonus for expiring drops
    ttl = drop["ttl_remaining"]
    urgency = 1.0
    if ttl < 10:
        urgency = 2.0
    elif ttl < 20:
        urgency = 1.4

    # Efficiency: value per unit of cost/time
    score = (fv * urgency) / (cost_drop + danger * 0.08 + 1.0)
    return score


def _runner_in_danger(rx, ry, police):
    for cop in police:
        if _dist(rx, ry, cop["x"], cop["y"]) <= cop["radius"] * SAFETY_MARGIN:
            return True
    return False


def decide_for_runner(runner, tick, nodes, neighbors, drops, police, safehouses):
    """Core decision for one runner."""
    global bot_state

    rid = runner["id"]
    here = runner["current_node"]
    carrying = runner["carrying_loot"]
    capacity = runner["max_capacity"]
    ticks_left = TOTAL_TICKS - tick

    if runner["status"] == "STUNNED":
        return {"runner_id": rid, "action": "HOLD"}

    # --- DEPOSIT if on safehouse with loot ---
    if carrying > 0 and nodes[here]["type"] == "safehouse":
        bot_state.target_drop.pop(rid, None)
        return {"runner_id": rid, "action": "DEPOSIT"}

    # --- COLLECT if drop on current node and room ---
    drop_here = None
    for d in drops:
        if d["node_id"] == here and d["current_value"] > 0:
            if drop_here is None or d["current_value"] > drop_here["current_value"]:
                drop_here = d
    if drop_here is not None and carrying < capacity:
        bot_state.target_drop.pop(rid, None)
        return {"runner_id": rid, "action": "COLLECT"}

    # --- Compute paths ---
    cost_to, first_step, hops = _dijkstra(neighbors, nodes, here, police, use_danger=True)

    # --- Nearest reachable safehouse ---
    best_sh, best_sh_cost, best_sh_hops = None, INF, INF
    for s in safehouses:
        if s in cost_to and s != here and cost_to[s] < best_sh_cost:
            best_sh = s
            best_sh_cost = cost_to[s]
            best_sh_hops = hops.get(s, INF)

    # --- Decide whether to deposit ---
    in_danger = carrying > 0 and _runner_in_danger(runner["x"], runner["y"], police)

    if ticks_left <= TICKS_CRITICAL:
        deposit_thresh = 0.0  # any loot → go deposit
    elif ticks_left <= TICKS_URGENT:
        deposit_thresh = DEPOSIT_FRAC_URGENT
    else:
        deposit_thresh = DEPOSIT_FRAC_NORMAL

    nearly_full = carrying >= deposit_thresh * capacity
    must_deposit = carrying > 0 and best_sh is not None and best_sh_hops + 2 >= ticks_left
    no_drops = len(drops) == 0

    should_deposit = (must_deposit or in_danger or (carrying > 0 and (nearly_full or no_drops)))

    if should_deposit and best_sh is not None:
        step = first_step.get(best_sh)
        if step is not None and step != here:
            return {"runner_id": rid, "action": "TRAVERSE", "target_node": step}

    # --- Score drops ---
    scored = []
    for drop in drops:
        dn = drop["node_id"]
        if dn == here or dn not in cost_to:
            continue
        # Estimate hops from drop to nearest safehouse
        dn_info = nodes.get(dn)
        sh_hops_est = INF
        if dn_info:
            for s in safehouses:
                si = nodes.get(s)
                if si:
                    rough = _dist(dn_info["x"], dn_info["y"], si["x"], si["y"]) / 55.0
                    sh_hops_est = min(sh_hops_est, max(1, int(rough) + 1))

        sc = _score_drop(drop, cost_to[dn], hops.get(dn), sh_hops_est,
                         ticks_left, dn_info, police)
        if sc > -INF:
            scored.append((sc, drop["drop_id"], dn))

    scored.sort(key=lambda x: -x[0])

    if scored:
        best_sc, best_did, best_dn = scored[0]

        # Hysteresis: stick with previous target unless new one is much better
        prev = bot_state.target_drop.get(rid)
        if prev is not None:
            for sc, did, dn in scored:
                if did == prev:
                    if best_sc <= sc * HYSTERESIS:
                        best_did, best_dn = did, dn
                    break

        bot_state.target_drop[rid] = best_did
        step = first_step.get(best_dn)
        if step is not None and step != here:
            return {"runner_id": rid, "action": "TRAVERSE", "target_node": step}

    # --- Fallback: deposit if carrying ---
    if carrying > 0 and best_sh is not None:
        step = first_step.get(best_sh)
        if step is not None and step != here:
            return {"runner_id": rid, "action": "TRAVERSE", "target_node": step}

    # --- Explore: go to a money spot ---
    spots = [(cost_to[nid], nid) for nid in nodes
             if nodes[nid]["type"] == "money_spot" and nid in cost_to and nid != here]
    if spots:
        spots.sort()
        step = first_step.get(spots[0][1])
        if step is not None and step != here:
            return {"runner_id": rid, "action": "TRAVERSE", "target_node": step}

    return {"runner_id": rid, "action": "HOLD"}


def strategy(tick, nodes, neighbors, runners, drops, police, safehouses, money_spots):
    """Return a list with one action per runner."""
    return [decide_for_runner(runner, tick, nodes, neighbors, drops, police, safehouses)
            for runner in runners]


# ===========================================================================
#  >>> YOUR CODE ENDS HERE <<<
#  Everything below is handled for you. You normally do not edit it.
# ===========================================================================
VALID_ACTIONS = {"TRAVERSE", "HOLD", "COLLECT", "DEPOSIT"}


def _hold(runner_id):
    return {"runner_id": runner_id, "action": "HOLD"}


def check_actions(raw, runners, neighbors):
    """Validate what strategy() returned. Bad or missing actions become HOLD."""
    if not isinstance(raw, list):
        log("WARNING: strategy must return a list, got %s. All runners hold." % type(raw).__name__)
        raw = []

    runner_by_id = {r["id"]: r for r in runners}
    chosen = {}
    for item in raw:
        if not isinstance(item, dict) or item.get("runner_id") not in runner_by_id:
            log("WARNING: ignored action with unknown runner: %r" % (item,))
            continue
        rid = item["runner_id"]
        if rid in chosen:
            log("WARNING: runner %s got more than one action, using the first." % rid)
            continue
        kind = item.get("action")
        if kind not in VALID_ACTIONS:
            log("WARNING: runner %s has invalid action %r." % (rid, kind))
            continue
        if kind == "TRAVERSE":
            here = runner_by_id[rid]["current_node"]
            target = item.get("target_node")
            if not any(n == target and w != INF for n, w in neighbors[here]):
                log("WARNING: runner %s cannot traverse %s -> %r (not a free neighbour)."
                    % (rid, here, target))
                continue
            chosen[rid] = {"runner_id": rid, "action": "TRAVERSE", "target_node": target}
        else:
            chosen[rid] = {"runner_id": rid, "action": kind}

    final = []
    for r in runners:
        if r["status"] == "STUNNED":
            final.append(_hold(r["id"]))
        elif r["id"] in chosen:
            final.append(chosen[r["id"]])
        else:
            log("WARNING: runner %s had no action, holding." % r["id"])
            final.append(_hold(r["id"]))
    return final


def run_tick(state):
    """Turn one tick of raw JSON into the list of actions to send back."""
    runners = state.get("my_runners", [])
    try:
        nodes = {n["id"]: {"x": n["x"], "y": n["y"], "type": n["type"]}
                 for n in state["graph_nodes"]}
        neighbors = {node_id: [] for node_id in nodes}
        for e in state["graph_edges"]:
            weight = e["current_weight"]
            if weight < 0:
                weight = INF
            neighbors[e["from"]].append((e["to"], weight))
            neighbors[e["to"]].append((e["from"], weight))

        safehouses = [i for i, n in nodes.items() if n["type"] == "safehouse"]
        money_spots = [i for i, n in nodes.items() if n["type"] == "money_spot"]

        raw = strategy(state["tick"], nodes, neighbors, runners,
                       state["active_money_drops"], state["police_officers"],
                       safehouses, money_spots)
        return check_actions(raw, runners, neighbors)
    except Exception:
        log(traceback.format_exc())
        return [_hold(r["id"]) for r in runners]


SAMPLE_STATE = {
    "tick": 0,
    "graph_nodes": [
        {"id": 0, "x": 0, "y": 0, "type": "safehouse"},
        {"id": 1, "x": 1, "y": 0, "type": "junction"},
        {"id": 2, "x": 2, "y": 0, "type": "money_spot"},
    ],
    "graph_edges": [
        {"from": 0, "to": 1, "current_weight": 1.0},
        {"from": 1, "to": 2, "current_weight": 1.0},
    ],
    "my_runners": [
        {"id": 0, "current_node": 0, "x": 0, "y": 0, "carrying_loot": 0,
         "max_capacity": 100, "status": "ACTIVE", "stun_ticks_remaining": 0},
    ],
    "active_money_drops": [
        {"drop_id": 1, "node_id": 2, "target_safehouse_node": 0,
         "current_value": 50, "leak_lambda": 0.01, "ttl_remaining": 30},
    ],
    "police_officers": [
        {"id": 0, "x": 5, "y": 5, "vx": 0, "vy": 0, "radius": 1},
    ],
}


def self_test():
    """Run strategy() on a tiny sample and show the result."""
    print("Sample input : 1 runner on safehouse 0, a 50-value drop on node 2")
    actions = run_tick(SAMPLE_STATE)
    print("Your output  :", json.dumps(actions))
    print("Format is valid. (Warnings, if any, are shown above.)")


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        actions = run_tick(json.loads(line))
        print(json.dumps({"runner_actions": actions}), flush=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        main()
