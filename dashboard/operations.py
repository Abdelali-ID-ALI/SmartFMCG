"""Streamlit controls, operational metrics, maps and simulator replay."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import engine, maps
from src.optimization.fmcg_validator import UNITS_PER_MAD
from src.simulation.state import Congestion, ServiceDelay

ERRORS = (ValueError, OSError, RuntimeError, KeyError, EOFError)
TEMPLATE_NAMES = {"V01": "V01 · Van", "V02": "V02 · Van", "V03": "V03 · Medium",
                  "V04": "V04 · Medium", "V05": "V05 · Heavy", "V06": "V06 · Refrigerated"}


def clock(seconds):
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02}"


def policy_label(key):
    return f"{key} · {engine.POLICIES[key]}"


def get_inputs() -> engine.Inputs:
    return st.session_state.get("scenario_inputs", engine.Inputs())


def loss_savings(fixed: engine.Run, adaptive: engine.Run) -> float:
    return fixed.metrics["objective_mad"] - adaptive.metrics["objective_mad"]


def execute(label, operation):
    if not engine.WORK.acquire(blocking=False):
        st.warning("A calculation is already running on this server. Retry when it finishes.")
        return None
    try:
        with st.spinner(label, show_time=True):
            return operation()
    except FileNotFoundError as exc:
        name = Path(exc.filename).name if exc.filename else Path(str(exc)).name
        st.error(f"Required artifact is missing ({name}). Restore it from the project checkpoint and retry.")
    except ERRORS as exc:
        # Artifact readers may include a host filename in their error messages.
        message = str(exc).replace(str(Path.cwd()) + "/", "")
        st.error(f"Calculation could not finish. {message}")
    finally:
        engine.WORK.release()
    return None


def prepared():
    if "prepared" not in st.session_state:
        result = engine.prepare(get_inputs())
        st.session_state.prepared = result
    return st.session_state.prepared


def planned(strategy):
    plans = st.session_state.setdefault("plans", {})
    if strategy not in plans:
        plans[strategy] = engine.plan(prepared(), strategy)
    return plans[strategy]


def prepare_comparison(policy, seed):
    case, initial = prepared(), planned("Robust ML")
    runs = st.session_state.setdefault("runs", {})
    for key in dict.fromkeys((("B4", None), (policy, seed if policy not in engine.RULES else None))):
        if key not in runs:
            runs[key] = engine.replay(case, initial, key[0], key[1] or 2917)
    active = (policy, seed if policy not in engine.RULES else None)
    st.session_state.active_run = active
    return runs[active]


def scenario_page():
    st.title("Scenario configuration")
    st.caption("Casablanca · Synthetic orders and disruptions · Cached directed OSRM road tables")
    current = get_inputs()
    st.selectbox("Disruption preset", ("controlled", "stress"), index=("controlled", "stress").index(current.regime),
                 format_func=str.title, key="preset",
                 on_change=apply_preset)
    with st.form("scenario"):
        a, b, c = st.columns(3)
        customers = a.selectbox("Customers", engine.SIZES, index=engine.SIZES.index(current.customers),
                                help="12 uses a subset of the 30-customer cache. 30, 50 and 100 use qualified tables.")
        seed = b.number_input("Scenario seed", 0, 2**32 - 1, current.seed, step=1)
        fleet = c.number_input("Maximum available vehicles", 1, 16, current.max_vehicles, step=1,
                               help="Constructive sizing selects the fleet from the templates below.")
        templates = st.multiselect("Fleet templates", engine.TEMPLATES, default=current.templates,
                                   format_func=TEMPLATE_NAMES.get)
        a, b, c = st.columns(3)
        demand = a.slider("Regular delivery demand multiplier", .75, 1.20, current.demand, .05)
        reserve = b.slider("Reserve per stocked vehicle (crates)", 0, 12, current.reserve, key="reserve_input")
        traffic = c.slider("Congestion travel-time multiplier", 1., 2.25, current.traffic, .05, key="traffic_input")
        a, b, c, d = st.columns(4)
        service = a.slider("Extra service time (min)", 0, 20, current.service_min, key="service_input")
        urgents = b.slider("Urgent orders", 0, 10, current.urgents, key="urgent_input")
        window = c.slider("Urgent time window (min)", 30, 90, current.urgent_window_min, 5, key="window_input")
        crates = d.slider("Urgent order size (crates)", 1, 16, current.urgent_crates, key="crates_input")
        policy = st.selectbox("Dispatch policy", engine.POLICIES, format_func=policy_label, key="scenario_policy",
                              index=tuple(engine.POLICIES).index(st.session_state.get("preferred_policy", "B5")))
        submit = st.form_submit_button("Build scenario", type="primary")
    if submit:
        def build():
            inputs = engine.Inputs(customers, seed, fleet, tuple(templates), demand, reserve, st.session_state.preset,
                                   traffic, service, urgents, window, tuple(crates))
            case = st.session_state.get("prepared") if inputs == current else None
            if case is None:
                case = engine.prepare(inputs)
            if inputs != current:
                for key in ("plans", "runs", "active_run", "replay_slider"):
                    st.session_state.pop(key, None)
            st.session_state.update(scenario_inputs=inputs, prepared=case, preferred_policy=policy, sim_policy=policy)
            return case
        if execute("Preparing cached geography, forecasts and synthetic demand…", build):
            st.success("Scenario ready. Open Route planning or Dispatch replay.")
    st.caption("End-to-end OSRM tests cover up to 100 customers and 16 available vehicles. "
               "The 250- and 500-customer Haversine tests are partial workloads.")
    if case := st.session_state.get("prepared"):
        st.subheader("Scenario data")
        st.caption(f"Synthetic date {case.day} · Seed {case.inputs.seed} · Case {case.case_id[:12]} · Preparation {case.wall_s:.2f} s wall time")
        order_rows = [{"Customer": stop.id, "Delivery (crates)": stop.delivery[2],
                       "Pickup (crates)": stop.pickup[2], "Window opens": clock(stop.window[0]),
                       "Window closes": clock(stop.window[1]), "Cold chain": stop.refrigerated,
                       "Heavy vehicle restricted": stop.no_heavy} for stop in case.problem.stops]
        fleet_rows = [{"Vehicle": vehicle.id, "Refrigerated": vehicle.refrigerated, "Heavy": vehicle.heavy,
                       "Capacity (kg)": vehicle.capacity[0] / 1000, "Capacity (L)": vehicle.capacity[1],
                       "Capacity (crates)": vehicle.capacity[2], "Capacity (pallets)": vehicle.capacity[3],
                       "Allocated reserve (crates)": sum(load[2] for load in stock)}
                      for vehicle, stock in zip(case.problem.vehicles, case.reserves, strict=True)]
        orders, vehicles = st.tabs(("Orders", "Fleet"))
        orders.dataframe(pd.DataFrame(order_rows), hide_index=True)
        vehicles.dataframe(pd.DataFrame(fleet_rows), hide_index=True)


def apply_preset():
    stress = st.session_state.preset == "stress"
    values = (4, 1.85, 17, 2, 40, (3, 9)) if stress else (8, 1.30, 7, 1, 55, (2, 6))
    for key, value in zip(("reserve_input", "traffic_input", "service_input", "urgent_input", "window_input",
                           "crates_input"), values, strict=True):
        st.session_state[key] = value


def overview():
    st.title("Distribution overview")
    st.markdown("Plan Casablanca deliveries, replay disruptions and compare dispatch decisions.")
    active = st.session_state.get("active_run")
    if active is not None:
        case = prepared()
        fixed, run = st.session_state.runs[("B4", None)], st.session_state.runs[active]
        st.subheader("Latest simulated day")
        st.caption(f"{policy_label(run.policy)} · {case.inputs.customers} customers · Seed {case.inputs.seed}"
                   + (f" · Training seed {run.training_seed}" if run.training_seed else ""))
        show_metrics(case, run, fixed)
        show_pair(fixed, run)
    else:
        st.info("Run a paired dispatch simulation to see realized KPIs for your scenario.", icon="ℹ️")
    st.subheader("Archived benchmark")
    st.caption("Phase 10 · 35 common days across 7 date blocks · Synthetic economics (MAD)")
    file = Path("results/experiments/final_benchmark/summary.csv")
    if not file.is_file():
        st.warning("Archived benchmark summary is unavailable.")
        return
    data = pd.read_csv(file)
    chosen = st.selectbox("Archived policy", ("B5", "B8", "B4", "B9", "B10"), format_func=policy_label)
    values = data[data.policy.eq(chosen)].set_index("metric")["mean"]
    for card, (label, metric, style) in zip(st.columns(4), (
        ("Operational loss", "objective_mad", ",.2f"), ("OTIF", "otif", ".1%"),
        ("Urgent orders served", "urgent_service_rate", ".1%"), ("Replanning attempts", "replan_attempts", ".2f"),
    ), strict=True):
        card.metric(label, f"{values[metric]:{style}}" + (" MAD" if metric == "objective_mad" else ""),
                    border=True)
    st.caption("Use Scientific benchmark for dispersion, paired uncertainty and scalability protocols.")


def show_metrics(case, run, fixed):
    metrics = run.metrics
    urgent = (f"{metrics['urgent_served']:.0f} / {metrics['urgent_total']:.0f}"
              if metrics["urgent_total"] else "—")
    cards = (
        ("Operational loss", f"{metrics['objective_mad']:,.2f} MAD", "Dispatch ledger including penalties and attempts."),
        ("Operating cost", f"{metrics['operating_cost_mad']:,.2f} MAD", "Activation, distance and elapsed vehicle time."),
        ("OTIF", f"{metrics['otif']:.1%}", "Orders completed within their service window, including released urgents."),
        ("Urgent orders served", urgent, "Completed urgent orders, including any completed late."),
        ("Customers", str(case.inputs.customers), None),
        ("Vehicles used / available", f"{metrics['vehicles']} / {len(case.problem.vehicles)}", None),
        ("Initial load utilization", f"{metrics['initial_weight_utilization']:.1%}", "Mean departure weight share."),
        ("Total distance", f"{metrics['distance_km']:.1f} km", None),
        ("Total delay", f"{metrics['delay_min']:.1f} min", None),
        ("Replanning attempts", str(metrics["replan_attempts"]), None),
        ("Replans applied", str(metrics["replans_applied"]), None),
        ("Loss savings vs B4", f"{loss_savings(fixed, run):+,.2f} MAD", "Fixed loss minus policy loss."),
    )
    for start in range(0, len(cards), 4):
        for card, (label, value, help_text) in zip(st.columns(4), cards[start:start + 4], strict=True):
            card.metric(label, value, help=help_text, border=True)


def show_pair(fixed, run):
    st.dataframe(pd.DataFrame([
        {"Role": label, "Policy": item.policy, "Operational loss (MAD)": item.metrics["objective_mad"],
         "Operating cost (MAD)": item.metrics["operating_cost_mad"], "OTIF (%)": item.metrics["otif"] * 100,
         "Urgents served": item.metrics["urgent_served"], "Route changes": item.metrics["route_changes"],
         "Calculation (wall s)": item.wall_s, "Inference (wall s)": item.inference_s}
        for label, item in (("Fixed reference", fixed), ("Selected policy", run))]), hide_index=True)
    st.caption("Both policies start from the same robust ML plan, fleet, reserve, synthetic traffic and disruptions. "
               "One simulated day illustrates the trade-off.")
    with st.expander("Operational loss breakdown"):
        names = {"fixed_mad": "Activation", "distance_mad": "Distance", "time_mad": "Vehicle time",
                 "lateness_mad": "Late service", "overtime_mad": "Overtime", "instability_mad": "Route changes",
                 "attempt_mad": "Replanning attempts", "omission_mad": "Omissions"}
        breakdown = pd.DataFrame([
            {"Run": label, "Component": names[key], "Amount (MAD)": value}
            for label, item in ((f"{fixed.policy} · Reference", fixed), (f"{run.policy} · Selected", run))
            for key, value in item.costs.items()])
        st.plotly_chart(px.bar(breakdown, x="Amount (MAD)", y="Run", color="Component",
                              orientation="h", height=280))


def route_page():
    st.title("Route planning")
    st.caption("Plan against the same fleet, demand, reserve and cached directed road network.")
    strategies = st.multiselect("Planning strategies", engine.PLANNERS, default=["Robust ML"])
    if st.button("Generate plans", type="primary", disabled=not strategies):
        def calculate():
            completed, errors = {}, []
            for strategy in strategies:
                try:
                    completed[strategy] = planned(strategy)
                except ValueError as exc:
                    errors.append(f"{strategy} · {exc}")
            return completed, errors
        outcome = execute("Solving and independently validating each selected plan…", calculate)
        if outcome:
            for error in outcome[1]:
                st.warning(error)
    choices = {name: item for name, item in st.session_state.get("plans", {}).items() if name in strategies}
    if not choices:
        st.info("Generate a plan to inspect vehicle assignments, visit windows and cargo.")
        return
    case = prepared()
    st.caption(f"Case {case.case_id[:12]} · Forecast hour {case.forecast_hour:02}:00 · "
               f"Robust ML uplift {case.margin:.2%} · Cached OSRM static travel times")
    st.dataframe(pd.DataFrame([{"Strategy": name, "Status": item.status,
                               "Plan cost (MAD)": item.report.cost / UNITS_PER_MAD,
                               "Distance (km)": sum(path.distance_m for path in item.report.tours) / 1000,
                               "Vehicles": len(item.tours), "Omitted": len(item.report.unserved),
                               "Solver budget (s)": item.budget_s, "Wall time (s)": item.wall_s}
                              for name, item in choices.items()]), hide_index=True)
    name = st.selectbox("Map strategy", tuple(choices))
    item = choices[name]
    st.caption(item.detail)
    route_map(case, item)


def route_map(case, item):
    summaries, visits, routes, markers = [], [], [], []
    for tour, report in zip(item.tours, item.report.tours, strict=True):
        vehicle = item.problem.vehicles[tour.vehicle]
        previous, end = 0, tour.departure_s
        stock = item.reserves[tour.vehicle]
        loads = [engine.cargo_with_reserve(load.onboard, stock) for load in report.loads]
        capacity_usage = [100 * max(load[i] for load in loads) / cap if cap else 0.
                          for i, cap in enumerate(vehicle.capacity)]
        summaries.append({"Vehicle": vehicle.id, "Depart": clock(tour.departure_s), "Return": clock(tour.return_s),
                          "Distance (km)": report.distance_m / 1000, "Duration (min)": report.duration_s / 60,
                          "Peak weight (%)": capacity_usage[0], "Peak volume (%)": capacity_usage[1],
                          "Peak crates (%)": capacity_usage[2], "Peak pallets (%)": capacity_usage[3]})
        routes.append((tour.vehicle, vehicle.id, (0, *tour.stops, 0)))
        for sequence, (node, start, cargo) in enumerate(zip(tour.stops, tour.service_starts_s, loads[1:],
                                                             strict=True), 1):
            stop = item.problem.stops[node - 1]
            arrival = end + item.problem.travel_s[previous][node]
            visits.append({"Vehicle": vehicle.id, "Stop": sequence, "Customer": stop.id,
                           "Arrival": clock(arrival), "Service starts": clock(start),
                           "Window opens": clock(stop.window[0]), "Window closes": clock(stop.window[1]),
                           "Delivery (crates)": stop.delivery[2], "Pickup (crates)": stop.pickup[2],
                           "Load after (kg)": cargo[0] / 1000, "Load after (L)": cargo[1],
                           "Load after (crates)": cargo[2], "Load after (pallets)": cargo[3]})
            markers.append((node, f"{vehicle.id} · {sequence}. {stop.id} · ETA {clock(arrival)} · "
                                  f"Deliver {stop.delivery[2]} crates · Pick up {stop.pickup[2]}",
                            maps.COLORS[tour.vehicle]))
            previous, end = node, start + stop.service_s
    map_panel(case, routes, markers, "planning")
    st.caption("Solid segments use matching archived OSRM geometry. Dashed links connect stop coordinates. "
               "Distance and duration use the directed road tables.")
    routes_tab, visits_tab = st.tabs(("Vehicle routes", "Customer visits"))
    routes_tab.dataframe(pd.DataFrame(summaries), hide_index=True)
    visits_tab.dataframe(pd.DataFrame(visits), hide_index=True)


def map_panel(case, routes, markers, key):
    offline = st.toggle("Map without external tiles or scripts", key=f"offline_{key}")
    figure = maps.display(case.locations, routes, markers, offline=offline, key=key)
    if figure:
        st.plotly_chart(figure, key=f"{key}_offline_chart")
    st.caption("Road data © OpenStreetMap contributors · [ODbL 1.0](https://www.openstreetmap.org/copyright)"
               " · OSRM archive from FOSSGIS · [Fix the map](https://www.openstreetmap.org/fixthemap)")


def simulation_page():
    st.title("Dispatch replay")
    st.caption("A paired discrete-event simulation with the fixed B4 plan as reference.")
    a, b = st.columns(2)
    preferred = st.session_state.get("preferred_policy", "B5")
    policy = a.selectbox("Dispatch policy", engine.POLICIES, index=tuple(engine.POLICIES).index(preferred),
                         format_func=policy_label, key="sim_policy")
    seed = b.selectbox("Frozen training seed", (2917, 2927, 2953), disabled=policy in engine.RULES,
                       help="Default seed is the first archived seed. All three were selected before final test outcomes.")
    if policy in {"B10", "B12"}:
        st.warning("Experimental policy. The continuous DQN's training ranges often fail to cover states "
                   "observed at 100 customers. See Learning policies for the transfer analysis.")
    elif policy == "B9":
        st.caption("Experimental policy. Q-Learning showed promise at 100 customers. "
                   "Results vary with scenario, reserve and demand.")
    if st.button("Start paired simulation", type="primary"):
        if execute("Running B4 and the selected policy on the same synthetic day…",
                   lambda: prepare_comparison(policy, seed)):
            st.session_state.replay_slider = 0
    active = st.session_state.get("active_run")
    if active is None:
        st.info("Start a paired simulation to load the decision timeline.")
        return
    case, fixed, run = prepared(), st.session_state.runs[("B4", None)], st.session_state.runs[active]
    st.subheader(f"Replay · {policy_label(run.policy)}")
    st.caption(f"Case {case.case_id[:12]} · Seed {case.inputs.seed} · "
               f"Wall time {run.wall_s:.2f} s · Displayed clock is simulated operational time")
    last = len(run.frames) - 1
    current = st.session_state.get("replay_slider", 0)
    a, b, c, d = st.columns(4)
    if a.button("Previous", disabled=current == 0):
        current -= 1
    if b.button("Advance", disabled=current == last):
        current += 1
    if c.button("Replay from start"):
        current = 0
    if d.button("Reset replay"):
        st.session_state.pop("active_run", None)
        st.session_state.pop("replay_slider", None)
        st.rerun()
    st.session_state.replay_slider = current
    index = st.slider("Decision timeline", 0, last, key="replay_slider") if last else 0
    replay_frame(case, st.session_state.plans["Robust ML"], run.frames[index], terminal=index == last)
    if index == last:
        st.subheader("Realized day comparison")
        show_metrics(case, run, fixed)
        show_pair(fixed, run)


def replay_frame(case, initial, frame, terminal):
    cards = st.columns(4)
    cards[0].metric("Operational time", clock(frame.time_s), border=True)
    cards[1].metric("Completed orders", len(frame.completed), border=True)
    remaining = len(case.problem.stops) + sum(
        not isinstance(event, (ServiceDelay, Congestion)) for event in frame.events) - len(frame.completed)
    cards[2].metric("Known unfinished orders", remaining, border=True)
    cards[3].metric("Accrued loss", f"{frame.loss_mad:,.2f} MAD" if frame.loss_mad is not None else "—", border=True,
                    help="Omission penalties settle at the end of the day.")
    ids = {stop.id: i for i, stop in enumerate(case.problem.stops, 1)}
    events = []
    for event in frame.events:
        if isinstance(event, Congestion):
            detail = f"{event.factor:.2f}× travel time until {clock(event.end_s)}"
            kind = "Congestion"
        elif isinstance(event, ServiceDelay):
            detail, kind = f"{event.order_id} · +{event.extra_s / 60:.1f} min", "Service delay"
        else:
            order = event.order
            ids[order.stop.id] = order.node
            kind = "Urgent order"
            status = ("Completed" if order.stop.id in frame.completed else "Waiting for allocation"
                      if order.stop.id in frame.waiting_urgent else "Unserved" if terminal else "Allocated or committed")
            detail = (f"{order.stop.id} · {order.stop.delivery[2]} {order.product} crates · "
                      f"Due {clock(order.stop.window[1])} · {status}")
        events.append({"Time": clock(event.at_s), "Event": kind, "Detail": detail})
    fleet, routes, markers = [], [], []
    for view in frame.vehicles:
        vehicle = case.problem.vehicles[view.index]
        target = case.locations[view.destination].id if view.destination is not None else "—"
        node = case.locations[view.node].id
        if view.phase != "unused":
            markers.append((view.node, f"{vehicle.id} · {view.phase} · last known node {node} · destination {target}",
                            maps.COLORS[view.index]))
            anchor = view.destination if view.destination is not None else view.node
            if view.destination is not None:
                routes.append((view.index, f"{vehicle.id} · In transit", (view.node, anchor)))
            if view.tail:
                routes.append((view.index, f"{vehicle.id} · Current suffix",
                               (anchor, *(ids[key] for key in view.tail), 0)))
        fleet.append({"Vehicle": vehicle.id, "Phase": view.phase, "Last known node": node,
                      "Destination": target, "Committed order": view.committed or "—",
                      "Remaining delivery (crates)": view.regular[2],
                      "Ambient reserve (crates)": view.reserves[0][2],
                      "Chilled reserve (crates)": view.reserves[1][2], "Collected crates": view.collected[2],
                      "Onboard (kg)": view.onboard[0] / 1000,
                      "Mutable suffix": " → ".join(view.tail)})
    map_tab, fleet_tab, event_tab, action_tab = st.tabs(("Vehicle positions", "Fleet & inventory", "Known events",
                                                        "Decision log"))
    with map_tab:
        map_panel(case, routes, markers, "replay")
        st.caption("Moving vehicles show their last recorded node and destination. Lines indicate current "
                   "commitments and suffixes from the engine. Position is resolved at a simulation event.")
    fleet_tab.dataframe(pd.DataFrame(fleet), hide_index=True)
    with fleet_tab.expander("Original robust plan"):
        st.dataframe(pd.DataFrame([
            {"Vehicle": case.problem.vehicles[tour.vehicle].id,
             "Original order sequence": " → ".join(case.locations[node].id for node in tour.stops)}
            for tour in initial.tours]), hide_index=True)
    if events:
        event_tab.dataframe(pd.DataFrame(events), hide_index=True)
    else:
        event_tab.caption("No disruptions announced at this decision.")
    if frame.decisions:
        action_tab.dataframe(pd.DataFrame([
            {**item, "time_s": clock(item["time_s"]),
             "applied_s": clock(item["applied_s"]) if item["applied_s"] is not None else "—"}
            for item in reversed(frame.decisions)]).rename(columns={"time_s": "Decision time",
                "applied_s": "Applied time", "changes": "Changed predecessors"}), hide_index=True)
    else:
        action_tab.caption("The dispatch policy is awaiting its first decision.")
