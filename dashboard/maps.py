"""Geographic display using cached legs and explicit schematic fallbacks."""

from functools import lru_cache
from html import escape
from pathlib import Path
import json

import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium

COLORS = ("#00786F", "#375DB6", "#A43B70", "#AD6000", "#6949A1", "#217EAE", "#537620", "#B7462E",
          "#5D6D1A", "#007A9B", "#934D1B", "#5D50C3", "#B93358", "#366A57", "#735A35", "#00627F")


@lru_cache(maxsize=1)
def archived_legs():
    path = Path("data/matrices/osrm-D01-C30.geojson")
    if not path.is_file():
        return {}
    feature = json.loads(path.read_text())
    return {tuple(feature["properties"]["ids"]):
            tuple((lat, lon) for lon, lat in feature["geometry"]["coordinates"])}


def pieces(locations, nodes):
    for left, right in zip(nodes, nodes[1:]):
        a, b = locations[left], locations[right]
        geometry = archived_legs().get((a.id, b.id))
        yield geometry or ((a.latitude, a.longitude), (b.latitude, b.longitude)), bool(geometry)


def display(locations, routes, markers, *, offline, key):
    depot = locations[0]
    if offline:
        fig = go.Figure()
        for vehicle, label, nodes in routes:
            for leg, (points, road) in enumerate(pieces(locations, nodes)):
                lat, lon = zip(*points, strict=True)
                fig.add_scatter(x=lon, y=lat, mode="lines", name=label, legendgroup=label,
                                showlegend=leg == 0, hovertemplate=f"{label}<br>"
                                + ("Archived OSRM geometry" if road else "Schematic link") + "<extra></extra>",
                                line=dict(color=COLORS[vehicle], width=2.5, dash="solid" if road else "dot"))
        for label, size, group in (("Depot", 14, [(0, depot.id, "#152B40")]), ("Stops", 8, markers)):
            fig.add_scatter(x=[locations[n].longitude for n, _, _ in group],
                            y=[locations[n].latitude for n, _, _ in group], text=[text for _, text, _ in group],
                            marker=dict(size=size, color=[color for _, _, color in group]),
                            mode="markers", name=label, hovertemplate="%{text}<extra></extra>")
        fig.update_layout(height=490, dragmode="pan", xaxis_title="Longitude (°)", yaxis_title="Latitude (°)",
                          margin=dict(l=10, r=10, t=20, b=10))
        fig.update_yaxes(scaleanchor="x", scaleratio=1.20)
        return fig

    view = folium.Map(location=[depot.latitude, depot.longitude], zoom_start=12, tiles="OpenStreetMap",
                      control_scale=True, prefer_canvas=True)
    folium.Marker((depot.latitude, depot.longitude), tooltip=f"Depot · {depot.id}",
                  icon=folium.Icon(color="darkblue", icon="home")).add_to(view)
    for vehicle, label, nodes in routes:
        layer = folium.FeatureGroup(name=escape(label))
        for points, road in pieces(locations, nodes):
            folium.PolyLine(points, color=COLORS[vehicle], weight=4, opacity=.9,
                            dash_array=None if road else "7 7",
                            tooltip=f"{escape(label)} · {'Archived OSRM geometry' if road else 'Schematic link'}"
                            ).add_to(layer)
        layer.add_to(view)
    for node, label, color in markers:
        point = locations[node]
        folium.CircleMarker((point.latitude, point.longitude), radius=5, color=color, fill=True,
                            fill_opacity=.85, tooltip=escape(label)).add_to(view)
    view.fit_bounds([(point.latitude, point.longitude) for point in locations], padding=(25, 25))
    folium.LayerControl(collapsed=True).add_to(view)
    st_folium(view, height=490, use_container_width=True, returned_objects=[], key=key)
    return None
