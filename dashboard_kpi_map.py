import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
from psycopg2 import OperationalError
import datetime
import numpy as np
import math

# Folium map deps (pip install streamlit-folium folium)
try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="5G FWA KPI Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config: where to find the GCELL geo file on your machine ───────────────────
GEO_CSV_PATH = "GCELL_W23.csv"   # <── put your GCELL csv next to this script, or upload via sidebar

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2330; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #252d42; border-radius: 12px; padding: 16px;
}
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace; font-size: 10px !important;
    letter-spacing: 1.5px; text-transform: uppercase; color: #64748b !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace; font-size: 22px !important; color: #38bdf8 !important;
}
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #f8fafc; font-size: 1.4rem !important; }
h3 { color: #38bdf8; font-size: 0.9rem !important; }
hr { border-color: #1e2330; }
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0d0f14; font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 12px; letter-spacing: 1px;
    border: none; border-radius: 8px;
}
.status-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 1px;
}
.status-connected { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-demo      { background: #1c1917; color: #f59e0b; border: 1px solid #78350f; }
.drill-banner {
    background: linear-gradient(135deg, #422006 0%, #3b1d05 100%);
    border: 1px solid #b45309; border-radius: 10px; padding: 10px 16px;
    font-family: 'Space Mono', monospace; font-size: 13px; color: #fbbf24;
    letter-spacing: 0.5px;
}
.hint {
    font-family: 'Space Mono', monospace; font-size: 11px;
    color: #475569; letter-spacing: 0.5px;
}
.legend-pill {
    display:inline-block; padding:2px 9px; margin:2px 4px; border-radius:999px;
    font-family:'Space Mono',monospace; font-size:10px; letter-spacing:.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
CT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8", font_family="DM Sans", margin=dict(l=0, r=0, t=36, b=0),
)
AX     = dict(gridcolor="#1e2330", showline=False, tickfont_color="#64748b")
AX_REV = dict(gridcolor="#1e2330", showline=False, tickfont_color="#64748b", autorange="reversed")
PAL    = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#fb7185", "#a78bfa", "#22d3ee"]
HILITE = "#fbbf24"   # gold for the drilled-into cell

# Vendor brand colours for the map
VENDOR_COLORS = {
    "Huawei": "#FEE715", "Nokia": "#1e90ff", "ZTE": "#34d399",
    "Baicells": "#a78bfa", "Fiberhome": "#f59e0b", "Orex": "#22d3ee",
}
VENDOR_DEFAULT = "#94a3b8"

def vendor_color(v):
    if not isinstance(v, str):
        return VENDOR_DEFAULT
    key = v.strip().lower()
    for name, col in VENDOR_COLORS.items():
        if name.lower() in key:
            return col
    return VENDOR_DEFAULT

def sector_points(lat, lon, az, beam, radius_m, steps=18):
    """Polygon vertices for a cell sector wedge: apex at site, swept ±beam/2 around azimuth."""
    if any(pd.isna(x) for x in (lat, lon, az)) or radius_m <= 0:
        return None
    beam = 65.0 if (pd.isna(beam) or beam <= 0) else float(beam)
    start, end = az - beam / 2.0, az + beam / 2.0
    coslat = math.cos(math.radians(lat)) or 1e-6
    pts = [(lat, lon)]
    n = max(2, int(steps))
    for i in range(n + 1):
        b  = math.radians(start + (end - start) * i / n)   # bearing from North, clockwise
        dn = radius_m * math.cos(b)                          # metres north
        de = radius_m * math.sin(b)                          # metres east
        pts.append((lat + dn / 111320.0, lon + de / (111320.0 * coslat)))
    pts.append((lat, lon))
    return pts

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_conn(host, port, dbname, user, password):
    return psycopg2.connect(host=host, port=int(port),
                            dbname=dbname, user=user, password=password)

def run_query(conn, sql):
    return pd.read_sql_query(sql, conn)

# ── GCELL geo loader ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_geo(source):
    """source = uploaded file buffer OR a filesystem path string."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    ren = {
        "Cell Name": "cell_name", "Site Name Surge": "site",
        "Site ID Surge": "site_id", "Vendor gNB": "vendor",
        "Latitude": "lat", "Longitude": "lon", "Province": "province",
        "Kab": "kab", "Kec": "kec",
        "RRC Setup Success Rate": "rrc_sr", "SDR": "sdr",
        "QoS Flow Setup Success Rate": "qos_sr",
        "NR User Downlink Average Throughput": "dl_thp", "DL PRB USAGE": "dl_prb",
        "Azimuth": "azimuth", "RADIUS": "radius_km", "BEAM": "beam",
        "M-Tilt": "mtilt", "E-Tilt": "etilt",
    }
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ["lat","lon","rrc_sr","sdr","qos_sr","dl_thp","dl_prb",
              "azimuth","radius_km","beam","mtilt","etilt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "cell_name"])
    # normalise vendor label spelling
    if "vendor" in df.columns:
        df["vendor"] = df["vendor"].astype(str).str.replace("FiberHome", "Fiberhome", regex=False)
    keep = [c for c in ["cell_name","site","site_id","vendor","lat","lon",
                        "province","kab","kec","rrc_sr","sdr","qos_sr","dl_thp","dl_prb",
                        "azimuth","radius_km","beam","mtilt","etilt"]
            if c in df.columns]
    return df[keep].drop_duplicates("cell_name").reset_index(drop=True)

# ── SQL Templates ──────────────────────────────────────────────────────────────
KPI_COLS_SQL = """
    SUM(rrc_succconnestab_time) / NULLIF(SUM(rrc_attconnestab_time), 0) * 100           AS rrc_setup_sr,
    SUM(flow_nbrsuccestab_number) / NULLIF(SUM(flow_nbrattestab_number), 0) * 100        AS qos_flow_sr,
    100 * SUM(context_attrelgnb_number - context_attrelgnb_normal_number)
        / NULLIF(SUM(context_succinitalsetup_time), 0)                                   AS sdr,
    SUM(rlc_upoctdl_kbyte - rlc_uplastttioctdl_kbyte) * 8
        / NULLIF(SUM(rlc_thrptimedl_ms), 0)                                              AS dl_thp,
    SUM(rlc_upoctul_kbyte - rlc_uplastttioctul_kbyte) * 8
        / NULLIF(SUM(rlc_thrptimeul_ms), 0)                                              AS ul_thp,
    (SUM(pdcp_upoctdl_kbyte) + SUM(pdcp_upoctul_kbyte)) / 1024.0/1024.0/1024.0          AS traffic_tb,
    SUM(rru_pdschprbassn_number) / NULLIF(SUM(rru_pdschprbtot_number), 0) * 100          AS dl_prb,
    SUM(rru_puschprbassn_number) / NULLIF(SUM(rru_puschprbtot_number), 0) * 100          AS ul_prb,
    (SUM(periodic_interval_minute) - SUM(rru_cellunavailabletime_s))
        / NULLIF(SUM(periodic_interval_minute), 0) * 100                                 AS availability,
    SUM(rrc_nbrmeanactiveue_number)                                                       AS active_user,
    SUM(mac_nbrreserrtbdl_number) / NULLIF(SUM(mac_nbrinittbdl_number), 0) * 100         AS dl_bler,
    SUM(mac_nbrreserrtbul_number) / NULLIF(SUM(mac_nbrinittbul_number), 0) * 100         AS ul_bler,
    100 * SUM(mac_nbrtbdl_rank4_number)
        / NULLIF(SUM(mac_nbrtbdl_rank1_number+mac_nbrtbdl_rank2_number
                    +mac_nbrtbdl_rank3_number+mac_nbrtbdl_rank4_number), 0)              AS nr_rank4
"""

def build_sql(date_from, date_to, time_level, agg_level, cell_clause):
    is_national = agg_level == "National (All)"
    is_hourly   = time_level == "Hourly"
    if is_national:
        group_cols  = "date, time" if is_hourly else "date"
        select_dims = "date, time," if is_hourly else "date,"
    else:
        group_cols  = "date, time, cell_name" if is_hourly else "date, cell_name"
        select_dims = """date, time,
            LEFT(cell_name, LENGTH(cell_name) - POSITION('_' IN REVERSE(cell_name))) AS nename,
            cell_name,""" if is_hourly else """date,
            LEFT(cell_name, LENGTH(cell_name) - POSITION('_' IN REVERSE(cell_name))) AS nename,
            cell_name,"""
    return f"""
        SELECT {select_dims}
        {KPI_COLS_SQL}
        FROM raw_counter_bai
        WHERE date BETWEEN '{date_from}' AND '{date_to}' {cell_clause}
        GROUP BY {group_cols}
        ORDER BY {group_cols}
    """

# ── Demo data (seeded from real GCELL cells so the map drill works) ────────────
def demo_data(date_from, date_to, time_level, agg_level, cells_list):
    rng     = np.random.default_rng(42)
    days    = pd.date_range(date_from, date_to, freq="D")
    is_nat  = agg_level == "National (All)"
    is_hour = time_level == "Hourly"
    cells   = cells_list if cells_list else ["FWA_DEMO_001"]
    hours   = [f"{h:02d}:00" for h in range(24)] if is_hour else [None]
    rows    = []

    def kpi_row():
        return dict(
            rrc_setup_sr=rng.uniform(95, 99.9), qos_flow_sr=rng.uniform(94, 99.5),
            sdr=rng.uniform(0.1, 2.5),          dl_thp=rng.uniform(50, 300),
            ul_thp=rng.uniform(10, 80),          traffic_tb=rng.uniform(0.001, 0.05),
            dl_prb=rng.uniform(20, 85),          ul_prb=rng.uniform(10, 60),
            availability=rng.uniform(98, 100),   active_user=int(rng.integers(5, 60)),
            dl_bler=rng.uniform(0.1, 5),         ul_bler=rng.uniform(0.1, 4),
            nr_rank4=rng.uniform(10, 60),
        )

    for d in days:
        for h in hours:
            if is_nat:
                row = {"date": d.date()}
                if h: row["time"] = h
                row.update(kpi_row()); row["traffic_tb"] = rng.uniform(0.1, 1.0)
                rows.append(row)
            else:
                for cell in cells:
                    nm  = "_".join(cell.split("_")[:2])
                    row = {"date": d.date(), "nename": nm, "cell_name": cell}
                    if h: row["time"] = h
                    row.update(kpi_row())
                    rows.append(row)
    return pd.DataFrame(rows)

# ── Session state defaults ─────────────────────────────────────────────────────
for k, v in [("connected", False), ("db_cfg", None),
             ("selected_cell", None), ("_last_map_tip", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Click-event helpers ────────────────────────────────────────────────────────
def get_clicked_cell(event):
    try:
        pts = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    if not pts:
        return None
    p  = pts[0]
    cd = p.get("customdata")
    if cd:
        return cd[0] if isinstance(cd, (list, tuple)) else cd
    return p.get("y") or p.get("label")

def handle_click(cell):
    if cell and cell != st.session_state.selected_cell:
        st.session_state.selected_cell = cell
        st.rerun()

def clickable_hbar(data, value_col, title, cmap_lohi, fmt, key, x_range=None):
    """Horizontal bar chart. When a cell is drilled in, only that cell's bar is shown."""
    sel = st.session_state.selected_cell
    line_w = [3 if c == sel else 0 for c in data["cell_name"]]
    fig = go.Figure(go.Bar(
        x=data[value_col], y=data["cell_name"], orientation="h",
        customdata=data["cell_name"].tolist(),
        marker=dict(color=data[value_col], colorscale=cmap_lohi,
                    line=dict(color=HILITE, width=line_w)),
        text=data[value_col].apply(lambda v: fmt.format(v)),
        textposition="outside", textfont=dict(color="#94a3b8", size=10),
        hovertemplate="%{y}<br>" + value_col + ": %{x:.2f}<br><i>click to drill in</i><extra></extra>",
    ))
    xaxis = dict(**AX, range=x_range) if x_range else AX
    fig.update_layout(**CT, height=340, title=title, xaxis=xaxis, yaxis=AX_REV)
    event = st.plotly_chart(fig, use_container_width=True, key=key,
                            on_select="rerun", selection_mode="points")
    return get_clicked_cell(event)

def rank_or_single(by_cell, kpi, largest=True, n=10):
    """Full top-N ranking, OR — when drilled in — just the selected cell's single bar."""
    sel = st.session_state.selected_cell
    if sel and sel in by_cell["cell_name"].values:
        return by_cell[by_cell["cell_name"] == sel]
    return by_cell.nlargest(n, kpi) if largest else by_cell.nsmallest(n, kpi)

# ── Sidebar — DB + GCELL upload ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 5G FWA KPI Dashboard")
    st.markdown("---")
    st.markdown("### 🔌 Database")
    db_host = st.text_input("Host",     value="localhost")
    db_port = st.text_input("Port",     value="5432")
    db_name = st.text_input("Database", value="surge_data")
    db_user = st.text_input("User",     value="postgres")
    db_pass = st.text_input("Password", type="password", value="surge")
    connect_btn = st.button("Connect", use_container_width=True)
    use_demo    = st.checkbox("Use demo data", value=True)

    st.markdown("---")
    st.markdown("### 🗺️ GCELL Geo File")
    geo_upload = st.file_uploader("Upload GCELL CSV", type=["csv"])

    if st.session_state.connected and not use_demo:
        st.markdown('<span class="status-badge status-connected">● CONNECTED</span>',
                    unsafe_allow_html=True)

# ── Load geo ───────────────────────────────────────────────────────────────────
geo = None
geo_err = None
try:
    if geo_upload is not None:
        geo = load_geo(geo_upload)
    else:
        import os
        if os.path.exists(GEO_CSV_PATH):
            geo = load_geo(GEO_CSV_PATH)
except Exception as e:
    geo_err = str(e)

# ── Connection ─────────────────────────────────────────────────────────────────
conn = None
if connect_btn and not use_demo:
    try:
        cfg = dict(host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)
        conn = get_conn(**cfg)
        st.session_state.connected = True
        st.session_state.db_cfg    = cfg
    except OperationalError as e:
        st.session_state.connected = False
        st.session_state.db_cfg    = None
        st.sidebar.error(f"Failed: {e}")

if st.session_state.connected and st.session_state.db_cfg and conn is None and not use_demo:
    try:
        conn = get_conn(**st.session_state.db_cfg)
    except Exception:
        st.session_state.connected = False
        conn = None

connected = st.session_state.connected and not use_demo

# ── Header ─────────────────────────────────────────────────────────────────────
hh1, hh2 = st.columns([5, 1])
with hh1:
    st.markdown("# 📡 5G FWA KPI Dashboard")
with hh2:
    badge = "status-demo" if (use_demo or not connected) else "status-connected"
    label = "◈ DEMO MODE"  if (use_demo or not connected) else "● LIVE DATA"
    st.markdown(f'<br><span class="status-badge {badge}">{label}</span>', unsafe_allow_html=True)
st.markdown("---")

# ══ MAP SECTION (top of dashboard) ══════════════════════════════════════════════
st.markdown("### 🗺️ Network Map")

DEMO_CELLS = []
if geo is None:
    if geo_err:
        st.warning(f"Couldn't read the GCELL file: {geo_err}")
    elif not FOLIUM_OK:
        st.warning("Map libraries missing. Install with: `pip install streamlit-folium folium`")
    else:
        st.info("Upload your GCELL CSV in the sidebar (or place it next to this script as "
                f"`{GEO_CSV_PATH}`) to show the network map.")
else:
    # geo filters
    mf1, mf2, mf3 = st.columns([2, 2, 2])
    with mf1:
        provs = ["All"] + sorted(geo["province"].dropna().unique().tolist()) if "province" in geo else ["All"]
        prov_sel = st.selectbox("🌍 Province", provs, index=0)
    with mf2:
        vends = sorted(geo["vendor"].dropna().unique().tolist()) if "vendor" in geo else []
        vend_sel = st.multiselect("🏭 Vendor", vends, default=[])
    with mf3:
        beam_mult = st.slider("Beam length ×", 1.0, 8.0, 3.0, 0.5,
            help="Visual exaggeration of the GCELL RADIUS (~250 m) for readability")
        HL_OPTS = {
            "Vendor colour":           None,
            "High SDR (>2%)":          ("sdr",          "gt", 2.0),
            "Low RRC SR (<95%)":       ("rrc_sr",       "lt", 95.0),
            "Low QoS Flow (<95%)":     ("qos_sr",       "lt", 95.0),
            "Low Availability (<99%)": ("availability", "lt", 99.0),
        }
        hl_choice = st.selectbox("🎨 Colour wedges red by", list(HL_OPTS.keys()), index=0)
        hl_rule = HL_OPTS[hl_choice]
        if hl_rule and hl_rule[0] not in geo.columns:
            st.caption(f"⚠️ '{hl_choice}' isn't in the GCELL file — nothing flagged.")

    scope_geo = geo.copy()
    if "province" in scope_geo and prov_sel != "All":
        scope_geo = scope_geo[scope_geo["province"] == prov_sel]
    if vend_sel and "vendor" in scope_geo:
        scope_geo = scope_geo[scope_geo["vendor"].isin(vend_sel)]

    # Demo time-series is generated from a small sample of real cells (for speed),
    # but the MAP always shows every scoped sector (all 3 per site).
    if use_demo or not connected:
        demo_pool = scope_geo.copy()
        if len(demo_pool) > 150:
            demo_pool = demo_pool.sample(150, random_state=42)
        DEMO_CELLS = demo_pool["cell_name"].tolist()
    else:
        DEMO_CELLS = []
    map_geo = scope_geo

    # cap markers for performance
    MAX_MARKERS = 2500
    capped = False
    if len(map_geo) > MAX_MARKERS:
        map_geo = map_geo.head(MAX_MARKERS); capped = True

    active_cell_now = st.session_state.selected_cell

    if FOLIUM_OK and not map_geo.empty:
        # centre / zoom
        if active_cell_now and active_cell_now in set(geo["cell_name"]):
            r = geo[geo["cell_name"] == active_cell_now].iloc[0]
            center, zoom = [r["lat"], r["lon"]], 16
        elif prov_sel != "All" or vend_sel:
            center, zoom = [map_geo["lat"].mean(), map_geo["lon"].mean()], 10
        else:
            center, zoom = [map_geo["lat"].mean(), map_geo["lon"].mean()], 7

        fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")
        mc = MarkerCluster(disableClusteringAtZoom=13).add_to(fmap)

        def fnum(v, suf=""):
            try:    return f"{float(v):.2f}{suf}"
            except: return "—"

        for r in map_geo.itertuples(index=False):
            d = r._asdict()
            col = vendor_color(d.get("vendor"))
            popup = folium.Popup(
                f"<div style='font-family:monospace;font-size:12px;min-width:170px'>"
                f"<b>{d.get('cell_name','')}</b><br>{d.get('site','')}<br>"
                f"<span style='color:#888'>{d.get('vendor','')} · {d.get('province','')}</span>"
                f"<hr style='margin:5px 0'>"
                f"RRC SR: {fnum(d.get('rrc_sr'),'%')}<br>"
                f"SDR: {fnum(d.get('sdr'),'%')}<br>"
                f"DL Thp: {fnum(d.get('dl_thp'),' Mbps')}<br>"
                f"DL PRB: {fnum(d.get('dl_prb'),'%')}</div>",
                max_width=240)
            folium.CircleMarker(
                location=[d["lat"], d["lon"]], radius=6, color=col, weight=1,
                fill=True, fill_color=col, fill_opacity=0.85,
                tooltip=d["cell_name"], popup=popup,
            ).add_to(mc)

        # ── Sector beams — every sector drawn as a wedge by default ───────────
        def kpi_breach(d):
            if not hl_rule:
                return False
            col, cmp, thr = hl_rule
            v = d.get(col)
            if v is None or pd.isna(v):
                return False
            return v > thr if cmp == "gt" else v < thr

        def draw_sector(d, highlight=False):
            rk = d.get("radius_km"); rk = 0.25 if pd.isna(rk) else float(rk)
            pts = sector_points(d["lat"], d["lon"], d.get("azimuth"),
                                d.get("beam"), rk * 1000.0 * beam_mult, steps=14)
            if not pts:
                return
            if highlight:                                   # drilled-in cell
                fill, line, op, w = "#2563eb", HILITE, 0.60, 3
            elif kpi_breach(d):                             # KPI threshold breach → red
                fill, line, op, w = "#ff2d2d", "#ff2d2d", 0.60, 1
            else:                                           # normal → full vendor colour
                fill = vendor_color(d.get("vendor")); line, w = fill, 1
                op = 0.32
            folium.Polygon(locations=pts, color=line, weight=w, fill=True,
                           fill_color=fill, fill_opacity=op,
                           tooltip=d["cell_name"]).add_to(fmap)

        # every in-view sector as a wedge (no click required)
        for s in map_geo.itertuples(index=False):
            sd = s._asdict()
            draw_sector(sd, highlight=(sd["cell_name"] == active_cell_now))

        # when drilled in, also draw any co-sited sectors missing from the view
        if active_cell_now and active_cell_now in set(geo["cell_name"]):
            sid = geo.loc[geo["cell_name"] == active_cell_now, "site_id"]
            sid = sid.iloc[0] if len(sid) and pd.notna(sid.iloc[0]) else None
            if sid is not None:
                shown = set(map_geo["cell_name"])
                for s in geo[geo["site_id"] == sid].itertuples(index=False):
                    sd = s._asdict()
                    if sd["cell_name"] not in shown:
                        draw_sector(sd, highlight=(sd["cell_name"] == active_cell_now))

        ret = st_folium(fmap, height=420, use_container_width=True, key="netmap",
                        returned_objects=["last_object_clicked_tooltip"])

        # legend
        legend = "".join(
            f'<span class="legend-pill" style="background:{c}22;color:{c};border:1px solid {c}">{n}</span>'
            for n, c in VENDOR_COLORS.items())
        if hl_rule:
            legend += ('<span class="legend-pill" style="background:#ff2d2d22;color:#ff5555;'
                       f'border:1px solid #ff2d2d">⬤ {hl_choice}</span>')
        st.markdown(f'<div class="hint">💡 Click a wedge or marker to drill into that cell · '
                    f'{len(map_geo)} sectors shown{" (capped — filter by province)" if capped else ""}</div>'
                    f'<div style="margin-top:4px">{legend}</div>', unsafe_allow_html=True)

        # handle map marker click — only act on a NEW tooltip to avoid fighting bar clicks
        tip = ret.get("last_object_clicked_tooltip") if ret else None
        if tip and tip != st.session_state._last_map_tip:
            st.session_state._last_map_tip = tip
            if tip != st.session_state.selected_cell:
                st.session_state.selected_cell = tip
                st.rerun()

st.markdown("---")

# ── Filter Bar ────────────────────────────────────────────────────────────────
st.markdown("### 🎛️ Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    date_from = st.date_input("📅 Start Date", datetime.date(2026, 4, 26))
with f2:
    date_to = st.date_input("📅 End Date", datetime.date.today())
with f3:
    agg_level = st.selectbox("📊 Aggregation Level", ["National (All)", "Vendor"], index=1,
        help="National = all cells aggregated\nVendor = per-cell (needed to drill into a cell)")
with f4:
    time_level = st.selectbox("⏱️ Time Level", ["Daily", "Hourly"], index=0,
        help="Daily = 1 row per day | Hourly = 1 row per hour")

is_national = agg_level == "National (All)"

if not is_national:
    if use_demo or not connected:
        all_cells = DEMO_CELLS
    else:
        try:
            cell_df   = run_query(conn,
                f"SELECT DISTINCT cell_name FROM raw_counter_bai "
                f"WHERE date BETWEEN '{date_from}' AND '{date_to}' ORDER BY cell_name")
            all_cells = cell_df["cell_name"].tolist()
        except Exception:
            all_cells = []
    selected_cells = st.multiselect("📡 Cell Name (leave empty = all cells)",
        options=all_cells, default=[], placeholder="Type to search cells...")
else:
    selected_cells = []
    st.info("ℹ️ Cell filter & map drill-down are disabled in **National** mode — switch to **Vendor**.")

st.markdown("---")

# ── SQL cell clause ────────────────────────────────────────────────────────────
if selected_cells and not is_national:
    cell_list   = ", ".join(f"'{c}'" for c in selected_cells)
    cell_clause = f"AND cell_name IN ({cell_list})"
else:
    cell_clause = ""

# ── Load data ──────────────────────────────────────────────────────────────────
if use_demo or not connected:
    demo_cells = list(selected_cells or DEMO_CELLS)
    sc = st.session_state.selected_cell
    if sc and sc not in demo_cells:
        demo_cells.append(sc)          # clicking any wedge drills in, even in demo
    df = demo_data(date_from, date_to, time_level, agg_level, demo_cells)
else:
    try:
        df = run_query(conn, build_sql(date_from, date_to, time_level, agg_level, cell_clause))
    except Exception as e:
        st.error(f"Query error: {e}")
        df = demo_data(date_from, date_to, time_level, agg_level, selected_cells or DEMO_CELLS)

# ── x-axis ─────────────────────────────────────────────────────────────────────
if time_level == "Hourly" and "time" in df.columns:
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    x_col = "datetime"
else:
    x_col = "date"

# ── Resolve drill-down ─────────────────────────────────────────────────────────
if is_national or "cell_name" not in df.columns:
    st.session_state.selected_cell = None
elif (st.session_state.selected_cell is not None
      and st.session_state.selected_cell not in df["cell_name"].unique()):
    st.session_state.selected_cell = None

active_cell = st.session_state.selected_cell
df_view = df[df["cell_name"] == active_cell] if (active_cell and "cell_name" in df.columns) else df

# ── Aggregations ───────────────────────────────────────────────────────────────
KPI_COLS = ["rrc_setup_sr","qos_flow_sr","sdr","dl_thp","ul_thp","traffic_tb",
            "dl_prb","ul_prb","availability","active_user","dl_bler","ul_bler","nr_rank4"]
agg_dict = {k: "mean" for k in KPI_COLS if k in df_view.columns}
agg_dict["traffic_tb"]  = "sum"; agg_dict["active_user"] = "sum"
trend = df_view.groupby(x_col).agg(agg_dict).reset_index()

by_cell = None
if not is_national and "cell_name" in df.columns:
    bc = {k: "mean" for k in KPI_COLS if k in df.columns}
    bc["traffic_tb"] = "sum"; bc["active_user"] = "sum"
    by_cell = df.groupby("cell_name").agg(bc).reset_index()

# ── Drill banner ───────────────────────────────────────────────────────────────
if active_cell:
    b1, b2 = st.columns([6, 1])
    with b1:
        st.markdown(f'<div class="drill-banner">🔎 DRILL-DOWN ACTIVE &nbsp;·&nbsp; '
                    f'showing cell <b>{active_cell}</b> only</div>', unsafe_allow_html=True)
    with b2:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.selected_cell = None
            st.rerun()

# ── Scope line + KPI cards ─────────────────────────────────────────────────────
if active_cell:
    scope_txt = f"Drill-down → **{active_cell}**"
else:
    n_cells   = df["cell_name"].nunique() if "cell_name" in df.columns else "All"
    scope_txt = "National" if is_national else f"{n_cells} cells"
st.markdown(f"**Scope:** {scope_txt} &nbsp;|&nbsp; **Time Level:** {time_level} &nbsp;|&nbsp; "
            f"**Period:** {date_from.strftime('%d %b %Y')} → {date_to.strftime('%d %b %Y')}")

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("RRC Setup SR",   f"{trend['rrc_setup_sr'].mean():.2f}%")
k2.metric("QoS Flow SR",    f"{trend['qos_flow_sr'].mean():.2f}%")
k3.metric("SDR",            f"{trend['sdr'].mean():.2f}%")
k4.metric("Availability",   f"{trend['availability'].mean():.2f}%")
k5.metric("DL Throughput",  f"{trend['dl_thp'].mean():.1f} Mbps")
k6.metric("UL Throughput",  f"{trend['ul_thp'].mean():.1f} Mbps")
k7.metric("Traffic Volume", f"{trend['traffic_tb'].sum():.3f} TB")
k8.metric("Active Users",   f"{int(trend['active_user'].sum()):,}")
st.markdown("---")

# ── Chart helper ───────────────────────────────────────────────────────────────
def line_chart(x, y, color, fill_color, title, target=None, target_label="",
               height=260, y_range=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=2.5),
                             fill="tozeroy", fillcolor=fill_color, showlegend=False))
    if target is not None:
        fig.add_hline(y=target, line_dash="dash", line_color="#fb7185",
                      annotation_text=target_label, annotation_font_color="#fb7185")
    fig.update_layout(**CT, height=height, title=f"{title} [{time_level}]",
                      xaxis=AX, yaxis=dict(**AX, **({"range": y_range} if y_range else {})))
    return fig

# ── Success Rates ──────────────────────────────────────────────────────────────
st.markdown("### 📈 Success Rate Trends")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_chart(trend[x_col], trend["rrc_setup_sr"], PAL[0],
        "rgba(56,189,248,0.07)", "RRC Setup SR (%)", 95, "Target 95%", y_range=[80,101]),
        use_container_width=True)
with c2:
    st.plotly_chart(line_chart(trend[x_col], trend["qos_flow_sr"], PAL[1],
        "rgba(129,140,248,0.07)", "QoS Flow SR (%)", 95, "Target 95%", y_range=[80,101]),
        use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(line_chart(trend[x_col], trend["sdr"], PAL[4],
        "rgba(251,113,133,0.07)", "SDR (%)", 2, "Threshold 2%", height=240),
        use_container_width=True)
with c4:
    st.plotly_chart(line_chart(trend[x_col], trend["availability"], PAL[2],
        "rgba(52,211,153,0.07)", "Cell Availability (%)", 99, "Target 99%",
        height=240, y_range=[90,101]), use_container_width=True)

# ── Throughput + Traffic ───────────────────────────────────────────────────────
st.markdown("### 🚀 Throughput & Traffic")
c5, c6 = st.columns(2)
with c5:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend[x_col], y=trend["dl_thp"], name="DL", marker_color=PAL[0], opacity=0.85))
    fig.add_trace(go.Bar(x=trend[x_col], y=trend["ul_thp"], name="UL", marker_color=PAL[1], opacity=0.85))
    fig.update_layout(**CT, height=260, title=f"DL / UL Throughput (Mbps) [{time_level}]",
                      barmode="group", xaxis=AX, yaxis=AX,
                      legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
with c6:
    st.plotly_chart(line_chart(trend[x_col], trend["traffic_tb"], PAL[3],
        "rgba(245,158,11,0.07)", "Traffic Volume (TB)", height=260), use_container_width=True)

# ── PRB + BLER ─────────────────────────────────────────────────────────────────
st.markdown("### 📶 PRB Usage & BLER")
c7, c8 = st.columns(2)
with c7:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["dl_prb"], name="DL PRB",
                             line=dict(color=PAL[0], width=2)))
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["ul_prb"], name="UL PRB",
                             line=dict(color=PAL[1], width=2, dash="dot")))
    fig.add_hline(y=80, line_dash="dash", line_color="#fb7185",
                  annotation_text="Warning 80%", annotation_font_color="#fb7185")
    fig.update_layout(**CT, height=240, title=f"DL / UL PRB Usage (%) [{time_level}]",
                      xaxis=AX, yaxis=dict(**AX, range=[0,100]),
                      legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)
with c8:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["dl_bler"], name="DL BLER",
                             line=dict(color=PAL[4], width=2)))
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["ul_bler"], name="UL BLER",
                             line=dict(color=PAL[3], width=2, dash="dot")))
    fig.add_hline(y=10, line_dash="dash", line_color="#fb7185",
                  annotation_text="Threshold 10%", annotation_font_color="#fb7185")
    fig.update_layout(**CT, height=240, title=f"DL / UL BLER (%) [{time_level}]",
                      xaxis=AX, yaxis=AX,
                      legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

# ── Worst Performers / Selected-cell snapshot ──────────────────────────────────
if by_cell is not None and not by_cell.empty:
    if active_cell:
        st.markdown(f"### 🎯 Selected Cell — {active_cell}")
        st.markdown('<div class="hint">Showing only the drilled-in cell · '
                    'click another map marker to switch · ✕ Clear to see full rankings</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("### ⚠️ Worst Performing Cells")
        st.markdown('<div class="hint">💡 Click any bar (or a map marker) to drill into that cell</div>',
                    unsafe_allow_html=True)

    cr1, cr2 = st.columns(2)
    with cr1:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "sdr", largest=True), "sdr",
            "🔴 Highest SDR — Worst Drop Rate",
            [[0, PAL[3]], [1, PAL[4]]], "{:.2f}%", key="bar_sdr"))
    with cr2:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "rrc_setup_sr", largest=False), "rrc_setup_sr",
            "🔴 Lowest RRC Setup SR — Worst Success Rate",
            [[0, PAL[4]], [1, PAL[3]]], "{:.1f}%", key="bar_rrc", x_range=[0, 105]))

    cr3, cr4 = st.columns(2)
    with cr3:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "dl_thp", largest=False), "dl_thp",
            "🔴 Lowest DL Throughput (Mbps)",
            [[0, PAL[4]], [1, PAL[3]]], "{:.1f}", key="bar_dlthp"))
    with cr4:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "dl_prb", largest=True), "dl_prb",
            "🔴 Highest DL PRB Usage — Most Congested",
            [[0, PAL[3]], [1, PAL[4]]], "{:.1f}%", key="bar_dlprb", x_range=[0, 105]))

# ── Raw data ───────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Raw KPI Data" + (f"  —  filtered to {active_cell}" if active_cell else "")):
    disp = df_view.copy()
    for col in ["rrc_setup_sr","qos_flow_sr","sdr","dl_prb","ul_prb",
                "availability","dl_bler","ul_bler","nr_rank4"]:
        if col in disp.columns: disp[col] = disp[col].round(2)
    for col in ["dl_thp","ul_thp"]:
        if col in disp.columns: disp[col] = disp[col].round(1)
    sort_cols = [c for c in ["date","time","cell_name","nename"] if c in disp.columns]
    st.dataframe(disp.sort_values(sort_cols, ascending=[False]*len(sort_cols))
                    .reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;font-family:Space Mono,monospace;font-size:11px;"
            "color:#334155;letter-spacing:1px'>5G NR KPI Dashboard · Baicells · surge_data</div>",
            unsafe_allow_html=True)