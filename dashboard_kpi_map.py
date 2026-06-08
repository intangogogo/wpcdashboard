import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
from psycopg2 import OperationalError
import datetime
import numpy as np
import math

# ── Auth module ────────────────────────────────────────────────────────────────
import auth  # auth.py must be in the same folder as this script

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
    page_title="5G FWA KPI Analysis Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config: where to find the GCELL geo file on your machine ───────────────────
GEO_CSV_PATH = "GCELL_W23.csv"   # <── put your GCELL csv next to this script, or upload via sidebar

# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# DB credentials for the users table (same surge_data DB).
# Override via environment variables for production.
# ══════════════════════════════════════════════════════════════════════════════
import os as _os

_AUTH_CFG = dict(
    host    = _os.environ.get("AUTH_DB_HOST",    "localhost"),
    port    = int(_os.environ.get("AUTH_DB_PORT", 5432)),
    dbname  = _os.environ.get("AUTH_DB_NAME",   "surge_data"),
    user    = _os.environ.get("AUTH_DB_USER",   "postgres"),
    password= _os.environ.get("AUTH_DB_PASSWORD","surge"),
)

# Ensure the users table exists (no-op after first run)
try:
    auth.init_db(_AUTH_CFG)
except Exception as _e:
    st.error(f"⚠️ Cannot connect to auth database: {_e}")
    st.stop()

# Show login if not authenticated
if not auth.is_logged_in():
    if auth.render_login(_AUTH_CFG):
        st.rerun()
    st.stop()

# Force password change if flagged
if auth.must_change_pw():
    auth.render_force_change_pw(_AUTH_CFG)
    st.stop()

# Reached here = authenticated
_current_user = auth.current_user()
_current_role = auth.current_role()
_is_admin     = auth.is_admin()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* dark-red page background + light text */
.stApp { background-color: #5c0d12; color: #f3e3e3; }
.stApp p, .stApp li { color: #f3e3e3; }

/* bright-red left sidebar */
section[data-testid="stSidebar"] { background-color: #DD1E26; border-right: 1px solid #7a0f14; }
section[data-testid="stSidebar"] * { color: #ffffff; }
section[data-testid="stSidebar"] input { color: #1f2430; background: #ffffff; }

/* widget labels + markdown stay light on the dark-red bg */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #f3e3e3 !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li { color: #f3e3e3; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #f3e3e3 !important; }

/* metric cards: transparent on dark-red, light label, white value */
[data-testid="metric-container"] {
    background: transparent; border: none; border-left: 3px solid #DD1E26;
    border-radius: 0; padding: 4px 0 4px 10px; box-shadow: none;
}
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace; font-size: 10px !important;
    letter-spacing: 1px; text-transform: uppercase; color: #e7b9b9 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace; font-size: 20px !important; color: #ffffff !important;
}

/* greyish rounded chart cards */
div[data-testid="stPlotlyChart"] {
    background: #DFDFDF; border-radius: 10px; overflow: hidden;
    box-shadow: 0 3px 12px rgba(0,0,0,0.22);
}

/* headers light on dark-red */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #ffffff !important; font-size: 1.4rem !important; }
h2 { color: #ffffff !important; }
h3 { color: #ffd4d4 !important; font-size: 0.95rem !important; }
hr { border-color: #7a0f14; }

.stButton > button {
    background: linear-gradient(135deg, #ffffff, #ffe3e3);
    color: #b3161c; font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 12px; letter-spacing: 1px;
    border: none; border-radius: 8px;
}
.status-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 1px;
}
.status-connected { background: #e7f6ec; color: #1a7f37; border: 1px solid #9bdcb0; }
.status-demo      { background: #fff4e5; color: #b45309; border: 1px solid #f6c684; }
.drill-banner {
    background: #ffffff; border: 1px solid #DD1E26; border-radius: 10px; padding: 10px 16px;
    font-family: 'Space Mono', monospace; font-size: 13px; color: #b3161c; letter-spacing: 0.5px;
}
.hint {
    font-family: 'Space Mono', monospace; font-size: 11px; color: #f0c4c4 !important; letter-spacing: 0.5px;
}
.legend-pill {
    display:inline-block; padding:2px 9px; margin:2px 4px; border-radius:999px;
    font-family:'Space Mono',monospace; font-size:10px; letter-spacing:.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
CT = dict(
    paper_bgcolor="#DFDFDF", plot_bgcolor="#DFDFDF",
    font_color="#3a4150", font_family="DM Sans", margin=dict(l=12, r=12, t=48, b=10),
    legend_font_color="#939393",
)
AX     = dict(gridcolor="#c4c4c4", showline=False, tickfont_color="#5b6270")
AX_REV = dict(gridcolor="#c4c4c4", showline=False, tickfont_color="#5b6270", autorange="reversed")
PAL    = ["#DD1E26", "#2563eb", "#16a34a", "#f59e0b", "#db2777", "#7c3aed", "#0891b2"]
HILITE = "#FFD400"   # bright highlight for the drilled-into cell on the dark map

_TITLE_FONT = dict(color="#4D4D4D", size=14, family="Space Mono")
def _T(text):
    """Return a Plotly title dict with explicit font — survives Streamlit's theme injection."""
    return dict(text=text, font=_TITLE_FONT)

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
_GEO_COLS = ["cell_name","site","site_id","vendor","lat","lon",
             "province","kab","kec","rrc_sr","sdr","qos_sr","dl_thp","dl_prb",
             "azimuth","radius_km","beam","mtilt","etilt"]

_GEO_RENAME = {
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

_GEO_NUMERIC = ["lat","lon","rrc_sr","sdr","qos_sr","dl_thp","dl_prb",
                "azimuth","radius_km","beam","mtilt","etilt"]

_CREATE_GCELL_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_gcell (
    id          SERIAL PRIMARY KEY,
    cell_name   VARCHAR(128) UNIQUE NOT NULL,
    site        VARCHAR(128),
    site_id     VARCHAR(64),
    vendor      VARCHAR(64),
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    province    VARCHAR(64),
    kab         VARCHAR(64),
    kec         VARCHAR(64),
    rrc_sr      DOUBLE PRECISION,
    sdr         DOUBLE PRECISION,
    qos_sr      DOUBLE PRECISION,
    dl_thp      DOUBLE PRECISION,
    dl_prb      DOUBLE PRECISION,
    azimuth     DOUBLE PRECISION,
    radius_km   DOUBLE PRECISION,
    beam        DOUBLE PRECISION,
    mtilt       DOUBLE PRECISION,
    etilt       DOUBLE PRECISION,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

def _parse_geo_df(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalise a raw GCELL CSV dataframe into the standard schema."""
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.rename(columns={k: v for k, v in _GEO_RENAME.items() if k in df.columns})
    for c in _GEO_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "cell_name"])
    if "vendor" in df.columns:
        df["vendor"] = df["vendor"].astype(str).str.replace("FiberHome", "Fiberhome", regex=False)
    keep = [c for c in _GEO_COLS if c in df.columns]
    return df[keep].drop_duplicates("cell_name").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_geo(source):
    """source = uploaded file buffer OR a filesystem path string."""
    return _parse_geo_df(pd.read_csv(source))

def init_gcell_table(cfg: dict):
    """Create dashboard_gcell if it doesn't exist."""
    with get_conn(**cfg) as c, c.cursor() as cur:
        cur.execute(_CREATE_GCELL_TABLE)
        c.commit()

def save_geo_to_db(cfg: dict, df: "pd.DataFrame"):
    """Upsert all rows from df into dashboard_gcell."""
    cols = [c for c in _GEO_COLS if c in df.columns]
    with get_conn(**cfg) as c, c.cursor() as cur:
        # truncate + re-insert is simplest for a config table
        cur.execute("TRUNCATE TABLE dashboard_gcell")
        for _, row in df.iterrows():
            vals = [row.get(col) for col in cols]
            ph   = ", ".join(["%s"] * len(cols))
            col_str = ", ".join(cols)
            cur.execute(
                f"INSERT INTO dashboard_gcell ({col_str}) VALUES ({ph})",
                vals
            )
        c.commit()

@st.cache_data(show_spinner=False, ttl=300)
def load_geo_from_db(cfg_tuple) -> "tuple[pd.DataFrame | None, str | None]":
    """
    Load geo from dashboard_gcell table.
    cfg_tuple is a hashable tuple version of the cfg dict for st.cache_data.
    Returns (df, last_updated_str) or (None, None) if table is empty.
    """
    cfg = dict(cfg_tuple)
    try:
        init_gcell_table(cfg)
        df = pd.read_sql_query(
            "SELECT * FROM dashboard_gcell ORDER BY cell_name", get_conn(**cfg)
        )
        if df.empty:
            return None, None
        ts_col = "uploaded_at"
        last_ts = None
        if ts_col in df.columns:
            last_ts = pd.to_datetime(df[ts_col]).max()
            df = df.drop(columns=[ts_col, "id"], errors="ignore")
        ts_str = last_ts.strftime("%d %b %Y %H:%M") if last_ts is not None else "unknown"
        return df, ts_str
    except Exception:
        return None, None

# ── SQL Templates ──────────────────────────────────────────────────────────────
KPI_COLS_SQL = """
    SUM(rrc_sr_num)            / NULLIF(SUM(rrc_sr_denum), 0)            * 100   AS rrc_setup_sr,
    SUM(qos_flow_sr_num)       / NULLIF(SUM(qos_flow_sr_denum), 0)       * 100   AS qos_flow_sr,
    SUM(service_drop_rate_num) / NULLIF(SUM(service_drop_rate_denum), 0) * 100   AS sdr,
    SUM(dl_throughput_num)     / NULLIF(SUM(dl_throughput_denum), 0)             AS dl_thp,
    SUM(ul_throughput_num)     / NULLIF(SUM(ul_throughput_denum), 0)             AS ul_thp,
    SUM(total_traffic_tb)                                                        AS traffic_tb,
    SUM(dl_prb_usage_num)      / NULLIF(SUM(dl_prb_usage_denum), 0)      * 100   AS dl_prb,
    SUM(ul_prb_usage_num)      / NULLIF(SUM(ul_prb_usage_denum), 0)      * 100   AS ul_prb,
    SUM(availability_num)      / NULLIF(SUM(availability_denum), 0)      * 100   AS availability,
    AVG(active_user)                                                             AS active_user,
    SUM(rank4_num)             / NULLIF(SUM(rank4_denum), 0)             * 100   AS nr_rank4,
    SUM(average_cqi_num)       / NULLIF(SUM(average_cqi_denum), 0)               AS average_cqi,
    SUM(qpsk_ratio_num)        / NULLIF(SUM(qpsk_ratio_denum), 0)        * 100   AS qpsk_ratio,
    SUM(rank2_num)             / NULLIF(SUM(rank2_denum), 0)             * 100   AS nr_rank2,
    SUM(dl_se_num)             / NULLIF(SUM(dl_se_denum), 0)                     AS dl_se
"""

def build_sql(date_from, date_to, time_level, agg_level, cell_clause):
    is_national = agg_level == "National (All)"
    is_hourly   = time_level == "Hourly"
    if is_national:
        group_cols  = "date, time" if is_hourly else "date"
        select_dims = "date, time," if is_hourly else "date,"
    else:
        group_cols  = "date, time, cell_name, ne_name" if is_hourly else "date, cell_name, ne_name"
        select_dims = "date, time, ne_name AS nename, cell_name," if is_hourly \
                      else "date, ne_name AS nename, cell_name,"
    return f"""
        SELECT {select_dims}
        {KPI_COLS_SQL}
        FROM raw_dashboard_kpi
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
            rrc_setup_sr=rng.uniform(93, 99.9), qos_flow_sr=rng.uniform(93, 99.5),
            sdr=rng.uniform(0.1, 7.5),          dl_thp=rng.uniform(1, 300),
            ul_thp=rng.uniform(1, 80),           traffic_tb=rng.uniform(0.001, 0.05),
            dl_prb=rng.uniform(20, 98),          ul_prb=rng.uniform(10, 70),
            availability=rng.uniform(97, 100),   active_user=int(rng.integers(5, 60)),
            nr_rank4=rng.uniform(10, 60),
            average_cqi=rng.uniform(4, 12),      qpsk_ratio=rng.uniform(30, 80),
            nr_rank2=rng.uniform(15, 60),         dl_se=rng.uniform(2, 8),
            ul_ni_avg=rng.uniform(-115, -85),    # not in raw_dashboard_kpi (demo only)
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
    fig.update_layout(**CT, height=340, title=_T(title), xaxis=xaxis, yaxis=AX_REV)
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
    st.markdown("## 📡 5G FWA KPI Analysis")

    # ── Logged-in user info + logout ──────────────────────────────────────────
    role_icon = "🛡️" if _is_admin else "👤"
    st.markdown(f"""<div style='background:#3d080c;border:1px solid #7a0f14;border-radius:10px;
             padding:10px 14px;margin-bottom:8px'>
  <div style='font-family:Space Mono,monospace;font-size:11px;color:#e7b9b9;
               letter-spacing:1px'>LOGGED IN AS</div>
  <div style='font-family:Space Mono,monospace;font-size:15px;color:#ffffff;
               font-weight:700;margin-top:2px'>{role_icon} {_current_user}</div>
  <div style='font-family:Space Mono,monospace;font-size:10px;color:#ffd4d4;
               letter-spacing:2px;text-transform:uppercase'>{_current_role}</div>
</div>""", unsafe_allow_html=True)

    if st.button("Logout", use_container_width=True, key="_sidebar_logout"):
        auth.logout()
        st.rerun()

    auth.render_change_own_pw_widget(_AUTH_CFG)

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
    geo_upload = st.file_uploader("Upload & save GCELL CSV", type=["csv"],
        help="Upload to update the stored GCELL data. Map always loads from last saved data.")
    if geo_upload is not None:
        st.session_state["_geo_pending_upload"] = geo_upload

    if st.session_state.connected and not use_demo:
        st.markdown('<span class="status-badge status-connected">● CONNECTED</span>',
                    unsafe_allow_html=True)

# ── Load geo ───────────────────────────────────────────────────────────────────
geo      = None
geo_err  = None
geo_ts   = None   # "last updated" timestamp string

# 1. If a new CSV was just uploaded → parse + save to DB
_pending = st.session_state.get("_geo_pending_upload")
if _pending is not None:
    try:
        _gdf = load_geo(_pending)
        # save to DB only when DB credentials are available
        _save_cfg = st.session_state.get("db_cfg") or dict(
            host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)
        try:
            init_gcell_table(_save_cfg)
            save_geo_to_db(_save_cfg, _gdf)
            load_geo_from_db.clear()   # bust cache so next read picks up new data
            st.session_state["_geo_pending_upload"] = None
            st.sidebar.success(f"✅ GCELL saved — {len(_gdf)} sectors in DB.")
        except Exception as _dbe:
            st.sidebar.warning(f"⚠️ Couldn't persist to DB: {_dbe}\nUsing in-memory only.")
        geo = _gdf
    except Exception as e:
        geo_err = str(e)

# 2. If nothing uploaded yet → try loading from DB (falls back to local CSV)
if geo is None:
    _db_cfg = st.session_state.get("db_cfg") or dict(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_pass)
    _cfg_tuple = tuple(sorted(_db_cfg.items()))
    geo, geo_ts = load_geo_from_db(_cfg_tuple)

# 3. Last resort: local filesystem CSV (backwards compat)
if geo is None:
    try:
        import os
        if os.path.exists(GEO_CSV_PATH):
            geo = load_geo(GEO_CSV_PATH)
            geo_ts = "local file"
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
    st.markdown("# 📡 5G FWA KPI Analysis Dashboard")
with hh2:
    badge = "status-demo" if (use_demo or not connected) else "status-connected"
    label = "◈ DEMO MODE"  if (use_demo or not connected) else "● LIVE DATA"
    st.markdown(f'<br><span class="status-badge {badge}">{label}</span>', unsafe_allow_html=True)
st.markdown("---")

# ══ MAP SECTION (top of dashboard) ══════════════════════════════════════════════
st.markdown("### 🗺️ Network Map")

DEMO_CELLS = []
if geo is None:
    if not FOLIUM_OK:
        st.warning("Map libraries missing. Install with: `pip install streamlit-folium folium`")
    else:
        st.info("📂 No GCELL data in database yet. Upload your **GCELL CSV** in the sidebar to enable the map. "
                "Once uploaded it will persist — you won't need to re-upload on every visit.")
    if geo_err:
        st.caption(f"⚠️ Load error: {geo_err}")
else:
    if geo_ts:
        st.caption(f"🗺️ Geo data last updated: **{geo_ts}** · {len(geo)} sectors · "
                   "Upload a new CSV in the sidebar to refresh.")
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
            "Vendor colour":             None,
            "High SDR (>5%)":            ("sdr",          "gt",  5.0),
            "Low RRC SR (<97%)":         ("rrc_sr",       "lt", 97.0),
            "Low QoS Flow (<97%)":       ("qos_sr",       "lt", 97.0),
            "Low Availability (<99%)":   ("availability", "lt", 99.0),
            "Low CQI (<8)":              ("average_cqi",  "lt",  8.0),
            "High QPSK Ratio (>60%)":    ("qpsk_ratio",   "gt", 60.0),
            "Low Rank 2 (<40%)":         ("nr_rank2",     "lt", 40.0),
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
                fill, line, op, w = "#249E94", HILITE, 0.70, 3
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
                f"SELECT DISTINCT cell_name FROM raw_dashboard_kpi "
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
            "dl_prb","ul_prb","availability","active_user","nr_rank4",
            "average_cqi","qpsk_ratio","nr_rank2","dl_se","ul_ni_avg"]
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

import datetime as _dt

# "good direction" for each KPI: True = higher is better
KPI_DIR = {"rrc_setup_sr": True, "qos_flow_sr": True, "sdr": False, "availability": True,
           "average_cqi": True, "dl_thp": True, "dl_prb": False, "qpsk_ratio": False,
           "nr_rank2": True, "ul_ni_avg": False, "traffic_tb": True}

def _daily_series(col):
    if col not in df.columns:
        return None
    # traffic is a sum per day; everything else is mean
    agg = "sum" if col in ("traffic_tb", "active_user") else "mean"
    s = df.groupby("date")[col].agg(agg).sort_index()
    return s if len(s) else None

def kpi_delta(col):
    """(latest-day value, change vs ~D-7, improved?)"""
    s = _daily_series(col)
    if s is None or s.empty:
        return None, None, None
    cur = s.iloc[-1]
    last = s.index[-1]
    d7 = None
    try:
        target = last - _dt.timedelta(days=7)
        prior = s[s.index <= target]
        d7 = prior.iloc[-1] if len(prior) else (s.iloc[0] if len(s) > 1 else None)
    except Exception:
        d7 = s.iloc[0] if len(s) > 1 else None
    if d7 is None or pd.isna(d7) or pd.isna(cur):
        return cur, None, None
    delta = cur - d7
    higher_better = KPI_DIR.get(col, True)
    improved = None if abs(delta) < 1e-9 else ((delta > 0) if higher_better else (delta < 0))
    return cur, delta, improved

def metric_card(label, col, unit="", dec=2):
    cur, delta, improved = kpi_delta(col)
    if cur is None:
        val = "—"; big_arrow = ""; delta_html = "<span style='color:#e0a8a8;font-size:10px'>not in raw_dashboard_kpi</span>"
    else:
        val = f"{cur:.{dec}f}{unit}"
        big_arrow = ("↑" if improved else "↓") if improved is not None else ""
        if delta is None:
            delta_html = "<span style='color:#e0a8a8;font-size:10px'>no D-7 data</span>"
        else:
            sm_arrow = "+" if improved else ("-" if improved is False else "—")
            delta_html = (f"<span style='color:#ffffff'>{sm_arrow} {abs(delta):.{dec}f}{unit}</span> "
                          f"<span style='color:#e0a8a8;font-size:10px'>vs D-7</span>")
    return (f"<div style='border-left:3px solid #DD1E26;padding:2px 0 8px 10px;margin-bottom:4px'>"
            f"<div style='font-family:Space Mono,monospace;font-size:10px;letter-spacing:1px;"
            f"text-transform:uppercase;color:#e7b9b9'>{label}</div>"
            f"<div style='font-family:Space Mono,monospace;font-size:26px;color:#ffffff;"
            f"line-height:1.15'>{val}&nbsp;{big_arrow}</div>"
            f"<div style='font-family:Space Mono,monospace;font-size:12px'>{delta_html}</div></div>")

ROW1 = [("Traffic",     "traffic_tb",   " TB",   3),
        ("RRC SR",      "rrc_setup_sr", "%",     2),
        ("QOS SR",      "qos_flow_sr",  "%",     2),
        ("SDR",         "sdr",          "%",     2),
        ("Availability","availability", "%",     2)]
ROW2 = [("PRB",         "dl_prb",       "%",     1),
        ("QPSK",        "qpsk_ratio",   "%",     1),
        ("RSSI",        "ul_ni_avg",    " dBm",  1),
        ("Rank 2",      "nr_rank2",     "%",     1),
        ("CQI",         "average_cqi",  "",      2)]
for rowdef in (ROW1, ROW2):
    n = len(rowdef)
    cols = st.columns(n)
    for c, (lbl, col, u, dec) in zip(cols, rowdef):
        c.markdown(metric_card(lbl, col, u, dec), unsafe_allow_html=True)
st.markdown("<div class='hint'>↑ improved · ↓ degraded &nbsp;·&nbsp; +/− delta vs 7 days earlier (D-7)</div>",
            unsafe_allow_html=True)
st.markdown("---")

# ── WPC Analysis & RCA ─────────────────────────────────────────────────────────
st.markdown("### 🧰 WPC Analysis & RCA")

WPC_RULES = {  # label : (column, comparison, threshold)
    "Availability": ("availability",  "lt", 99.0),
    "QOS SR":       ("qos_flow_sr",   "lt", 97.0),
    "RRC SR":       ("rrc_setup_sr",  "lt", 97.0),
    "SDR":          ("sdr",           "gt", 5.0),
    "LTC":          ("dl_thp",        "lt", 3.0),
}
WPC_COLORS = {"Availability": "#4472C4", "QOS SR": "#ED7D31", "RRC SR": "#A5A5A5",
              "SDR": "#FFC000", "LTC": "#5B9BD5"}

def _breach(frame, col, cmp, thr):
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    s = pd.to_numeric(frame[col], errors="coerce")
    return (s > thr) if cmp == "gt" else (s < thr)

def auto_rca(r):
    """First-match heuristic RCA from a breaching cell's KPI signature."""
    g = lambda k, d: (d if pd.isna(r.get(k, np.nan)) else r.get(k))
    if g("ul_ni_avg", -120) > -100:                              return "Interference"
    if g("dl_prb", 0) > 95:                                       return "Capacity"
    if g("qpsk_ratio", 0) > 60 or g("average_cqi", 99) < 8:       return "Coverage"
    if g("nr_rank2", 99) < 40:                                    return "Coverage - ISD >2km"
    if g("dl_thp", 99) < 3:                                       return "Coverage - End Cell"
    if (g("rrc_setup_sr", 100) < 97 or g("qos_flow_sr", 100) < 97) and g("active_user", 99) < 10:
        return "Low Attempt"
    return "Hygiene"

if is_national or "cell_name" not in df.columns:
    st.info("ℹ️ Switch **Aggregation Level** to **Vendor** for per-cell WPC breach analysis.")
else:
    dates = sorted(df["date"].unique())

    # daily breach counts per KPI category
    wcat = pd.DataFrame({"date": dates})
    for lbl, (col, cmp, thr) in WPC_RULES.items():
        cnt = df[_breach(df, col, cmp, thr)].groupby("date")["cell_name"].nunique()
        wcat[lbl] = wcat["date"].map(cnt).fillna(0).astype(int)

    # Open = cells breaching any KPI that day · Closed = recovered vs the previous day
    anyb = pd.Series(False, index=df.index)
    for lbl, (col, cmp, thr) in WPC_RULES.items():
        anyb = anyb | _breach(df, col, cmp, thr)
    open_sets = {d: set(df.loc[anyb & (df["date"] == d), "cell_name"]) for d in dates}
    opens, closeds, prev = [], [], set()
    for d in dates:
        cur = open_sets[d]
        opens.append(len(cur)); closeds.append(len(prev - cur)); prev = cur
    wstat = pd.DataFrame({"date": dates, "Open": opens, "Closed": closeds})

    # summary metric strip
    msum = st.columns(6)
    msum[0].metric("Open (latest day)", int(wstat["Open"].iloc[-1]) if len(wstat) else 0)
    for i, lbl in enumerate(WPC_RULES):
        msum[i + 1].metric(f"{lbl} hits", int(wcat[lbl].sum()))

    a1, a2 = st.columns(2)
    with a1:
        fig = go.Figure()
        for lbl in WPC_RULES:
            fig.add_trace(go.Bar(x=wcat["date"], y=wcat[lbl], name=lbl,
                                 marker_color=WPC_COLORS[lbl]))
        fig.update_layout(**CT, height=320, barmode="stack",
                          title=_T("WPC Category — Daily Breach Count"),
                          xaxis=AX, yaxis=AX,
                          legend=dict(orientation="h", y=1.13, bgcolor="rgba(0,0,0,0)", font=dict(color="#939393")))
        st.plotly_chart(fig, use_container_width=True, key="wpc_cat")
    with a2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=wstat["date"], y=wstat["Closed"], name="Closed", marker_color="#70AD47"))
        fig.add_trace(go.Bar(x=wstat["date"], y=wstat["Open"],   name="Open",   marker_color="#ED7D31"))
        fig.update_layout(**CT, height=320, barmode="stack",
                          title=_T("WPC Status — Open vs Closed (recovered)"),
                          xaxis=AX, yaxis=AX,
                          legend=dict(orientation="h", y=1.13, bgcolor="rgba(0,0,0,0)", font=dict(color="#939393")))
        st.plotly_chart(fig, use_container_width=True, key="wpc_stat")

    # RCA auto-classification on the cells breaching any KPI (period aggregate)
    breach_cells = pd.DataFrame()
    if by_cell is not None and not by_cell.empty:
        bany = pd.Series(False, index=by_cell.index)
        for lbl, (col, cmp, thr) in WPC_RULES.items():
            bany = bany | _breach(by_cell, col, cmp, thr)
        breach_cells = by_cell[bany].copy()
        if not breach_cells.empty:
            breach_cells["rca"] = breach_cells.apply(lambda r: auto_rca(r.to_dict()), axis=1)

    r1, r2 = st.columns([1, 1])
    with r1:
        if not breach_cells.empty:
            rc = breach_cells["rca"].value_counts()
            RCA_PAL = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000",
                       "#5B9BD5", "#70AD47", "#DD1E26", "#7c3aed"]
            fig = go.Figure(go.Pie(labels=rc.index, values=rc.values, hole=0.4,
                                   marker=dict(colors=RCA_PAL),
                                   textinfo="label+percent", textfont_size=11))
            fig.update_layout(**CT, height=340, title=_T("RCA Category (auto-classified)"),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#939393")))
            st.plotly_chart(fig, use_container_width=True, key="rca_pie")
        else:
            st.success("✅ No cells breaching WPC thresholds in this period.")
    with r2:
        st.markdown("**RCA → Recommended Action**")
        rca_action = pd.DataFrame({
            "RCA Category": ["Coverage", "Coverage - ISD >2km", "Coverage - End Cell",
                             "Hygiene", "Capacity", "Low Attempt", "Interference"],
            "Action": ["Parameter Tuning", "RET Tuning", "Hygiene Clearance",
                       "Speed Up New Site", "Monitoring / TS On Site", "TS On Site",
                       "Interference Hunting"],
        })
        st.dataframe(rca_action, use_container_width=True, hide_index=True)

    if not breach_cells.empty:
        with st.expander(f"⚠️ {len(breach_cells)} breaching cells — detail + auto-RCA "
                         "(click a cell on the map to drill in)"):
            cols = ["cell_name", "rca"] + [c for c in
                    ["availability", "qos_flow_sr", "rrc_setup_sr", "sdr", "dl_thp",
                     "average_cqi", "qpsk_ratio", "nr_rank2", "dl_prb"]
                    if c in breach_cells.columns]
            st.dataframe(breach_cells[cols].round(2)
                         .sort_values("rca").reset_index(drop=True),
                         use_container_width=True, hide_index=True)

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
    fig.update_layout(**CT, height=height, title=_T(f"{title} [{time_level}]"),
                      xaxis=AX, yaxis=dict(**AX, **({"range": y_range} if y_range else {})))
    return fig

# ── Success Rates ──────────────────────────────────────────────────────────────
st.markdown("### 📈 Success Rate Trends")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(line_chart(trend[x_col], trend["rrc_setup_sr"], PAL[0],
        "rgba(221,30,38,0.07)", "RRC Setup SR (%)", 97, "Target 97%", y_range=[80,101]),
        use_container_width=True, key="lc_rrc")
with c2:
    st.plotly_chart(line_chart(trend[x_col], trend["qos_flow_sr"], PAL[1],
        "rgba(37,99,235,0.07)", "QoS Flow SR (%)", 97, "Target 97%", y_range=[80,101]),
        use_container_width=True, key="lc_qos")

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(line_chart(trend[x_col], trend["sdr"], PAL[4],
        "rgba(219,39,119,0.07)", "SDR (%)", 5, "Threshold 5%", height=240),
        use_container_width=True, key="lc_sdr")
with c4:
    st.plotly_chart(line_chart(trend[x_col], trend["availability"], PAL[2],
        "rgba(22,163,74,0.07)", "Cell Availability (%)", 99, "Target 99%",
        height=240, y_range=[90,101]), use_container_width=True, key="lc_avail")

# ── Throughput + Traffic ───────────────────────────────────────────────────────
st.markdown("### 🚀 Throughput & Traffic")
c5, c6 = st.columns(2)
with c5:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend[x_col], y=trend["dl_thp"], name="DL", marker_color=PAL[0], opacity=0.85))
    fig.add_trace(go.Bar(x=trend[x_col], y=trend["ul_thp"], name="UL", marker_color=PAL[1], opacity=0.85))
    fig.add_hline(y=3, line_dash="dash", line_color="#fb7185",
                  annotation_text="LTC 3 Mbps", annotation_font_color="#fb7185")
    fig.update_layout(**CT, height=260, title=_T(f"DL / UL Throughput (Mbps) [{time_level}]"),
                      barmode="group", xaxis=AX, yaxis=AX,
                      legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(color="#939393")))
    st.plotly_chart(fig, use_container_width=True, key="bar_thp")
with c6:
    st.plotly_chart(line_chart(trend[x_col], trend["traffic_tb"], PAL[3],
        "rgba(245,158,11,0.07)", "Traffic Volume (TB)", height=260), use_container_width=True, key="lc_traffic")

# ── PRB Usage + DL SE ──────────────────────────────────────────────────────────
st.markdown("### 📶 PRB Usage & Spectral Efficiency")
c7, c8 = st.columns(2)
with c7:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["dl_prb"], name="DL PRB",
                             line=dict(color=PAL[0], width=2)))
    fig.add_trace(go.Scatter(x=trend[x_col], y=trend["ul_prb"], name="UL PRB",
                             line=dict(color=PAL[1], width=2, dash="dot")))
    fig.add_hline(y=95, line_dash="dash", line_color="#fb7185",
                  annotation_text="Threshold 95%", annotation_font_color="#fb7185")
    fig.update_layout(**CT, height=240, title=_T(f"DL / UL PRB Usage (%) [{time_level}]"),
                      xaxis=AX, yaxis=dict(**AX, range=[0,100]),
                      legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(color="#939393")))
    st.plotly_chart(fig, use_container_width=True, key="lc_prb")
with c8:
    if "dl_se" in trend.columns and not trend["dl_se"].isna().all():
        st.plotly_chart(line_chart(trend[x_col], trend["dl_se"], PAL[2],
            "rgba(22,163,74,0.07)", "DL Spectral Efficiency (bits/Hz)", height=240),
            use_container_width=True, key="lc_dlse_prb")
    else:
        st.info("⚠️ **dl_se** not available in this dataset.")

# ── WPC Weekly KPIs — CQI · QPSK · Rank 2 · DL SE · UL NI ───────────────────
st.markdown("### 📊 WPC Weekly KPIs")
def safe_line(col, *args, key=None, **kwargs):
    """Draw line chart only when the column actually exists in trend."""
    if col not in trend.columns or trend[col].isna().all():
        st.info(f"⚠️ **{col}** not available — verify counter name in schema.")
        return
    st.plotly_chart(line_chart(trend[x_col], trend[col], *args, **kwargs),
                    use_container_width=True, key=key)

wa1, wa2 = st.columns(2)
with wa1:
    safe_line("average_cqi", PAL[6], "rgba(8,145,178,0.07)",
              "CQI (avg)", target=8, target_label="Threshold 8", height=240)
with wa2:
    safe_line("qpsk_ratio", PAL[4], "rgba(219,39,119,0.07)",
              "QPSK / Last-TTI Ratio (%)", target=60, target_label="Threshold 60%", height=240)

wb1, wb2 = st.columns(2)
with wb1:
    safe_line("nr_rank2", PAL[1], "rgba(37,99,235,0.07)",
              "Rank 2 (%)", target=40, target_label="Target 40%",
              height=240, y_range=[0, 100])
with wb2:
    safe_line("dl_se", PAL[2], "rgba(22,163,74,0.07)",
              "DL Spectral Efficiency (bits/Hz)", height=240)

wc1, wc2 = st.columns(2)
with wc1:
    if "ul_ni_avg" in trend.columns and not trend["ul_ni_avg"].isna().all():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend[x_col], y=trend["ul_ni_avg"],
                                 line=dict(color=PAL[5], width=2.5),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,0.07)",
                                 showlegend=False))
        fig.add_hline(y=-100, line_dash="dash", line_color="#fb7185",
                      annotation_text="Threshold −100 dBm", annotation_font_color="#fb7185")
        fig.update_layout(**CT, height=240, title=_T(f"N.UL.NI.Avg / RSSI (dBm) [{time_level}]"),
                          xaxis=AX, yaxis=AX)
        st.plotly_chart(fig, use_container_width=True, key="lc_ulni")
    else:
        st.info("⚠️ **RSSI / UL NI** not in raw_dashboard_kpi schema — add the column to enable.")
with wc2:
    safe_line("nr_rank4", PAL[3], "rgba(245,158,11,0.07)",
              "Rank 4 (%)", height=240, y_range=[0, 100])


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
            [[0, "#3BC1A8"], [1, "#005461"]], "{:.2f}%", key="bar_sdr"))
    with cr2:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "rrc_setup_sr", largest=False), "rrc_setup_sr",
            "🔴 Lowest RRC Setup SR — Worst Success Rate",
            [[0, "#005461"], [1, "#3BC1A8"]], "{:.1f}%", key="bar_rrc", x_range=[0, 105]))

    cr3, cr4 = st.columns(2)
    with cr3:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "dl_thp", largest=False), "dl_thp",
            "🔴 Lowest DL Throughput — LTC (<3 Mbps)",
            [[0, "#005461"], [1, "#3BC1A8"]], "{:.1f}", key="bar_dlthp"))
    with cr4:
        handle_click(clickable_hbar(
            rank_or_single(by_cell, "dl_prb", largest=True), "dl_prb",
            "🔴 Highest DL PRB — Most Congested (>95%)",
            [[0, "#3BC1A8"], [1, "#005461"]], "{:.1f}%", key="bar_dlprb", x_range=[0, 105]))

    # WPC Weekly extra rankings
    if "average_cqi" in by_cell.columns and by_cell["average_cqi"].notna().any():
        cr5, cr6 = st.columns(2)
        with cr5:
            handle_click(clickable_hbar(
                rank_or_single(by_cell, "average_cqi", largest=False), "average_cqi",
                "🔴 Lowest CQI — Poor Channel Quality (<8)",
                [[0, "#005461"], [1, "#3BC1A8"]], "{:.2f}", key="bar_cqi"))
        with cr6:
            if "qpsk_ratio" in by_cell.columns and by_cell["qpsk_ratio"].notna().any():
                handle_click(clickable_hbar(
                    rank_or_single(by_cell, "qpsk_ratio", largest=True), "qpsk_ratio",
                    "🔴 Highest QPSK Ratio — Poor Modulation (>60%)",
                    [[0, "#3BC1A8"], [1, "#005461"]], "{:.1f}%", key="bar_qpsk", x_range=[0, 105]))

    if "nr_rank2" in by_cell.columns and by_cell["nr_rank2"].notna().any():
        cr7, cr8 = st.columns(2)
        with cr7:
            handle_click(clickable_hbar(
                rank_or_single(by_cell, "nr_rank2", largest=False), "nr_rank2",
                "🔴 Lowest Rank 2 — Poor MIMO Usage (<40%)",
                [[0, "#005461"], [1, "#3BC1A8"]], "{:.1f}%", key="bar_rank2", x_range=[0, 100]))
        with cr8:
            if "qos_flow_sr" in by_cell.columns:
                handle_click(clickable_hbar(
                    rank_or_single(by_cell, "qos_flow_sr", largest=False), "qos_flow_sr",
                    "🔴 Lowest QoS Flow SR — Worst Setup (<97%)",
                    [[0, "#005461"], [1, "#3BC1A8"]], "{:.1f}%", key="bar_qos", x_range=[0, 105]))

# ── Raw data ───────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Raw KPI Data" + (f"  —  filtered to {active_cell}" if active_cell else "")):
    disp = df_view.copy()
    for col in ["rrc_setup_sr","qos_flow_sr","sdr","dl_prb","ul_prb",
                "availability","nr_rank4","average_cqi","qpsk_ratio","nr_rank2","dl_se"]:
        if col in disp.columns: disp[col] = disp[col].round(2)
    for col in ["dl_thp","ul_thp","ul_ni_avg"]:
        if col in disp.columns: disp[col] = disp[col].round(1)
    sort_cols = [c for c in ["date","time","cell_name","nename"] if c in disp.columns]
    st.dataframe(disp.sort_values(sort_cols, ascending=[False]*len(sort_cols))
                    .reset_index(drop=True), use_container_width=True)

st.markdown("---")

# ── Admin Panel (admin role only) ─────────────────────────────────────────────
if _is_admin:
    with st.expander("🛡️ Admin Panel — User Management", expanded=False):
        auth.render_admin_panel(_AUTH_CFG)

st.markdown("---")
st.markdown("<div style='text-align:center;font-family:Space Mono,monospace;font-size:11px;"
            "color:#f0c4c4;letter-spacing:1px'>5G FWA KPI Analysis Dashboard · surge_data</div>",
            unsafe_allow_html=True)
