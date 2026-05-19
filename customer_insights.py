import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
from itertools import combinations

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Insights Explorer",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark matplotlib theme
plt.rcParams.update({
    'figure.facecolor':  '#151720',
    'axes.facecolor':    '#151720',
    'axes.edgecolor':    '#2e3246',
    'axes.labelcolor':   '#d4cfc7',
    'xtick.color':       '#d4cfc7',
    'ytick.color':       '#d4cfc7',
    'text.color':        '#d4cfc7',
    'grid.color':        '#2e3246',
    'figure.edgecolor':  '#151720',
})

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0ece3 !important;
}

/* All general text light */
p, span, div, label {
    color: #d4cfc7;
}

[data-testid="stSidebar"] {
    background: #0a0c12;
    border-right: 1px solid #1e2130;
}

[data-testid="stSidebar"] * {
    color: #e8e8e8 !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-card {
    background: #1c1f2b;
    border: 1px solid #2e3246;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}

.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7a8099;
    margin-bottom: 0.25rem;
}

.metric-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #f0ece3;
    line-height: 1;
}

.metric-card .sub {
    font-size: 0.78rem;
    color: #7a8099;
    margin-top: 0.2rem;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f0ece3 !important;
    border-bottom: 2px solid #e8c97e;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
    margin-top: 1.5rem;
}

/* All markdown text light */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] * {
    color: #d4cfc7 !important;
}

/* Tab labels */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] * {
    color: #d4cfc7 !important;
}

/* All widget labels */
.stRadio label, .stSlider label, .stSelectbox label,
.stMultiSelect label, [data-testid="stWidgetLabel"] * {
    color: #d4cfc7 !important;
}

/* Widget option/select text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] * {
    color: #d4cfc7 !important;
}

.tag {
    display: inline-block;
    background: #2a2310;
    color: #e8c97e;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}

.upload-box {
    background: #1c1f2b;
    border: 2px dashed #2e3246;
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
}

.upload-box p {
    color: #f0ece3 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2e3246;
    border-radius: 8px;
}

.stButton > button {
    background: #e8c97e;
    color: #0f1117 !important;
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}

.stButton > button p {
    color: #0f1117 !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: #f0d898;
    color: #0f1117 !important;
}

.stButton > button:hover p {
    color: #0f1117 !important;
}

/* Primary button */
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {
    background: #e8c97e;
    color: #0f1117 !important;
    font-weight: 700;
}

.stButton > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] p {
    color: #0f1117 !important;
    font-weight: 700 !important;
}

.stButton > button[kind="primary"]:hover {
    background: #f0d898;
    color: #0f1117 !important;
}

.stButton > button[kind="primary"]:hover p {
    color: #0f1117 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
PALETTE = ["#c9a84c", "#3d5a80", "#98c1d9", "#e07a5f", "#3d405b",
           "#81b29a", "#f2cc8f", "#6d6875", "#b5838d", "#e5989b"]

def fmt_currency(v):
    if v >= 1_000_000:
        return f"€{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"€{v/1_000:.1f}K"
    return f"€{v:.0f}"

def euro_axis_formatter(x, _):
    if x >= 1_000_000:
        return f"€{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"€{x/1_000:.0f}K"
    return f"€{x:.0f}"

def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {'<div class="sub">' + sub + '</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes):
    df = pd.read_csv(file_bytes) if file_bytes.name.endswith(".csv") else pd.read_excel(file_bytes)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Always treat IDs as strings
    for id_col in ['CustomerId', 'InvoiceId', 'ProductId']:
        if id_col in df.columns:
            df[id_col] = df[id_col].astype(str)

    # Parse dates
    for col in ['CreatedDate', 'OrderDate', 'Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.rename(columns={col: 'CreatedDate'})
            break

    # Revenue
    if 'LineRevenue' not in df.columns:
        df['LineRevenue'] = df['PricePerUnit'] * df['Quantity']

    return df

@st.cache_data(show_spinner=False)
def compute_base(df):
    snap = df['CreatedDate'].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby('CustomerId')
        .agg(
            Recency   =('CreatedDate',  lambda x: (snap - x.max()).days),
            Frequency =('InvoiceId',    'nunique'),
            Monetary  =('LineRevenue',  'sum'),
        )
        .reset_index()
    )

    prod_repeat = (
        df.groupby(['CustomerId','ProductId'])
        .agg(
            OrderCount    =('InvoiceId',   'nunique'),
            TotalQuantity =('Quantity',    'sum'),
            TotalSpend    =('LineRevenue', 'sum'),
        )
        .reset_index()
        .query('OrderCount > 1')
        .sort_values(['CustomerId','OrderCount'], ascending=[True, False])
    )

    return rfm, prod_repeat

@st.cache_data(show_spinner=False)
def compute_group_matrices(df, group_col):
    spend = (
        df.groupby(['CustomerId', group_col])['LineRevenue']
        .sum().unstack(fill_value=0)
    )
    share = spend.div(spend.sum(axis=1), axis=0)
    binary = (spend > 0).astype(int)
    sim_matrix = cosine_similarity(binary)
    sim_df = pd.DataFrame(sim_matrix, index=binary.index, columns=binary.index)
    return spend, share, binary, sim_df

@st.cache_data(show_spinner=False)
def run_kvi_classification(order_lines, kvi_score_threshold=2.0, core_percentile=75):
    df = order_lines.copy()

    quantity = df.groupby('ProductId')['Quantity'].sum().reset_index()

    df['_weighted_price'] = df['PricePerUnit'] * df['Quantity']
    price = (
        df.groupby('ProductId')
        .agg(_WeightedSum=('_weighted_price', 'sum'), _TotalQty=('Quantity', 'sum'))
        .reset_index()
    )
    price['Price'] = (price['_WeightedSum'] / price['_TotalQty']).round(2)
    price = price[['ProductId', 'Price']]
    df = df.drop(columns=['_weighted_price'])

    customers = (
        df.drop_duplicates(subset=['ProductId', 'CustomerId'])
        .groupby('ProductId')['CustomerId'].nunique()
        .reset_index()
        .rename(columns={'CustomerId': 'UniqueCustomers'})
    )

    purchase_count = df['ProductId'].value_counts().reset_index()
    purchase_count.columns = ['ProductId', 'PurchaseCount']

    dfAll = quantity.merge(price, on='ProductId', how='left')
    dfAll = dfAll.merge(customers, on='ProductId', how='left')
    dfAll = dfAll.merge(purchase_count, on='ProductId', how='left')
    dfAll['Revenue'] = dfAll['Quantity'] * dfAll['Price']

    if len(dfAll) == 0:
        st.warning("No products found for this customer group. Try a different selection.")
        st.stop()

    scaler = StandardScaler()

    dfAll['Demand_Proportion'] = dfAll['Quantity'] / dfAll['Quantity'].sum()
    dfAll['Demand_Proportion_Scaled'] = scaler.fit_transform(dfAll[['Demand_Proportion']])

    dfAll['Revenue_Proportion'] = dfAll['Revenue'] / dfAll['Revenue'].sum()
    dfAll['Revenue_Proportion_Scaled'] = scaler.fit_transform(dfAll[['Revenue_Proportion']])

    n_customers_total = df['CustomerId'].nunique()
    dfAll['UniqueCustomers_Proportion'] = dfAll['UniqueCustomers'] / n_customers_total
    dfAll['UniqueCustomers_Proportion_Scaled'] = scaler.fit_transform(dfAll[['UniqueCustomers_Proportion']])

    product_ids = dfAll['ProductId'].unique()
    product_to_idx = {p: i for i, p in enumerate(product_ids)}
    n_products = len(product_ids)

    filtered = df[df['ProductId'].isin(product_ids)]
    grouped = filtered.groupby('InvoiceId')['ProductId'].apply(list)

    rows, cols, data = [], [], []
    for products in grouped:
        for i in range(len(products)):
            for j in range(len(products)):
                rows.append(product_to_idx[products[i]])
                cols.append(product_to_idx[products[j]])
                data.append(1)

    Q = coo_matrix((data, (rows, cols)), shape=(n_products, n_products)).toarray()
    Q_j = Q.sum(axis=1)
    Q_j[Q_j == 0] = 1
    score = (Q / Q_j).sum(axis=1) / n_products
    product_score_map = {p: score[i] for i, p in enumerate(product_ids)}

    dfAll['Corr_Score'] = dfAll['ProductId'].map(product_score_map)
    dfAll['Corr_Score_Scaled'] = StandardScaler().fit_transform(dfAll[['Corr_Score']])

    dfAll['KVI_Score'] = (
        dfAll['Demand_Proportion_Scaled'] * 0.45 +
        dfAll['Revenue_Proportion_Scaled'] * 0.35 +
        dfAll['UniqueCustomers_Proportion_Scaled'] * 0.05 +
        dfAll['Corr_Score_Scaled'] * 0.15
    )

    dfAll = dfAll.sort_values('KVI_Score', ascending=False).reset_index(drop=True)

    last_kvi_idx = dfAll[dfAll['KVI_Score'] >= kvi_score_threshold].last_valid_index()
    dfAll['KVI'] = 0
    if last_kvi_idx is not None:
        dfAll.loc[:last_kvi_idx, 'KVI'] = 1

    non_kvi = dfAll.loc[dfAll['KVI'] == 0, 'PurchaseCount']
    threshold_pc = np.percentile(non_kvi, core_percentile) if len(non_kvi) > 0 else 0
    dfAll['Core'] = 0
    dfAll['Slow_Mover'] = 0
    dfAll.loc[(dfAll['KVI'] == 0) & (dfAll['PurchaseCount'] >= threshold_pc), 'Core'] = 1
    dfAll.loc[(dfAll['KVI'] == 0) & (dfAll['Core'] == 0), 'Slow_Mover'] = 1

    dfAll['Category'] = np.select(
        [dfAll['KVI'] == 1, dfAll['Core'] == 1],
        ['KVI', 'Core'],
        default='Slow Mover'
    )

    dfAll['ProductId'] = dfAll['ProductId'].astype(str)
    return dfAll

# ── App ────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0'>Customer Insights Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9a8f85;margin-top:0.2rem;margin-bottom:2rem'>Upload your order lines data to begin exploring</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded is None:
    st.markdown("""
    <div class="upload-box">
        <p style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#1a1a1a;margin-bottom:0.5rem">
            Drop your OrderLines file above
        </p>
        <p style="color:#9a8f85;font-size:0.85rem">
            Supports CSV and Excel · Expected columns: CustomerId, InvoiceId, ProductId,
            MainGroup, SubGroup, PricePerUnit, Quantity, CreatedDate
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df = load_and_prepare(uploaded)

with st.spinner("Computing metrics…"):
    rfm, prod_repeat = compute_base(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Explorer Controls")
    st.markdown("---")

    analysis = st.selectbox("Analysis view", [
        "Overview",
        "Category Breakdown",
        "Repeat Purchases",
        "Basket Analysis",
        "Customer Specialty",
        "KVI Classification",
        "Pricing Simulation",
    ])

    st.markdown("---")
    st.markdown("**Filters**")

    # Date filter
    if 'CreatedDate' in df.columns:
        min_date = df['CreatedDate'].min().date()
        max_date = df['CreatedDate'].max().date()
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    # Customer filter
    all_customers = sorted(df['CustomerId'].unique().tolist())
    selected_customers = st.multiselect("Customers", all_customers, placeholder="All customers")

    # Detect all low-cardinality categorical columns for filtering
    id_cols = {'CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate'}
    filter_cat_cols = [
        c for c in df.columns
        if c not in id_cols
        and 1 < df[c].nunique() < 200
        and (
            str(df[c].dtype) in ('object', 'category', 'str', 'string')
            or (str(df[c].dtype).startswith('str') )
            or (df[c].dtype in ['int64', 'float64', 'Int64'] and df[c].nunique() < 50)
        )
    ]

    cat_filters = {}
    for col in filter_cat_cols:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        selected = st.multiselect(col, vals, placeholder=f"All {col}", key=f"filter_{col}")
        cat_filters[col] = selected

    with st.expander("Debug: column detection", expanded=False):
        debug_rows = []
        for c in df.columns:
            n_unique = df[c].nunique()
            dtype = str(df[c].dtype)
            included = c in filter_cat_cols
            reason = ""
            if c in id_cols:
                reason = "ID column"
            elif n_unique <= 1:
                reason = f"only {n_unique} unique value"
            elif n_unique >= 200:
                reason = f"{n_unique} unique values (≥200)"
            elif dtype not in ('object', 'category') and not (df[c].dtype in ['int64', 'float64', 'Int64'] and n_unique < 50):
                reason = f"dtype {dtype} not categorical"
            else:
                reason = "included"
            debug_rows.append({'Column': c, 'dtype': dtype, 'Unique': n_unique, 'Included': included, 'Reason': reason})
        st.dataframe(pd.DataFrame(debug_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"<span style='font-size:0.75rem;color:#888'>{len(df):,} rows · {df['CustomerId'].nunique():,} customers</span>", unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
fdf = df.copy()

if date_range and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    fdf = fdf[(fdf['CreatedDate'] >= start) & (fdf['CreatedDate'] <= end)]

if selected_customers:
    fdf = fdf[fdf['CustomerId'].isin(selected_customers)]

for col, vals in cat_filters.items():
    if vals:
        fdf = fdf[fdf[col].astype(str).isin(vals)]

# Recompute base metrics on filtered data
rfm, prod_repeat = compute_base(fdf)

# ── Merge confirmed specialty labels into fdf if available ─────────────────
if 'confirmed_specialty' in st.session_state:
    _labels = st.session_state['confirmed_specialty']['labels'].rename(
        columns={'Specialty': 'Customer Cluster'}
    ).copy()
    _labels['CustomerId'] = _labels['CustomerId'].astype(str)
    fdf = fdf.merge(_labels, on='CustomerId', how='left')
    fdf['Customer Cluster'] = fdf['Customer Cluster'].fillna('Unassigned')

# Helper: detect categorical columns available for grouping
def get_cat_cols(df):
    id_cols = {'CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate'}
    return [
        c for c in df.columns
        if c not in id_cols
        and 1 < df[c].nunique() < 200
        and (
            str(df[c].dtype) in ('object', 'category', 'str', 'string')
            or str(df[c].dtype).startswith('str')
            or (df[c].dtype in ['int64', 'float64', 'Int64'] and df[c].nunique() < 50)
        )
    ]

cat_cols = get_cat_cols(fdf)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if analysis == "Overview":
    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Revenue", fmt_currency(fdf['LineRevenue'].sum()))
    with c2: metric_card("Customers", f"{fdf['CustomerId'].nunique():,}")
    with c3: metric_card("Orders", f"{fdf['InvoiceId'].nunique():,}")
    with c4: metric_card("Avg Order Value", fmt_currency(fdf.groupby('InvoiceId')['LineRevenue'].sum().mean()))

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        group_col_ov = st.selectbox(
            "Group revenue by", cat_cols,
            index=cat_cols.index('MainGroup') if 'MainGroup' in cat_cols else 0,
            key="ov_group"
        )
        grp_rev = fdf.groupby(group_col_ov)['LineRevenue'].sum().sort_values(ascending=True)
        st.markdown(f'<div class="section-header" style="font-size:1.1rem">Revenue by {group_col_ov}</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, max(3, len(grp_rev)*0.35)))
        bars = ax.barh(grp_rev.index.astype(str), grp_rev.values, color=PALETTE[0], alpha=0.85)
        ax.set_xlabel("Revenue (€)", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
        ax.tick_params(labelsize=8)
        ax.spines[['top','right','left']].set_visible(False)
        for bar, val in zip(bars, grp_rev.values):
            ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                    fmt_currency(val), va='center', fontsize=7.5, color='#d4cfc7')
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<div class="section-header" style="font-size:1.1rem">Revenue over Time</div>', unsafe_allow_html=True)
        time_rev = fdf.set_index('CreatedDate').resample('ME')['LineRevenue'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.fill_between(time_rev['CreatedDate'], time_rev['LineRevenue'], alpha=0.15, color=PALETTE[0])
        ax.plot(time_rev['CreatedDate'], time_rev['LineRevenue'], color=PALETTE[0], linewidth=2)
        ax.set_ylabel("Revenue (€)", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header" style="font-size:1.1rem">Customer Value Distribution</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(rfm['Monetary'], bins=40, color=PALETTE[1], alpha=0.8, edgecolor='white')
        ax.set_xlabel("Total Spend per Customer (€)", fontsize=9)
        ax.set_ylabel("# Customers", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col_d:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(rfm['Frequency'], bins=30, color=PALETTE[2], alpha=0.8, edgecolor='white')
        ax.set_xlabel("Order Frequency per Customer", fontsize=9)
        ax.set_ylabel("# Customers", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — CATEGORY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Category Breakdown":
    st.markdown('<div class="section-header">Category Breakdown</div>', unsafe_allow_html=True)

    col_ctrl, col_filter = st.columns([1, 2])
    with col_ctrl:
        group_col = st.selectbox(
            "Group by", cat_cols,
            index=cat_cols.index('MainGroup') if 'MainGroup' in cat_cols else 0,
            key="cb_group"
        )

    # Optional secondary filter on the chosen column
    all_vals = sorted(fdf[group_col].dropna().astype(str).unique().tolist())
    with col_filter:
        filter_vals = st.multiselect(
            f"Filter {group_col} values", all_vals, placeholder="All values", key="cb_filter"
        )
    cb_df = fdf if not filter_vals else fdf[fdf[group_col].astype(str).isin(filter_vals)]

    tab1, tab2 = st.tabs(["Summary", "Customer Deep-Dive"])

    with tab1:
        grp_summary = (
            cb_df.groupby(group_col)
            .agg(
                Revenue       =('LineRevenue', 'sum'),
                Customers     =('CustomerId',  'nunique'),
                Orders        =('InvoiceId',   'nunique'),
                AvgOrderValue =('LineRevenue', 'mean'),
            )
            .sort_values('Revenue', ascending=False)
            .reset_index()
        )
        grp_summary['RevenueShare'] = (grp_summary['Revenue'] / grp_summary['Revenue'].sum()).map('{:.1%}'.format)
        grp_summary['AvgOrderValue'] = grp_summary['AvgOrderValue'].map(lambda x: f"€{x:,.2f}")
        grp_summary['Revenue']       = grp_summary['Revenue'].map(fmt_currency)
        st.dataframe(grp_summary, use_container_width=True, hide_index=True)

    with tab2:
        cust_id = st.selectbox("Select customer", sorted(fdf['CustomerId'].unique().tolist()), key="cb_cust")
        cust_df = fdf[fdf['CustomerId'] == cust_id]

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Total Spend", fmt_currency(cust_df['LineRevenue'].sum()))
        with c2: metric_card("Orders", str(cust_df['InvoiceId'].nunique()))
        with c3: metric_card("Products Bought", str(cust_df['ProductId'].nunique()))

        st.markdown(f"**Spend & share by {group_col}**")
        cust_grp = cust_df.groupby(group_col)['LineRevenue'].sum().sort_values(ascending=False)
        cust_grp_share = cust_grp / cust_grp.sum()
        cust_grp_df = pd.DataFrame({
            group_col: cust_grp.index,
            'Spend': cust_grp.map(fmt_currency).values,
            'Share': cust_grp_share.map('{:.1%}'.format).values,
        })
        st.dataframe(cust_grp_df[cust_grp_df['Spend'] != '€0'], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 4 — REPEAT PURCHASES
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Repeat Purchases":
    st.markdown('<div class="section-header">Repeat Purchases</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["By Product", "By Category", "Customer Profile"])

    with tab1:
        min_orders = st.slider("Min order count to be 'repeat'", 2, 100, 2)
        top_repeat = (
            prod_repeat[prod_repeat['OrderCount'] >= min_orders]
            .groupby('ProductId')
            .agg(
                RepeatCustomers =('CustomerId',   'nunique'),
                AvgOrderCount   =('OrderCount',   'mean'),
                TotalSpend      =('TotalSpend',   'sum'),
                TotalQty        =('TotalQuantity','sum'),
            )
            .sort_values('RepeatCustomers', ascending=False)
            .reset_index()
        )
        top_repeat['TotalSpend']    = top_repeat['TotalSpend'].map(fmt_currency)
        top_repeat['AvgOrderCount'] = top_repeat['AvgOrderCount'].map('{:.1f}'.format)
        st.dataframe(top_repeat, use_container_width=True, hide_index=True)

    with tab2:
        rp_col = st.selectbox(
            "Group repeat rate by", cat_cols,
            index=cat_cols.index('MainGroup') if 'MainGroup' in cat_cols else 0,
            key="rp_col"
        )
        order_counts_df = (
            fdf.groupby(['CustomerId', 'ProductId', rp_col])['InvoiceId']
            .nunique().reset_index().rename(columns={'InvoiceId': 'OrderCount'})
        )
        order_counts_df['IsRepeat'] = order_counts_df['OrderCount'] > 1

        repeat_grp = (
            order_counts_df.groupby(rp_col)
            .agg(
                TotalCustomerProducts=('ProductId',  'count'),
                RepeatCount          =('IsRepeat',   'sum'),
                AvgOrderCount        =('OrderCount', 'mean'),
            )
            .assign(RepeatRate=lambda x: x['RepeatCount'] / x['TotalCustomerProducts'])
            .sort_values('RepeatRate', ascending=False)
            .reset_index()
        )
        repeat_grp['RepeatRate']    = repeat_grp['RepeatRate'].map('{:.1%}'.format)
        repeat_grp['AvgOrderCount'] = repeat_grp['AvgOrderCount'].map('{:.1f}'.format)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Repeat rate by {rp_col}**")
            st.dataframe(repeat_grp, use_container_width=True, hide_index=True)
        with col2:
            repeat_grp2 = (
                order_counts_df.groupby(rp_col)
                .agg(RepeatCount=('IsRepeat', 'sum'), Total=('ProductId', 'count'))
                .assign(RepeatRate=lambda x: x['RepeatCount'] / x['Total'])
                .sort_values('RepeatRate', ascending=False)
            )
            fig, ax = plt.subplots(figsize=(5, max(3, len(repeat_grp2)*0.38)))
            ax.barh(repeat_grp2.index.astype(str), repeat_grp2['RepeatRate'],
                    color=PALETTE[3], alpha=0.8)
            ax.set_xlabel("Repeat Rate", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

    with tab3:
        cust_id = st.selectbox("Select customer", sorted(fdf['CustomerId'].unique().tolist()), key="rp_cust")
        cust_repeats = prod_repeat[prod_repeat['CustomerId'] == cust_id].copy()

        if cust_repeats.empty:
            st.info("This customer has no repeat-purchased products.")
        else:
            c1, c2 = st.columns(2)
            with c1: metric_card("Repeat Products", str(len(cust_repeats)))
            with c2: metric_card("Repeat Spend", fmt_currency(cust_repeats['TotalSpend'].sum()))

            cust_repeats = cust_repeats.sort_values('OrderCount', ascending=False)
            cust_repeats['TotalSpend'] = cust_repeats['TotalSpend'].map(fmt_currency)
            st.dataframe(cust_repeats[['ProductId','OrderCount','TotalQuantity','TotalSpend']],
                         use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 5 — BASKET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Basket Analysis":
    st.markdown('<div class="section-header">Basket Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a8099;margin-top:-0.8rem;margin-bottom:1.2rem;font-size:0.85rem'>"
        "What products belong together — by specialty or by anchor product — and which customers are missing items from their expected basket."
        "</p>", unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Specialty Basket", "Anchor Product Basket", "Basket Segmentation", "Basket Exploration"])

    # ── Shared setup ───────────────────────────────────────────────────────────
    # Detect specialty column (reuse logic from specialty view)
    cat_cols_basket = [
        c for c in fdf.columns
        if str(fdf[c].dtype) in ('object', 'category')
        and c not in ['CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate']
        and fdf[c].nunique() < 200
    ]

    # ── TAB 1: SPECIALTY BASKET ────────────────────────────────────────────────
    with tab1:
        st.markdown("**Build the typical basket for a customer specialty, then find who's missing items.**")

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            spec_col = st.selectbox(
                "Specialty column",
                cat_cols_basket,
                index=cat_cols_basket.index('MainGroup') if 'MainGroup' in cat_cols_basket else 0,
                key="basket_spec_col"
            )
        with col_ctrl2:
            spec_threshold = st.slider(
                "Specialist threshold", 10, 80, 40, 5,
                format="%d%%", key="basket_thresh",
                help="Min spend share to be called a specialist"
            ) / 100
        with col_ctrl3:
            basket_min_rate = st.slider(
                "Min customer % to include in basket", 5, 80, 20, 5,
                format="%d%%", key="basket_rate",
                help="A product is part of the 'expected basket' if this % of specialty customers buy it"
            ) / 100

        # Assign specialty labels
        sb_spend = (
            fdf.groupby(['CustomerId', spec_col])['LineRevenue']
            .sum().unstack(fill_value=0)
        )
        sb_share = sb_spend.div(sb_spend.sum(axis=1), axis=0)
        spec_labels = pd.DataFrame({
            'Specialty':      sb_share.idxmax(axis=1),
            'SpecialtyShare': sb_share.max(axis=1),
        }).reset_index()
        spec_labels['Specialty'] = spec_labels.apply(
            lambda r: r['Specialty'] if r['SpecialtyShare'] >= spec_threshold else 'Generalist',
            axis=1
        )

        available_specs = sorted([s for s in spec_labels['Specialty'].unique() if s != 'Generalist'])
        if not available_specs:
            st.warning("No specialists found — try lowering the threshold.")
            st.stop()

        selected_spec = st.selectbox("Select specialty to analyse", available_specs, key="spec_basket_sel")

        spec_customer_ids = spec_labels[spec_labels['Specialty'] == selected_spec]['CustomerId'].tolist()
        spec_invoices = fdf[fdf['CustomerId'].isin(spec_customer_ids)]['InvoiceId'].unique()
        n_spec_customers = len(spec_customer_ids)

        # How often each product appears across this specialty's invoices
        prod_freq = (
            fdf[fdf['InvoiceId'].isin(spec_invoices)]
            .groupby('ProductId')['InvoiceId'].nunique()
            .reset_index()
            .rename(columns={'InvoiceId': 'InvoiceCount'})
        )
        prod_freq['CustomerRate'] = prod_freq['InvoiceCount'] / fdf[
            fdf['CustomerId'].isin(spec_customer_ids)
        ]['InvoiceId'].nunique()

        expected_basket = prod_freq[prod_freq['CustomerRate'] >= basket_min_rate].sort_values(
            'CustomerRate', ascending=False
        )

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Customers in specialty", str(n_spec_customers))
        with c2: metric_card("Basket size", str(len(expected_basket)))
        with c3: metric_card(
            "Avg basket coverage",
            f"{basket_min_rate:.0%}+ purchase rate"
        )

        st.markdown(f"**Expected basket for `{selected_spec}` specialists**")
        st.caption(f"Products bought by at least {basket_min_rate:.0%} of {selected_spec} customers")

        col_b1, col_b2 = st.columns([1, 1.4])
        with col_b1:
            disp_basket = expected_basket.copy()
            disp_basket['CustomerRate'] = disp_basket['CustomerRate'].map('{:.1%}'.format)
            st.dataframe(disp_basket, use_container_width=True, hide_index=True)

        with col_b2:
            top_basket = expected_basket.head(20)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top_basket) * 0.38)))
            ax.barh(top_basket['ProductId'].astype(str)[::-1],
                    top_basket['CustomerRate'].values[::-1],
                    color=PALETTE[0], alpha=0.85)
            ax.set_xlabel("% of specialty customers buying this product", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        # ── Wallet leakage: who's missing basket items? ────────────────────────
        st.markdown("---")
        st.markdown("**Wallet leakage — customers missing expected basket items**")
        st.caption("For each specialist customer, how many of the expected basket products have they never bought?")

        basket_products = expected_basket['ProductId'].tolist()

        # Per customer: which basket products have they bought at all?
        cust_basket_bought = (
            fdf[
                fdf['CustomerId'].isin(spec_customer_ids) &
                fdf['ProductId'].isin(basket_products)
            ]
            .groupby(['CustomerId', 'ProductId'])['InvoiceId']
            .nunique()
            .unstack(fill_value=0)
        )
        # Reindex to ensure all basket products are columns
        cust_basket_bought = cust_basket_bought.reindex(
            columns=basket_products, fill_value=0
        )
        cust_basket_binary = (cust_basket_bought > 0).astype(int)

        leakage = pd.DataFrame({
            'CustomerId':       cust_basket_binary.index,
            'BasketItemsBought': cust_basket_binary.sum(axis=1).values,
            'BasketSize':        len(basket_products),
        })
        leakage['ItemsMissing']   = leakage['BasketSize'] - leakage['BasketItemsBought']
        leakage['CoverageRate']   = leakage['BasketItemsBought'] / leakage['BasketSize']
        leakage['MissingProducts'] = cust_basket_binary.apply(
            lambda row: ', '.join([str(p) for p in basket_products if row[p] == 0]), axis=1
        )

        leakage = leakage.merge(
            fdf.groupby('CustomerId')['LineRevenue'].sum().reset_index().rename(
                columns={'LineRevenue': 'TotalSpend'}
            ), on='CustomerId', how='left'
        ).sort_values('ItemsMissing', ascending=False)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Customers with full basket", str((leakage['ItemsMissing'] == 0).sum()))
        with c2: metric_card("Customers missing 1+ items", str((leakage['ItemsMissing'] > 0).sum()))
        with c3: metric_card("Avg coverage", f"{leakage['CoverageRate'].mean():.1%}")

        leakage_display = leakage.copy()
        leakage_display['TotalSpend']   = leakage_display['TotalSpend'].map(fmt_currency)
        leakage_display['CoverageRate'] = leakage_display['CoverageRate'].map('{:.1%}'.format)
        st.dataframe(
            leakage_display[['CustomerId', 'TotalSpend', 'BasketItemsBought',
                              'ItemsMissing', 'CoverageRate', 'MissingProducts']],
            use_container_width=True, hide_index=True
        )

        # Individual customer leakage drill-down
        st.markdown("---")
        st.markdown("**Drill into a specific customer's basket gap**")
        leakage_cust = st.selectbox(
            "Select customer", leakage.sort_values('ItemsMissing', ascending=False)['CustomerId'].tolist(),
            key="leakage_cust"
        )
        lrow_match = leakage[leakage['CustomerId'] == leakage_cust]
        if lrow_match.empty:
            st.info("No leakage data available for this customer.")
        else:
            lrow = lrow_match.iloc[0]
            missing = [p for p in basket_products if p not in
                       cust_basket_binary.columns[cust_basket_binary.loc[leakage_cust] == 1].tolist()]

            col_l1, col_l2 = st.columns(2)
            with col_l1:
                metric_card("Basket coverage", f"{lrow['CoverageRate']:.1%}")
            with col_l2:
                metric_card("Items missing", str(int(lrow['ItemsMissing'])))

            if missing:
                st.markdown("**Products this customer has never bought (cross-sell opportunities)**")
                miss_df = expected_basket[expected_basket['ProductId'].isin(missing)].copy()
                miss_df['CustomerRate'] = miss_df['CustomerRate'].map('{:.1%}'.format)
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            else:
                st.success("This customer buys everything in the expected basket.")

    # ── TAB 2: ANCHOR PRODUCT BASKET ──────────────────────────────────────────
    with tab2:
        st.markdown("**Pick an anchor product — see what else customers buy alongside it, then check who's missing those items.**")

        top_n_anchor = st.sidebar.slider("Max products to analyse", 20, 200, 50,
                                          help="Limits product universe", key="anchor_top_n")
        top_prods = (
            fdf.groupby('ProductId')['InvoiceId'].nunique()
            .sort_values(ascending=False)
            .head(top_n_anchor)
            .index.tolist()
        )

        inv_prod = (
            fdf[fdf['ProductId'].isin(top_prods)]
            .groupby(['InvoiceId', 'ProductId'])['Quantity']
            .sum()
            .unstack(fill_value=0)
        )
        inv_prod_binary = (inv_prod > 0).astype(int)

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            anchor = st.selectbox("Select anchor product", sorted(top_prods), key="anchor_prod")
        with col_a2:
            anchor_min_rate = st.slider(
                "Min co-purchase rate for basket", 5, 80, 15, 5,
                format="%d%%", key="anchor_rate",
                help="Products bought alongside the anchor in at least this % of invoices"
            ) / 100

        if anchor in inv_prod_binary.columns:
            invoices_with_anchor = inv_prod_binary[inv_prod_binary[anchor] == 1]
            n_anchor_invoices = len(invoices_with_anchor)

            # Co-purchase rates for all other products
            co = (
                invoices_with_anchor.drop(columns=[anchor])
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            co.columns = ['ProductId', 'CoInvoiceCount']
            co['CoRate'] = co['CoInvoiceCount'] / n_anchor_invoices
            anchor_basket = co[co['CoRate'] >= anchor_min_rate]

            # Customers who bought the anchor
            anchor_invoice_ids = invoices_with_anchor.index.tolist()
            anchor_customers = (
                fdf[fdf['InvoiceId'].isin(anchor_invoice_ids)]['CustomerId']
                .unique().tolist()
            )

            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Invoices with anchor", str(n_anchor_invoices))
            with c2: metric_card("Customers buying anchor", str(len(anchor_customers)))
            with c3: metric_card("Basket size", str(len(anchor_basket)))

            col_b1, col_b2 = st.columns([1, 1.4])
            with col_b1:
                st.markdown(f"**Basket around `{anchor}`**")
                disp_anchor = anchor_basket.copy()
                disp_anchor['CoRate'] = disp_anchor['CoRate'].map('{:.1%}'.format)
                st.dataframe(disp_anchor, use_container_width=True, hide_index=True)
            with col_b2:
                top_anchor = anchor_basket.head(20)
                fig, ax = plt.subplots(figsize=(6, max(3, len(top_anchor) * 0.38)))
                ax.barh(top_anchor['ProductId'].astype(str)[::-1],
                        top_anchor['CoRate'].values[::-1],
                        color=PALETTE[1], alpha=0.85)
                ax.set_xlabel("Co-purchase rate with anchor", fontsize=9)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax.tick_params(labelsize=8)
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.set_title(f"Products bought alongside {anchor}", fontsize=9)
                fig.tight_layout()
                st.pyplot(fig); plt.close()

            # ── Wallet leakage for anchor basket ──────────────────────────────
            st.markdown("---")
            st.markdown("**Wallet leakage — anchor basket customers missing items**")
            st.caption(
                f"Customers who bought `{anchor}` but haven't bought one or more of its typical companion products"
            )

            anchor_basket_prods = anchor_basket['ProductId'].tolist()

            if anchor_basket_prods:
                cust_anchor_bought = (
                    fdf[
                        fdf['CustomerId'].isin(anchor_customers) &
                        fdf['ProductId'].isin(anchor_basket_prods)
                    ]
                    .groupby(['CustomerId', 'ProductId'])['InvoiceId']
                    .nunique()
                    .unstack(fill_value=0)
                    .reindex(columns=anchor_basket_prods, fill_value=0)
                )
                cust_anchor_binary = (cust_anchor_bought > 0).astype(int)

                # Customers who bought anchor but missing from the basket matrix
                missing_custs = [c for c in anchor_customers if c not in cust_anchor_binary.index]
                if missing_custs:
                    empty_rows = pd.DataFrame(
                        0, index=missing_custs, columns=anchor_basket_prods
                    )
                    cust_anchor_binary = pd.concat([cust_anchor_binary, empty_rows])

                anchor_leakage = pd.DataFrame({
                    'CustomerId':        cust_anchor_binary.index,
                    'BasketItemsBought': cust_anchor_binary.sum(axis=1).values,
                    'BasketSize':        len(anchor_basket_prods),
                })
                anchor_leakage['ItemsMissing']    = anchor_leakage['BasketSize'] - anchor_leakage['BasketItemsBought']
                anchor_leakage['CoverageRate']    = anchor_leakage['BasketItemsBought'] / anchor_leakage['BasketSize']
                anchor_leakage['MissingProducts'] = cust_anchor_binary.apply(
                    lambda row: ', '.join([str(p) for p in anchor_basket_prods if row.get(p, 0) == 0]),
                    axis=1
                )
                anchor_leakage = anchor_leakage.merge(
                    fdf.groupby('CustomerId')['LineRevenue'].sum().reset_index()
                    .rename(columns={'LineRevenue': 'TotalSpend'}),
                    on='CustomerId', how='left'
                ).sort_values('ItemsMissing', ascending=False)

                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Full basket customers", str((anchor_leakage['ItemsMissing'] == 0).sum()))
                with c2: metric_card("Missing 1+ items", str((anchor_leakage['ItemsMissing'] > 0).sum()))
                with c3: metric_card("Avg coverage", f"{anchor_leakage['CoverageRate'].mean():.1%}")

                anchor_leakage_display = anchor_leakage.copy()
                anchor_leakage_display['TotalSpend']   = anchor_leakage_display['TotalSpend'].map(fmt_currency)
                anchor_leakage_display['CoverageRate'] = anchor_leakage_display['CoverageRate'].map('{:.1%}'.format)
                st.dataframe(
                    anchor_leakage_display[['CustomerId', 'TotalSpend', 'BasketItemsBought',
                                            'ItemsMissing', 'CoverageRate', 'MissingProducts']],
                    use_container_width=True, hide_index=True
                )

                st.markdown("---")
                st.markdown("**Drill into a specific customer's basket gap**")
                leakage_cust = st.selectbox(
                    "Select customer",
                    anchor_leakage.sort_values('ItemsMissing', ascending=False)['CustomerId'].tolist(),
                    key="anchor_leakage_cust"
                )
                lrow_match = anchor_leakage[anchor_leakage['CustomerId'] == leakage_cust]
                if lrow_match.empty:
                    st.info("No leakage data available for this customer.")
                else:
                    lrow = lrow_match.iloc[0]
                    missing_anchor = [
                        p for p in anchor_basket_prods
                        if leakage_cust in cust_anchor_binary.index
                        and cust_anchor_binary.loc[leakage_cust, p] == 0
                    ]
                    col_l1, col_l2 = st.columns(2)
                    with col_l1:
                        metric_card("Basket coverage", f"{lrow['CoverageRate']:.1%}")
                    with col_l2:
                        metric_card("Items missing", str(int(lrow['ItemsMissing'])))

                    if missing_anchor:
                        st.markdown("**Products this customer hasn't bought (cross-sell opportunities)**")
                        miss_df = anchor_basket[anchor_basket['ProductId'].isin(missing_anchor)].copy()
                        miss_df['CoRate'] = miss_df['CoRate'].map('{:.1%}'.format)
                        st.dataframe(miss_df, use_container_width=True, hide_index=True)
                    else:
                        st.success("This customer buys everything in the expected basket.")
            else:
                st.info("No companion products meet the co-purchase rate threshold — try lowering the slider.")

    # ── TAB 3: BASKET SEGMENTATION ────────────────────────────────────────────
    with tab3:
        st.markdown(
            "**Define baskets that represent customer archetypes, then automatically score and assign every customer to the basket they match best.**"
        )

        # ── Session state init ─────────────────────────────────────────────────
        if 'defined_baskets' not in st.session_state:
            st.session_state['defined_baskets'] = {}  # {basket_name: [product_ids]}

        # ── Product universe filter ────────────────────────────────────────────
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            with st.expander("Filter product universe", expanded=False):
                filter_cols_bc = [c for c in cat_cols if c in fdf.columns]
                bc_filters = {}
                if filter_cols_bc:
                    fc_cols = st.columns(min(len(filter_cols_bc), 3))
                    for i, col in enumerate(filter_cols_bc):
                        with fc_cols[i % 3]:
                            vals = sorted(fdf[col].dropna().astype(str).unique().tolist())
                            selected_f = st.multiselect(col, vals, placeholder=f"All {col}", key=f"bseg_filter_{col}")
                            bc_filters[col] = selected_f

        with col_exp2:
            with st.expander("Filter inactive customers", expanded=False):
                cust_invoice_counts = fdf.groupby('CustomerId')['InvoiceId'].nunique().sort_values(ascending=False)
                total_custs = len(cust_invoice_counts)
                top_pct = st.slider(
                    "Keep top % of customers by invoices", 10, 100, 100, 5,
                    format="%d%%", key="bseg_active_pct"
                )
                n_keep = max(1, int(np.ceil(total_custs * top_pct / 100)))
                active_customers = set(cust_invoice_counts.head(n_keep).index.tolist())
                pct_threshold = int(cust_invoice_counts.iloc[n_keep - 1]) if n_keep <= total_custs else 0
                st.caption(
                    f"Keeping top {top_pct}% → {n_keep:,} of {total_custs:,} customers "
                    f"(min {pct_threshold} invoices)"
                )

        bc_df = fdf.copy()
        for col, vals in bc_filters.items():
            if vals:
                bc_df = bc_df[bc_df[col].astype(str).isin(vals)]

        # Apply active customer filter
        if top_pct < 100:
            bc_df = bc_df[bc_df['CustomerId'].isin(active_customers)]

        has_margin = 'TotalCostPerUnit' in bc_df.columns
        if has_margin:
            bc_df['LineMargin'] = (bc_df['PricePerUnit'] - bc_df['TotalCostPerUnit']) * bc_df['Quantity']

        all_products = sorted(bc_df['ProductId'].astype(str).unique().tolist())

        # ── Helper: basket info ────────────────────────────────────────────────
        def basket_info(product_list, df):
            """Return key stats for a list of products — revenue/margin from invoices where ALL products appear together."""
            prods = set(str(p) for p in product_list)
            if not prods:
                return None

            # Find invoices that contain all basket products
            inv_prod = (
                df[df['ProductId'].astype(str).isin(prods)]
                .groupby('InvoiceId')['ProductId']
                .apply(lambda x: set(x.astype(str)))
            )
            basket_invoices = inv_prod[inv_prod.apply(lambda x: prods.issubset(x))].index

            if len(basket_invoices) == 0:
                return {
                    'customers_all': 0,
                    'customers_any': len(set.union(*[
                        set(df[df['ProductId'].astype(str) == p]['CustomerId'].unique())
                        for p in prods
                    ])),
                    'total_rev': 0,
                    'total_mar': None,
                }

            # Revenue/margin only from those invoices, only for basket products
            basket_lines = df[
                df['InvoiceId'].isin(basket_invoices) &
                df['ProductId'].astype(str).isin(prods)
            ]

            customers_all = df[df['InvoiceId'].isin(basket_invoices)]['CustomerId'].nunique()
            customers_any = len(set.union(*[
                set(df[df['ProductId'].astype(str) == p]['CustomerId'].unique())
                for p in prods
            ]))
            total_rev = basket_lines['LineRevenue'].sum()
            total_mar = basket_lines['LineMargin'].sum() if 'LineMargin' in basket_lines.columns else None

            return {
                'customers_all': customers_all,
                'customers_any': customers_any,
                'total_rev':     total_rev,
                'total_mar':     total_mar,
            }

        # ── Auto-suggest baskets from co-occurrence ────────────────────────────
        st.markdown('<div class="section-header" style="font-size:1.1rem">Step 1 — Auto-suggested Baskets</div>', unsafe_allow_html=True)
        st.caption("Top product groups from co-occurrence data. Use these as starting points and edit before confirming.")

        @st.cache_data(show_spinner=False)
        def compute_top_combos_for_baskets(df, top_n=50, n_suggestions=8):
            top_prods = (
                df.groupby('ProductId')['InvoiceId'].nunique()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            inv_bin = (
                df[df['ProductId'].isin(top_prods)]
                .groupby(['InvoiceId', 'ProductId'])['Quantity']
                .sum()
                .unstack(fill_value=0)
            )
            inv_bin = (inv_bin > 0).astype(int)

            pairs = []
            for _, row in inv_bin.iterrows():
                bought = list(row[row == 1].index)
                for combo in combinations(sorted(bought), 2):
                    pairs.append(combo)

            if not pairs:
                return []

            pair_counts = pd.Series(pairs).value_counts()
            suggestions = []
            used_products = set()

            for combo, count in pair_counts.items():
                # Skip if any product in the core pair already used
                if any(p in used_products for p in combo):
                    continue

                combo_set = set(combo)

                # Find top companion not already used
                companion_counts = {}
                for _, row in inv_bin.iterrows():
                    if all(p in row.index and row[p] == 1 for p in combo_set):
                        for p in row[row == 1].index:
                            if p not in combo_set and p not in used_products:
                                companion_counts[p] = companion_counts.get(p, 0) + 1

                top_companion = sorted(companion_counts, key=companion_counts.get, reverse=True)[:1]
                basket_products_sug = list(combo_set) + top_companion  # exactly 3 products

                # Mark all as used
                used_products.update(basket_products_sug)

                suggestions.append({
                    'name':          f"Basket {' + '.join(str(p) for p in combo)}",
                    'products':      [str(p) for p in basket_products_sug],
                    'invoice_count': count,
                })

                if len(suggestions) >= n_suggestions:
                    break

            return suggestions

        with st.spinner("Computing basket suggestions..."):
            suggestions = compute_top_combos_for_baskets(bc_df)

        if suggestions:
            st.markdown("**Suggested baskets from co-occurrence analysis**")
            for i, sug in enumerate(suggestions[:6]):
                info = basket_info(sug['products'], bc_df)
                with st.expander(f"{sug['name']} — {len(sug['products'])} products", expanded=False):
                    if info:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: metric_card("Basket Customers", str(info["customers_all"]))
                        with c2: metric_card("Buy Any Product", str(info["customers_any"]))
                        with c3: metric_card("Basket Revenue", fmt_currency(info["total_rev"]))
                        with c4: metric_card("Basket Margin", fmt_currency(info["total_mar"]) if info["total_mar"] is not None else "—")
                    st.markdown(f"**Products:** {', '.join(sug['products'])}")
                    if st.button(f"Load into editor", key=f"load_sug_{i}"):
                        st.session_state['basket_editor_name'] = sug['name']
                        st.session_state['basket_editor_products'] = sug['products']

        # ── Basket editor ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header" style="font-size:1.1rem">Step 2 — Define & Save Baskets</div>', unsafe_allow_html=True)
        st.caption("Name your basket and select the products that define it. Save multiple baskets to build a full segmentation.")

        col_name, col_add = st.columns([1, 1])
        with col_name:
            basket_name = st.text_input(
                "Basket name (e.g. Ceiling Specialist)",
                value=st.session_state.get('basket_editor_name', ''),
                key="basket_name_input"
            )
        with col_add:
            basket_products = st.multiselect(
                "Products in this basket",
                all_products,
                default=st.session_state.get('basket_editor_products', []),
                key="basket_products_input"
            )

        # Live info for currently selected products
        if basket_products:
            info = basket_info(basket_products, bc_df)
            if info:
                st.markdown("**Current basket stats**")
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Basket Customers", str(info["customers_all"]))
                with c2: metric_card("Buy Any Product", str(info["customers_any"]))
                with c3: metric_card("Basket Revenue", fmt_currency(info["total_rev"]))
                with c4: metric_card("Basket Margin", fmt_currency(info["total_mar"]) if info["total_mar"] is not None else "—")

        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        with col_b1:
            if st.button("Save Basket", key="save_basket_btn"):
                if basket_name and basket_products:
                    st.session_state['defined_baskets'][basket_name] = [str(p) for p in basket_products]
                    st.session_state['basket_editor_name'] = ''
                    st.session_state['basket_editor_products'] = []
                    st.success(f"Saved '{basket_name}' with {len(basket_products)} products.")
                else:
                    st.warning("Please enter a basket name and select at least one product.")

        # Show defined baskets with info
        if st.session_state['defined_baskets']:
            st.markdown("**Defined baskets**")
            for bname, bprods in list(st.session_state['defined_baskets'].items()):
                with st.expander(f"{bname} — {len(bprods)} products", expanded=False):
                    info = basket_info(bprods, bc_df)
                    if info:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: metric_card("Basket Customers", str(info["customers_all"]))
                        with c2: metric_card("Buy Any Product", str(info["customers_any"]))
                        with c3: metric_card("Basket Revenue", fmt_currency(info["total_rev"]))
                        with c4: metric_card("Basket Margin", fmt_currency(info["total_mar"]) if info["total_mar"] is not None else "—")
                    st.markdown(f"**Products:** {', '.join(bprods)}")
                    if st.button("Remove", key=f"remove_basket_{bname}"):
                        del st.session_state['defined_baskets'][bname]
                        st.rerun()
        else:
            st.info("No baskets defined yet. Load a suggestion or build one manually above.")

        # ── Customer assignment ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header" style="font-size:1.1rem">Step 3 — Assign Customers</div>', unsafe_allow_html=True)

        min_invoices = st.slider(
            "Min invoices containing the full basket to be assigned",
            1, 20, 3,
            help="Counts only invoices where ALL basket products appear together. Customers below this threshold for every basket are labelled Insufficient Data"
        )

        if not st.session_state['defined_baskets']:
            st.info("Define at least one basket above to run assignment.")
        elif st.button("Run Customer Assignment", key="run_assignment_btn"):
            with st.spinner("Assigning customers..."):

                baskets = st.session_state['defined_baskets']

                # For each basket, count invoices where customer bought ALL basket products together
                basket_invoice_counts = {}
                for bname, bprods in baskets.items():
                    bprods_set = set(str(p) for p in bprods)

                    # Get invoices that contain ALL basket products
                    inv_prod = (
                        fdf[fdf['ProductId'].astype(str).isin(bprods_set)]
                        .groupby('InvoiceId')['ProductId']
                        .apply(lambda x: set(x.astype(str)))
                    )
                    full_basket_invoices = inv_prod[inv_prod.apply(lambda x: bprods_set.issubset(x))].index

                    # Count how many of those invoices each customer has
                    counts = (
                        fdf[fdf['InvoiceId'].isin(full_basket_invoices)]
                        .groupby('CustomerId')['InvoiceId']
                        .nunique()
                        .rename(bname)
                    )
                    basket_invoice_counts[bname] = counts

                # Build a customer × basket matrix of invoice counts
                counts_df = pd.DataFrame(basket_invoice_counts).fillna(0).astype(int)
                counts_df.index.name = 'CustomerId'
                counts_df = counts_df.reset_index()

                # All customers in the data
                all_custs = fdf['CustomerId'].unique()
                counts_df = counts_df.set_index('CustomerId').reindex(all_custs, fill_value=0).reset_index()

                results = []
                basket_names = list(baskets.keys())

                for _, row in counts_df.iterrows():
                    cust = row['CustomerId']
                    cust_counts = {b: row[b] for b in basket_names}
                    max_count   = max(cust_counts.values())

                    if max_count < min_invoices:
                        assigned = 'Insufficient Data'
                    else:
                        # Assign to basket with most invoices; ties broken alphabetically
                        assigned = max(cust_counts, key=lambda b: cust_counts[b])

                    results.append({
                        'CustomerId':     cust,
                        'AssignedBasket': assigned,
                        'BasketInvoices': max_count,
                        **{f'Invoices_{b}': cust_counts[b] for b in basket_names},
                    })

                assignment_df = pd.DataFrame(results)
                # Rename Insufficient Data to No Segmentation
                assignment_df['AssignedBasket'] = assignment_df['AssignedBasket'].replace(
                    'Insufficient Data', 'No Segmentation'
                )
                st.session_state['basket_assignment'] = assignment_df

        # ── Show assignment results ────────────────────────────────────────────
        if 'basket_assignment' in st.session_state:
            assignment_df = st.session_state['basket_assignment']

            st.markdown("**Assignment summary**")
            summary = (
                assignment_df.groupby('AssignedBasket')
                .agg(Customers=('CustomerId', 'count'))
                .sort_values('Customers', ascending=False)
                .reset_index()
            )

            # Merge in revenue
            cust_spend_df = fdf.groupby('CustomerId')['LineRevenue'].sum().reset_index().rename(columns={'LineRevenue': 'TotalSpend'})
            assignment_enriched = assignment_df.merge(cust_spend_df, on='CustomerId', how='left')
            rev_summary = assignment_enriched.groupby('AssignedBasket')['TotalSpend'].sum().reset_index()
            summary = summary.merge(rev_summary, on='AssignedBasket', how='left')
            summary['TotalSpend'] = summary['TotalSpend'].map(fmt_currency)

            c1, c2, c3 = st.columns(3)
            assigned_n = (assignment_df['AssignedBasket'].isin(st.session_state['defined_baskets'])).sum()
            with c1: metric_card("Assigned customers", str(assigned_n))
            with c2: metric_card("Unclassified", str((assignment_df['AssignedBasket'] == 'Unclassified').sum()))
            with c3: metric_card("No Segmentation", str((assignment_df['AssignedBasket'] == 'No Segmentation').sum()))

            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("**Full customer assignment table**")
            st.dataframe(assignment_df, use_container_width=True, hide_index=True)

            # ── Confirm to use in KVI ──────────────────────────────────────────
            st.markdown("---")
            confirm_basket_seg = st.checkbox(
                "Use this basket segmentation for KVI Classification",
                key="confirm_basket_seg_checkbox"
            )
            if confirm_basket_seg:
                # Only include customers with an actual basket assignment
                labels = (
                    assignment_df[
                        ~assignment_df['AssignedBasket'].isin(['No Segmentation', 'Unclassified'])
                    ][['CustomerId', 'AssignedBasket']]
                    .rename(columns={'AssignedBasket': 'Customer Cluster'})
                    .copy()
                )
                labels['CustomerId'] = labels['CustomerId'].astype(str)
                n_segmented = len(labels)
                st.session_state['confirmed_specialty'] = {
                    'labels':        labels,
                    'col':           'Basket Segmentation',
                    'threshold':     min_invoices,
                    'n_customers':   n_segmented,
                    'n_specialties': labels['Customer Cluster'].nunique(),
                }
                st.success(
                    f"Basket segmentation confirmed — {labels['Customer Cluster'].nunique()} groups "
                    f"across {n_segmented} customers (No Segmentation customers excluded). "
                    f"Head to KVI Classification to use it."
                )
            elif not confirm_basket_seg and 'confirmed_specialty' in st.session_state:
                if st.session_state['confirmed_specialty'].get('col') == 'Basket Segmentation':
                    del st.session_state['confirmed_specialty']

            # ── Export ────────────────────────────────────────────────────────
            if st.button("Export Assignment", key="export_assignment_btn"):
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as writer:
                    assignment_df.to_excel(writer, sheet_name='Customer Assignment', index=False)
                    summary.to_excel(writer, sheet_name='Summary', index=False)
                    # Basket definitions
                    basket_def_rows = []
                    for bname, bprods in st.session_state['defined_baskets'].items():
                        for p in bprods:
                            basket_def_rows.append({'Basket': bname, 'ProductId': p})
                    pd.DataFrame(basket_def_rows).to_excel(writer, sheet_name='Basket Definitions', index=False)
                out.seek(0)
                st.session_state['assignment_export_bytes'] = out.getvalue()

            if 'assignment_export_bytes' in st.session_state:
                st.download_button(
                    label="Download Assignment Excel",
                    data=st.session_state['assignment_export_bytes'],
                    file_name="basket_segmentation_assignment.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="assignment_download_btn"
                )

    # ── TAB 4: BASKET EXPLORATION ─────────────────────────────────────────────
    with tab4:
        st.markdown("**Explore the most common product combinations in the data, with revenue and customer metrics.**")

        # ── Controls ──────────────────────────────────────────────────────────
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            top_n_exp = st.slider("Top N baskets to show", 5, 50, 15, key="exp_top_n")
        with col_c2:
            no_overlap = st.toggle("No overlapping products", value=False, key="exp_no_overlap",
                                   help="Each product can only appear in one basket")
        with col_c3:
            top_pct_prods = st.slider(
                "Top % products by invoice frequency", 10, 100, 50, 10,
                format="%d%%", key="exp_top_pct_prods",
                help="Only include the top X% most frequently purchased products"
            )

        basket_size = 3

        # Apply top % product filter
        prod_inv_counts = fdf.groupby('ProductId')['InvoiceId'].nunique().sort_values(ascending=False)
        n_prods_keep = max(1, int(np.ceil(len(prod_inv_counts) * top_pct_prods / 100)))
        top_products_exp = set(prod_inv_counts.head(n_prods_keep).index.tolist())
        exp_input_df = fdf[fdf['ProductId'].isin(top_products_exp)]
        st.caption(f"Using top {top_pct_prods}% → {n_prods_keep:,} of {len(prod_inv_counts):,} products")

        @st.cache_data(show_spinner=False)
        def compute_basket_exploration(df, size, n_results):
            has_cost = 'TotalCostPerUnit' in df.columns
            if has_cost:
                df = df.copy()
                df['LineMargin'] = (df['PricePerUnit'] - df['TotalCostPerUnit']) * df['Quantity']

            inv_bin = (
                df.groupby(['InvoiceId', 'ProductId'])['Quantity']
                .sum()
                .unstack(fill_value=0)
            )
            inv_bin = (inv_bin > 0).astype(int)

            combos = []
            for _, row in inv_bin.iterrows():
                bought = list(row[row == 1].index)
                if len(bought) >= size:
                    for combo in combinations(sorted(bought), size):
                        combos.append(combo)

            if not combos:
                return pd.DataFrame()

            combo_counts = pd.Series(combos).value_counts().head(n_results * 3)

            rows = []
            for combo, inv_count in combo_counts.items():
                combo_set = set(combo)
                # Invoices with full basket
                inv_mask = inv_bin[list(combo_set)].all(axis=1)
                basket_invoices = inv_bin[inv_mask].index

                basket_lines = df[
                    df['InvoiceId'].isin(basket_invoices) &
                    df['ProductId'].isin(combo_set)
                ]
                cust_count = df[df['InvoiceId'].isin(basket_invoices)]['CustomerId'].nunique()
                revenue    = basket_lines['LineRevenue'].sum()
                margin     = basket_lines['LineMargin'].sum() if has_cost else None
                avg_rev_per_inv = revenue / len(basket_invoices) if len(basket_invoices) > 0 else 0

                rows.append({
                    'Combo':           combo,
                    'Products':        ' + '.join(str(p) for p in combo),
                    'InvoiceCount':    inv_count,
                    'CustomerCount':   cust_count,
                    'BasketRevenue':   round(revenue, 0),
                    'BasketMargin':    round(margin, 0) if margin is not None else None,
                    'AvgRevenuePerInvoice': round(avg_rev_per_inv, 0),
                })

            return pd.DataFrame(rows)

        with st.spinner("Computing basket exploration..."):
            exp_df = compute_basket_exploration(exp_input_df, basket_size, top_n_exp)

        if exp_df.empty:
            st.info("Not enough data for this basket size. Try reducing the basket size or expanding the product universe.")
        else:
            # Apply no-overlap filter
            if no_overlap:
                used = set()
                keep = []
                for _, row in exp_df.iterrows():
                    combo_set = set(row['Combo'])
                    if not combo_set & used:
                        keep.append(True)
                        used.update(combo_set)
                    else:
                        keep.append(False)
                exp_df = exp_df[keep].head(top_n_exp)
            else:
                exp_df = exp_df.head(top_n_exp)

            display_df = exp_df.drop(columns=['Combo']).copy()
            if 'BasketMargin' in display_df.columns and display_df['BasketMargin'].isna().all():
                display_df = display_df.drop(columns=['BasketMargin'])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ── Scatter plot ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Visualisation** — bubble size = basket revenue, hover for details")

            col_x, col_y = st.columns(2)
            axis_options = ['InvoiceCount', 'CustomerCount', 'AvgRevenuePerInvoice', 'BasketRevenue']
            if 'BasketMargin' in exp_df.columns and exp_df['BasketMargin'].notna().any():
                axis_options.append('BasketMargin')

            with col_x:
                x_axis = st.selectbox("X axis", axis_options, index=0, key="exp_x")
            with col_y:
                y_axis = st.selectbox("Y axis", axis_options, index=1, key="exp_y")

            import plotly.graph_objects as go

            plot_df = exp_df.copy()
            plot_df['BasketRevenue'] = plot_df['BasketRevenue'].fillna(0)
            max_rev = plot_df['BasketRevenue'].max()
            plot_df['BubbleSize'] = ((plot_df['BasketRevenue'] / max_rev * 60) + 5).fillna(5)

            hover_parts = [
                '<b>%{customdata[0]}</b><br>',
                f'{x_axis}: %{{x:,.0f}}<br>',
                f'{y_axis}: %{{y:,.0f}}<br>',
                'Basket Revenue: €%{customdata[1]:,.0f}<br>',
                'Customers: %{customdata[2]:,}<br>',
                'Invoices: %{customdata[3]:,}',
            ]
            if 'BasketMargin' in plot_df.columns and plot_df['BasketMargin'].notna().any():
                hover_parts.append('<br>Basket Margin: €%{customdata[4]:,.0f}')
                custom_data = plot_df[['Products', 'BasketRevenue', 'CustomerCount',
                                        'InvoiceCount', 'BasketMargin']].values
            else:
                custom_data = plot_df[['Products', 'BasketRevenue', 'CustomerCount',
                                        'InvoiceCount']].values

            fig = go.Figure(go.Scatter(
                x=plot_df[x_axis],
                y=plot_df[y_axis],
                mode='markers',
                marker=dict(
                    size=plot_df['BubbleSize'],
                    color=plot_df['BasketRevenue'],
                    colorscale='YlOrRd',
                    showscale=True,
                    colorbar=dict(title='Basket Revenue (€)', tickformat=',.0f'),
                    line=dict(color='#2e3246', width=0.5),
                    opacity=0.8,
                ),
                customdata=custom_data,
                hovertemplate=''.join(hover_parts) + '<extra></extra>',
            ))

            fig.update_layout(
                paper_bgcolor='#151720',
                plot_bgcolor='#151720',
                font=dict(color='#d4cfc7', size=11),
                xaxis=dict(
                    title=x_axis,
                    gridcolor='#2e3246',
                    zerolinecolor='#2e3246',
                ),
                yaxis=dict(
                    title=y_axis,
                    gridcolor='#2e3246',
                    zerolinecolor='#2e3246',
                ),
                title=dict(
                    text=f"Top {len(exp_df)} baskets of size {basket_size}" + (" (no overlap)" if no_overlap else ""),
                    font=dict(size=13, color='#f0ece3'),
                ),
                hoverlabel=dict(
                    bgcolor='#1c1f2b',
                    bordercolor='#e8c97e',
                    font=dict(color='#f0ece3', size=12),
                ),
                height=520,
                margin=dict(l=60, r=60, t=60, b=60),
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Customer Explorer ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header" style="font-size:1.1rem">Customer Explorer</div>', unsafe_allow_html=True)
            st.caption("Select products to define a basket and explore the customers who buy them together.")

            all_exp_products = sorted(fdf['ProductId'].astype(str).unique().tolist())

            # Pre-fill from a suggested basket in the table
            prefill_options = ["— select manually —"] + exp_df['Products'].tolist()
            prefill = st.selectbox("Pre-fill from a basket above", prefill_options, key="exp_prefill")

            if prefill != "— select manually —":
                prefill_row = exp_df[exp_df['Products'] == prefill].iloc[0]
                default_prods = [str(p) for p in prefill_row['Combo']]
            else:
                default_prods = []

            explorer_prods = st.multiselect(
                "Products in basket",
                all_exp_products,
                default=default_prods,
                key="exp_explorer_prods"
            )

            if not explorer_prods:
                st.info("Select at least one product to explore.")
            else:
                exp_prods_set = set(explorer_prods)

                # Invoices containing all selected products
                inv_prod_exp = (
                    fdf[fdf['ProductId'].astype(str).isin(exp_prods_set)]
                    .groupby('InvoiceId')['ProductId']
                    .apply(lambda x: set(x.astype(str)))
                )
                basket_inv = inv_prod_exp[inv_prod_exp.apply(lambda x: exp_prods_set.issubset(x))].index

                has_cost_exp = 'TotalCostPerUnit' in fdf.columns
                if has_cost_exp:
                    fdf['LineMargin'] = (fdf['PricePerUnit'] - fdf['TotalCostPerUnit']) * fdf['Quantity']

                basket_lines = fdf[
                    fdf['InvoiceId'].isin(basket_inv) &
                    fdf['ProductId'].astype(str).isin(exp_prods_set)
                ]

                # Top level metrics
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Basket Invoices",  str(len(basket_inv)))
                with c2: metric_card("Basket Customers", str(fdf[fdf['InvoiceId'].isin(basket_inv)]['CustomerId'].nunique()))
                with c3: metric_card("Basket Revenue",   fmt_currency(basket_lines['LineRevenue'].sum()))
                with c4: metric_card("Basket Margin",    fmt_currency(basket_lines['LineMargin'].sum()) if has_cost_exp else "—")

                st.markdown("")

                # Customer-level breakdown
                basket_customers = fdf[fdf['InvoiceId'].isin(basket_inv)]['CustomerId'].unique()

                cust_agg = (
                    fdf[fdf['CustomerId'].isin(basket_customers)]
                    .groupby('CustomerId')
                    .agg(
                        TotalSpend       =('LineRevenue',  'sum'),
                        TotalOrders      =('InvoiceId',    'nunique'),
                        **({'TotalMargin': ('LineMargin', 'sum')} if has_cost_exp else {}),
                    )
                    .reset_index()
                )

                # How many times each customer ordered the full basket
                basket_order_counts = (
                    fdf[fdf['InvoiceId'].isin(basket_inv)]
                    .groupby('CustomerId')['InvoiceId']
                    .nunique()
                    .reset_index()
                    .rename(columns={'InvoiceId': 'BasketOrders'})
                )

                # Basket revenue per customer (only from basket invoices + basket products)
                basket_rev_per_cust = (
                    basket_lines
                    .groupby('CustomerId')['LineRevenue']
                    .sum()
                    .reset_index()
                    .rename(columns={'LineRevenue': 'BasketRevenue'})
                )

                cust_agg = (
                    cust_agg
                    .merge(basket_order_counts, on='CustomerId', how='left')
                    .merge(basket_rev_per_cust, on='CustomerId', how='left')
                    .sort_values('BasketOrders', ascending=False)
                )

                # Add category info if available
                if cat_cols:
                    top_cat = (
                        fdf[fdf['CustomerId'].isin(basket_customers)]
                        .groupby('CustomerId')[cat_cols[0]]
                        .apply(lambda x: x.value_counts().index[0] if len(x) > 0 else '—')
                        .reset_index()
                        .rename(columns={cat_cols[0]: f'Top {cat_cols[0]}'})
                    )
                    cust_agg = cust_agg.merge(top_cat, on='CustomerId', how='left')

                cust_agg['BasketRevenue'] = cust_agg['BasketRevenue'].map(fmt_currency)
                cust_agg['TotalSpend']    = cust_agg['TotalSpend'].map(fmt_currency)
                if has_cost_exp and 'TotalMargin' in cust_agg.columns:
                    cust_agg['TotalMargin'] = cust_agg['TotalMargin'].map(fmt_currency)

                st.markdown("**Customers who ordered this basket**")
                st.dataframe(cust_agg, use_container_width=True, hide_index=True)

                # What else do these customers buy outside the basket
                st.markdown("**What else do these customers commonly buy?**")
                other = (
                    fdf[
                        fdf['CustomerId'].isin(basket_customers) &
                        ~fdf['ProductId'].astype(str).isin(exp_prods_set)
                    ]
                    .groupby('ProductId')
                    .agg(
                        CustomerCount =('CustomerId',  'nunique'),
                        TotalRevenue  =('LineRevenue',  'sum'),
                        **({'TotalMargin': ('LineMargin', 'sum')} if has_cost_exp else {}),
                    )
                    .reset_index()
                    .sort_values('CustomerCount', ascending=False)
                    .head(20)
                )
                other['CustomerRate'] = (other['CustomerCount'] / len(basket_customers)).map('{:.1%}'.format)
                other['TotalRevenue'] = other['TotalRevenue'].map(fmt_currency)
                if has_cost_exp and 'TotalMargin' in other.columns:
                    other['TotalMargin'] = other['TotalMargin'].map(fmt_currency)
                st.dataframe(other, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 6 — CUSTOMER SPECIALTY
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Customer Specialty":
    st.markdown('<div class="section-header">Customer Specialty</div>', unsafe_allow_html=True)

    # ── Controls ───────────────────────────────────────────────────────────────
    # Detect categorical columns suitable for specialty grouping
    cat_cols = [
        c for c in fdf.columns
        if str(fdf[c].dtype) in ('object', 'category')
        and c not in ['CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate']
        and fdf[c].nunique() < 200
    ]

    col_ctrl, col_thresh = st.columns([1, 1])
    with col_ctrl:
        specialty_col = st.selectbox(
            "Base specialty on",
            cat_cols,
            index=cat_cols.index('MainGroup') if 'MainGroup' in cat_cols else 0,
            help="The column whose dominant value determines a customer's specialty"
        )
    with col_thresh:
        threshold = st.slider(
            "Specialist threshold (min spend share)",
            min_value=10, max_value=80, value=40, step=5,
            format="%d%%",
            help="Customers below this share in their top group are labelled Generalist"
        ) / 100

    # ── Compute specialty per customer ─────────────────────────────────────────
    spend_by_col = (
        fdf.groupby(['CustomerId', specialty_col])['LineRevenue']
        .sum()
        .unstack(fill_value=0)
    )
    share_by_col = spend_by_col.div(spend_by_col.sum(axis=1), axis=0)

    specialty_df = pd.DataFrame({
        'Specialty':      share_by_col.idxmax(axis=1),
        'SpecialtyShare': share_by_col.max(axis=1),
        'TotalSpend':     spend_by_col.sum(axis=1),
    }).reset_index()

    specialty_df['Specialty'] = specialty_df.apply(
        lambda r: r['Specialty'] if r['SpecialtyShare'] >= threshold else 'Generalist',
        axis=1
    )

    # Merge in order/recency info
    specialty_df = specialty_df.merge(
        rfm[['CustomerId', 'Recency', 'Frequency']], on='CustomerId', how='left'
    )

    n_specialists  = (specialty_df['Specialty'] != 'Generalist').sum()
    n_generalists  = (specialty_df['Specialty'] == 'Generalist').sum()
    n_specialties  = specialty_df[specialty_df['Specialty'] != 'Generalist']['Specialty'].nunique()

    # ── Top metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Customers", str(len(specialty_df)))
    with c2: metric_card("Specialists", str(n_specialists))
    with c3: metric_card("Generalists", str(n_generalists))
    with c4: metric_card("Distinct Specialties", str(n_specialties))

    st.markdown("")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Specialty Overview", "Customer List", "Drill-down"])

    with tab1:
        specialty_summary = (
            specialty_df.groupby('Specialty')
            .agg(
                Customers    =('CustomerId',      'count'),
                TotalSpend   =('TotalSpend',      'sum'),
                AvgSpend     =('TotalSpend',      'mean'),
                AvgShare     =('SpecialtyShare',  'mean'),
                AvgRecency   =('Recency',         'mean'),
                AvgFrequency =('Frequency',       'mean'),
            )
            .sort_values('TotalSpend', ascending=False)
            .reset_index()
        )

        # Format for display
        display_summary = specialty_summary.copy()
        display_summary['TotalSpend']   = display_summary['TotalSpend'].map(fmt_currency)
        display_summary['AvgSpend']     = display_summary['AvgSpend'].map(fmt_currency)
        display_summary['AvgShare']     = display_summary['AvgShare'].map('{:.1%}'.format)
        display_summary['AvgRecency']   = display_summary['AvgRecency'].map('{:.0f} days'.format)
        display_summary['AvgFrequency'] = display_summary['AvgFrequency'].map('{:.1f}'.format)
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.markdown("")
        col_a, col_b = st.columns(2)

        with col_a:
            # Customer count per specialty bar chart
            chart_data = specialty_summary.sort_values('Customers', ascending=True)
            fig, ax = plt.subplots(figsize=(6, max(3, len(chart_data) * 0.38)))
            colors = [PALETTE[5] if s == 'Generalist' else PALETTE[0]
                      for s in chart_data['Specialty']]
            ax.barh(chart_data['Specialty'], chart_data['Customers'],
                    color=colors, alpha=0.85)
            ax.set_xlabel("# Customers", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_title("Customers per Specialty", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_b:
            # Revenue per specialty bar chart
            chart_data2 = specialty_summary.sort_values('TotalSpend', ascending=True)
            fig, ax = plt.subplots(figsize=(6, max(3, len(chart_data2) * 0.38)))
            colors2 = [PALETTE[5] if s == 'Generalist' else PALETTE[1]
                       for s in chart_data2['Specialty']]
            ax.barh(chart_data2['Specialty'], chart_data2['TotalSpend'],
                    color=colors2, alpha=0.85)
            ax.set_xlabel("Total Revenue (€)", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_title("Revenue per Specialty", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("**All customers with their assigned specialty**")

        # Filter by specialty
        all_specialties = sorted(specialty_df['Specialty'].unique().tolist())
        selected_specialties = st.multiselect(
            "Filter by specialty", all_specialties,
            placeholder="All specialties"
        )

        display_customers = specialty_df.copy()
        if selected_specialties:
            display_customers = display_customers[
                display_customers['Specialty'].isin(selected_specialties)
            ]

        display_customers = display_customers.sort_values('TotalSpend', ascending=False).copy()
        display_customers['TotalSpend']      = display_customers['TotalSpend'].map(fmt_currency)
        display_customers['SpecialtyShare']  = display_customers['SpecialtyShare'].map('{:.1%}'.format)
        display_customers['Recency']         = display_customers['Recency'].map('{:.0f} days'.format)

        st.dataframe(
            display_customers[['CustomerId', 'Specialty', 'SpecialtyShare',
                                'TotalSpend', 'Frequency', 'Recency']],
            use_container_width=True, hide_index=True
        )

        st.markdown("---")

        # Show current confirmed segmentation status
        if 'confirmed_specialty' in st.session_state:
            cs = st.session_state['confirmed_specialty']
            st.success(
                f"Active segmentation: **{cs['col']}** — "
                f"{cs['n_specialties']} specialties, "
                f"{cs['n_customers']} customers, "
                + (f"threshold {cs['threshold']:.0%}" if cs['col'] != 'Basket Segmentation' else f"min {cs['threshold']} invoices")
            )

        confirm = st.checkbox(
            "Use this segmentation for KVI Classification",
            value='confirmed_specialty' in st.session_state,
            key="confirm_seg_checkbox"
        )

        if confirm:
            labels = specialty_df[['CustomerId', 'Specialty']].copy()
            labels['CustomerId'] = labels['CustomerId'].astype(str)
            st.session_state['confirmed_specialty'] = {
                'labels':        labels,
                'col':           specialty_col,
                'threshold':     threshold,
                'n_customers':   len(specialty_df),
                'n_specialties': specialty_df['Specialty'].nunique(),
            }
            st.success(
                f"Segmentation confirmed — {specialty_df['Specialty'].nunique()} groups "
                f"across {len(specialty_df)} customers. "
                f"Head to KVI Classification to use it."
            )
        elif not confirm and 'confirmed_specialty' in st.session_state:
            del st.session_state['confirmed_specialty']

    with tab3:
        st.markdown("**Select a specialty to see its top customers and their category breakdown**")

        drill_specialty = st.selectbox(
            "Select specialty",
            [s for s in all_specialties if s != 'Generalist'],
            key="drill_spec"
        )

        spec_customers = specialty_df[specialty_df['Specialty'] == drill_specialty]['CustomerId'].tolist()
        spec_df = fdf[fdf['CustomerId'].isin(spec_customers)]

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Customers", str(len(spec_customers)))
        with c2: metric_card("Total Revenue", fmt_currency(spec_df['LineRevenue'].sum()))
        with c3: metric_card("Avg Specialty Share",
                              f"{specialty_df[specialty_df['Specialty']==drill_specialty]['SpecialtyShare'].mean():.1%}")

        st.markdown("")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**Top customers in {drill_specialty}**")
            top_spec_custs = (
                specialty_df[specialty_df['Specialty'] == drill_specialty]
                .sort_values('TotalSpend', ascending=False)
                .head(15)
                .copy()
            )
            top_spec_custs['TotalSpend']     = top_spec_custs['TotalSpend'].map(fmt_currency)
            top_spec_custs['SpecialtyShare'] = top_spec_custs['SpecialtyShare'].map('{:.1%}'.format)
            st.dataframe(
                top_spec_custs[['CustomerId', 'TotalSpend', 'SpecialtyShare', 'Frequency', 'Recency']],
                use_container_width=True, hide_index=True
            )

        with col_b:
            st.markdown(f"**What else do {drill_specialty} customers buy?**")
            # Spend share across ALL groups for this specialty's customers
            other_spend = (
                spec_df.groupby(specialty_col)['LineRevenue']
                .sum()
                .sort_values(ascending=False)
            )
            other_share = other_spend / other_spend.sum()

            fig, ax = plt.subplots(figsize=(6, max(3, len(other_spend) * 0.38)))
            colors_drill = [PALETTE[0] if g == drill_specialty else PALETTE[2]
                            for g in other_spend.index]
            ax.barh(other_spend.index[::-1], other_share.values[::-1],
                    color=colors_drill[::-1], alpha=0.85)
            ax.set_xlabel("Spend Share", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        # Individual customer deep-dive within this specialty
        st.markdown("---")
        st.markdown(f"**Individual customer breakdown within {drill_specialty}**")
        cust_pick = st.selectbox("Select customer", sorted(spec_customers), key="spec_cust_pick")

        cust_spec_df = fdf[fdf['CustomerId'] == cust_pick]
        cust_grp_spend = (
            cust_spec_df.groupby(specialty_col)['LineRevenue']
            .sum().sort_values(ascending=False)
        )
        cust_grp_share = cust_grp_spend / cust_grp_spend.sum()

        cust_display = pd.DataFrame({
            specialty_col: cust_grp_spend.index,
            'Spend':       cust_grp_spend.map(fmt_currency).values,
            'Share':       cust_grp_share.map('{:.1%}'.format).values,
        })
        st.dataframe(cust_display[cust_display['Spend'] != '€0'],
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 7 — KVI CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "KVI Classification":
    st.markdown('<div class="section-header">KVI Classification</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a8099;margin-top:-0.8rem;margin-bottom:1.2rem;font-size:0.85rem'>"
        "Classifies every product into KVI, Core, or Slow Mover based on demand, revenue, "
        "customer breadth, and basket co-occurrence."
        "</p>", unsafe_allow_html=True
    )

    # Controls
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        kvi_threshold = st.slider(
            "KVI score threshold", min_value=0.0, max_value=5.0, value=2.0, step=0.1,
            help="Products with KVI Score >= this value are classified as KVI"
        )
    with col_c2:
        core_pct = st.slider(
            "Core percentile (PurchaseCount)", min_value=50, max_value=95, value=75, step=5,
            help="Non-KVI products above this purchase count percentile are classified as Core"
        )

    # ── Segmentation source ────────────────────────────────────────────────────
    has_confirmed = 'confirmed_specialty' in st.session_state

    if has_confirmed:
        cs = st.session_state['confirmed_specialty']
        st.info(
            f"Using confirmed segmentation: "
            f"**{cs['col']}**, "
            + (f"threshold {cs['threshold']:.0%}" if cs['col'] != 'Basket Segmentation' else f"min {cs['threshold']} invoices")
            + f", {cs['n_specialties']} groups across {cs['n_customers']} customers."
        )
        seg_source = st.radio(
            "Run on", ["Confirmed specialty segmentation", "Custom group", "All customers"],
            horizontal=True, key="kvi_seg_source"
        )
    else:
        st.caption("No segmentation confirmed yet. Go to Customer Specialty → Customer List to confirm one, or select a group manually below.")
        seg_source = st.radio(
            "Run on", ["Custom group", "All customers"],
            horizontal=True, key="kvi_seg_source"
        )

    selected_group_val = None
    group_col_kvi = None

    if seg_source == "Confirmed specialty segmentation" and has_confirmed:
        confirmed_labels = st.session_state['confirmed_specialty']['labels'].copy()
        # Handle both old (Specialty) and new (Customer Cluster) column names
        if 'Specialty' in confirmed_labels.columns and 'Customer Cluster' not in confirmed_labels.columns:
            confirmed_labels = confirmed_labels.rename(columns={'Specialty': 'Customer Cluster'})
        specialty_options = sorted(confirmed_labels['Customer Cluster'].unique().tolist())
        selected_group_val = st.selectbox("Select specialty group", specialty_options, key="kvi_spec_val")
        group_col_kvi = "Customer Cluster"

    elif seg_source == "Custom group":
        group_col_kvi = st.selectbox("Group column", cat_cols, key="kvi_group_col")
        group_vals = sorted(fdf[group_col_kvi].dropna().astype(str).unique().tolist())
        selected_group_val = st.selectbox("Select group", group_vals, key="kvi_group_val")

    with st.spinner("Running KVI classification..."):
        if seg_source == "Confirmed specialty segmentation" and has_confirmed and selected_group_val:
            confirmed_labels = st.session_state['confirmed_specialty']['labels'].copy()
            if 'Specialty' in confirmed_labels.columns and 'Customer Cluster' not in confirmed_labels.columns:
                confirmed_labels = confirmed_labels.rename(columns={'Specialty': 'Customer Cluster'})
            group_customers = confirmed_labels[
                confirmed_labels['Customer Cluster'] == selected_group_val
            ]['CustomerId'].unique()
            kvi_input = fdf[fdf['CustomerId'].isin(group_customers)]
            scope_label = f"Specialty = {selected_group_val}"

        elif seg_source == "Custom group" and group_col_kvi and selected_group_val:
            group_customers = fdf[fdf[group_col_kvi].astype(str) == selected_group_val]['CustomerId'].unique()
            kvi_input = fdf[fdf['CustomerId'].isin(group_customers)]
            scope_label = f"{group_col_kvi} = {selected_group_val}"

        else:
            kvi_input = fdf
            scope_label = "All customers"

        if len(kvi_input) == 0:
            st.warning("No order lines found for this selection. Try a different group or check your filters.")
            st.stop()

        kvi_df = run_kvi_classification(kvi_input, kvi_threshold, core_pct)

    # Summary metrics
    n_kvi  = (kvi_df['Category'] == 'KVI').sum()
    n_core = (kvi_df['Category'] == 'Core').sum()
    n_slow = (kvi_df['Category'] == 'Slow Mover').sum()

    st.markdown(f"**Scope: {scope_label}** — {len(kvi_df):,} products analysed")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("KVI Products",    str(n_kvi))
    with c2: metric_card("Core Products",   str(n_core))
    with c3: metric_card("Slow Movers",     str(n_slow))
    with c4: metric_card("KVI Revenue Share",
                          f"{kvi_df[kvi_df['Category']=='KVI']['Revenue'].sum() / kvi_df['Revenue'].sum():.1%}")

    tab1, tab2, tab3, tab4 = st.tabs(["KVI", "Core", "Slow Movers", "Score Breakdown"])

    col_map = {'KVI': PALETTE[0], 'Core': PALETTE[1], 'Slow Mover': PALETTE[4]}

    def display_category_tab(cat_name):
        sub = kvi_df[kvi_df['Category'] == cat_name].copy()
        sub['Price']   = sub['Price'].map(lambda x: f"€{x:,.2f}")
        sub['Revenue'] = sub['Revenue'].map(fmt_currency)
        sub['Demand_Proportion']         = sub['Demand_Proportion'].map('{:.2%}'.format)
        sub['Revenue_Proportion']        = sub['Revenue_Proportion'].map('{:.2%}'.format)
        sub['UniqueCustomers_Proportion'] = sub['UniqueCustomers_Proportion'].map('{:.2%}'.format)
        sub['KVI_Score']                 = sub['KVI_Score'].map('{:.3f}'.format)
        sub['Corr_Score']                = sub['Corr_Score'].map('{:.4f}'.format)
        st.dataframe(
            sub[['ProductId', 'Quantity', 'Price', 'Revenue', 'UniqueCustomers',
                 'PurchaseCount', 'Demand_Proportion', 'Revenue_Proportion',
                 'UniqueCustomers_Proportion', 'Corr_Score', 'KVI_Score']],
            use_container_width=True, hide_index=True
        )

    with tab1:
        display_category_tab('KVI')

    with tab2:
        display_category_tab('Core')

    with tab3:
        display_category_tab('Slow Mover')

    with tab4:
        st.markdown("**KVI Score distribution across all products**")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors = [col_map[c] for c in kvi_df['Category']]
            ax.scatter(range(len(kvi_df)), kvi_df['KVI_Score'],
                       c=colors, alpha=0.6, s=8)
            ax.axhline(kvi_threshold, color='#e8c97e', linewidth=1.2,
                       linestyle='--', label=f'KVI threshold ({kvi_threshold})')
            ax.set_xlabel("Product rank", fontsize=9)
            ax.set_ylabel("KVI Score", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_b:
            # Category revenue breakdown
            cat_rev = kvi_df.groupby('Category')['Revenue'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(cat_rev['Category'], cat_rev['Revenue'],
                          color=[col_map[c] for c in cat_rev['Category']], alpha=0.85)
            ax.set_ylabel("Revenue (€)", fontsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            for bar, val in zip(bars, cat_rev['Revenue']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                        fmt_currency(val), ha='center', fontsize=8, color='#d4cfc7')
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("**Score component breakdown** — what drives each product's KVI Score")
        top_n_score = st.slider("Show top N products by KVI Score", 10, 100, 30, key="kvi_top_n")
        score_df = kvi_df.head(top_n_score)[
            ['ProductId', 'Category', 'Demand_Proportion_Scaled', 'Revenue_Proportion_Scaled',
             'UniqueCustomers_Proportion_Scaled', 'Corr_Score_Scaled', 'KVI_Score']
        ].copy()
        score_df['KVI_Score'] = score_df['KVI_Score'].map('{:.3f}'.format)
        score_df[['Demand_Proportion_Scaled', 'Revenue_Proportion_Scaled',
                  'UniqueCustomers_Proportion_Scaled', 'Corr_Score_Scaled']] = \
            score_df[['Demand_Proportion_Scaled', 'Revenue_Proportion_Scaled',
                      'UniqueCustomers_Proportion_Scaled', 'Corr_Score_Scaled']].round(3)
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.1rem">Export</div>', unsafe_allow_html=True)
    st.markdown("Exports one sheet per customer group with the full KVI classification for that group.")

    if st.button("Generate Export", key="kvi_export_btn"):
        with st.spinner("Building export..."):
            export_cols = ['ProductId', 'Quantity', 'Price', 'Revenue', 'UniqueCustomers',
                           'PurchaseCount', 'Demand_Proportion', 'Revenue_Proportion',
                           'UniqueCustomers_Proportion', 'Corr_Score', 'KVI_Score', 'Category']

            output = io.BytesIO()
            sheets_written = 0

            with pd.ExcelWriter(output, engine='openpyxl') as writer:

                if seg_source == "Confirmed specialty segmentation" and has_confirmed:
                    confirmed_labels = st.session_state['confirmed_specialty']['labels'].copy()
                    # Handle both old and new column names
                    if 'Specialty' in confirmed_labels.columns and 'Customer Cluster' not in confirmed_labels.columns:
                        confirmed_labels = confirmed_labels.rename(columns={'Specialty': 'Customer Cluster'})
                    all_groups = sorted(confirmed_labels['Customer Cluster'].unique().tolist())

                    summary_rows = []
                    for group in all_groups:
                        group_customers = confirmed_labels[
                            confirmed_labels['Customer Cluster'] == group
                        ]['CustomerId'].unique()
                        group_input = fdf[fdf['CustomerId'].isin(group_customers)]
                        if len(group_input) == 0:
                            continue
                        try:
                            group_kvi = run_kvi_classification(group_input, kvi_threshold, core_pct)
                        except Exception:
                            continue
                        sheet_name = str(group)[:31].replace('/', '-').replace('\\', '-').replace('*', '').replace('?', '').replace('[', '').replace(']', '') or f"Group_{sheets_written}"
                        group_kvi[export_cols].to_excel(writer, sheet_name=sheet_name, index=False)
                        sheets_written += 1
                        summary_rows.append({
                            'Group':         group,
                            'Customers':     len(group_customers),
                            'KVI':           (group_kvi['Category'] == 'KVI').sum(),
                            'Core':          (group_kvi['Category'] == 'Core').sum(),
                            'Slow Mover':    (group_kvi['Category'] == 'Slow Mover').sum(),
                            'KVI Revenue':   group_kvi[group_kvi['Category'] == 'KVI']['Revenue'].sum().round(2),
                            'Total Revenue': group_kvi['Revenue'].sum().round(2),
                        })

                    if summary_rows:
                        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
                        sheets_written += 1

                else:
                    sheet_name = (scope_label[:31].replace('/', '-').replace('=', '-')) or 'Results'
                    kvi_df[export_cols].to_excel(writer, sheet_name=sheet_name, index=False)
                    sheets_written += 1

            if sheets_written == 0:
                st.warning("No data to export — the selected groups had no matching customers in the current data.")
            else:
                output.seek(0)
                st.session_state['kvi_export_bytes'] = output.getvalue()
                st.session_state['kvi_export_name'] = f"kvi_classification_{'Specialty' if seg_source == 'Confirmed specialty segmentation' and has_confirmed else group_col_kvi if seg_source == 'Custom group' and group_col_kvi else 'all_customers'}.xlsx"

    if 'kvi_export_bytes' in st.session_state:
        st.download_button(
            label="Download Excel",
            data=st.session_state['kvi_export_bytes'],
            file_name=st.session_state['kvi_export_name'],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="kvi_download_btn"
        )

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 8 — PRICING SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Pricing Simulation":
    st.markdown('<div class="section-header">Pricing Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a8099;margin-top:-0.8rem;margin-bottom:1.2rem;font-size:0.85rem'>"
        "Set price change rules per segment × product category. Compare simulated revenue "
        "and margin against the baseline for a selected time period."
        "</p>", unsafe_allow_html=True
    )

    # ── Setup ──────────────────────────────────────────────────────────────────
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        sim_seg_default = (
            cat_cols.index('Customer Cluster') if 'Customer Cluster' in cat_cols
            else cat_cols.index('MainGroup') if 'MainGroup' in cat_cols
            else 0
        )
        seg_col_sim = st.selectbox(
            "Segment column (rows)", cat_cols,
            index=sim_seg_default,
            key="sim_seg_col"
        )
    with col_s2:
        kvi_col_sim = st.selectbox(
            "Product category column (columns)",
            ["KVI Classification"] + cat_cols,
            key="sim_kvi_col",
            help="Use 'KVI Classification' to use KVI/Core/Slow Mover categories from the KVI page, or pick any column"
        )
    with col_s3:
        # Date range for baseline
        if 'CreatedDate' in fdf.columns:
            min_d = fdf['CreatedDate'].min().date()
            max_d = fdf['CreatedDate'].max().date()
            sim_dates = st.date_input(
                "Simulation period",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
                key="sim_dates"
            )
        else:
            sim_dates = None

    # ── Filter to simulation period ────────────────────────────────────────────
    if sim_dates and len(sim_dates) == 2:
        sim_start = pd.Timestamp(sim_dates[0])
        sim_end   = pd.Timestamp(sim_dates[1])
        sim_df    = fdf[(fdf['CreatedDate'] >= sim_start) & (fdf['CreatedDate'] <= sim_end)].copy()
    else:
        sim_df = fdf.copy()

    # ── Check required cost column ─────────────────────────────────────────────
    if 'TotalCostPerUnit' not in sim_df.columns:
        st.error("TotalCostPerUnit column not found in data. Margin calculation requires this column.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def compute_simulation_baseline(sim_df_hash, seg_col, kvi_col):
        _df = sim_df_hash.copy()
        _df['LineMargin'] = (_df['PricePerUnit'] - _df['TotalCostPerUnit']) * _df['Quantity']

        if kvi_col == "KVI Classification":
            kvi_result = run_kvi_classification(_df)
            prod_cat_map = kvi_result.set_index('ProductId')['Category'].to_dict()
            _df['ProductCategory'] = _df['ProductId'].map(prod_cat_map).fillna('Unclassified')
            cats = ['KVI', 'Core', 'Slow Mover']
        else:
            _df['ProductCategory'] = _df[kvi_col].astype(str)
            cats = sorted(_df['ProductCategory'].dropna().unique().tolist())

        segs = sorted(_df[seg_col].dropna().astype(str).unique().tolist())

        baseline = (
            _df.groupby([seg_col, 'ProductCategory'])
            .agg(
                BaseRevenue=('LineRevenue', 'sum'),
                BaseMargin =('LineMargin',  'sum'),
                Quantity   =('Quantity',    'sum'),
            )
            .reset_index()
            .rename(columns={seg_col: 'Segment'})
        )
        baseline['Segment'] = baseline['Segment'].astype(str)
        return baseline, cats, segs

    baseline, prod_categories, seg_values = compute_simulation_baseline(
        sim_df, seg_col_sim, kvi_col_sim
    )

    total_base_revenue = baseline['BaseRevenue'].sum()
    total_base_margin  = baseline['BaseMargin'].sum()

    # ── Baseline summary metrics ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.1rem">Baseline</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Baseline Revenue", fmt_currency(total_base_revenue))
    with c2: metric_card("Baseline Margin",  fmt_currency(total_base_margin))
    with c3: metric_card("Baseline Margin %", f"{total_base_margin/total_base_revenue*100:.1f}%" if total_base_revenue else "—")
    with c4: metric_card("Period", f"{sim_df['CreatedDate'].min().strftime('%b %Y')} – {sim_df['CreatedDate'].max().strftime('%b %Y')}")

    # ── Pricing Rules Matrix ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.1rem">Pricing Rules Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        "Set a **% price change** per segment × product category cell. "
        "Positive = price increase, negative = discount. Leave at 0 for no change."
    )

    # Initialise matrix in session state so values persist
    matrix_key = f"price_matrix_{seg_col_sim}_{kvi_col_sim}"
    if matrix_key not in st.session_state:
        st.session_state[matrix_key] = {
            (seg, cat): 0.0
            for seg in seg_values
            for cat in prod_categories
        }

    # Render matrix as a grid of number inputs
    # Header row
    header_cols = st.columns([1.5] + [1] * len(prod_categories))
    header_cols[0].markdown(f"**{seg_col_sim}**")
    for i, cat in enumerate(prod_categories):
        header_cols[i + 1].markdown(f"**{cat}**")

    price_changes = {}
    for seg in seg_values:
        row_cols = st.columns([1.5] + [1] * len(prod_categories))
        row_cols[0].markdown(f"{seg}")
        for i, cat in enumerate(prod_categories):
            key = f"pm_{seg}_{cat}"
            val = row_cols[i + 1].number_input(
                "", min_value=-50.0, max_value=50.0,
                value=float(st.session_state[matrix_key].get((seg, cat), 0.0)),
                step=0.5, format="%.1f",
                key=key, label_visibility="collapsed"
            )
            price_changes[(seg, cat)] = val / 100.0
            st.session_state[matrix_key][(seg, cat)] = val

    # ── Run Simulation button ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Run Simulation", key="run_sim_btn"):
        st.session_state['sim_results'] = {
            'price_changes': price_changes,
            'baseline':      baseline.copy(),
            'base_revenue':  total_base_revenue,
            'base_margin':   total_base_margin,
            'seg_col':       seg_col_sim,
            'prod_cats':     prod_categories,
            'seg_values':    seg_values,
            'period':        f"{sim_df['CreatedDate'].min().strftime('%b %Y')} – {sim_df['CreatedDate'].max().strftime('%b %Y')}" if 'CreatedDate' in sim_df.columns else "",
        }

    if 'sim_results' not in st.session_state:
        st.info("Set your pricing rules above and click Run Simulation to see results.")
    else:
        sr = st.session_state['sim_results']
        results = sr['baseline'].copy()
        results['PriceChange'] = results.apply(
            lambda r: sr['price_changes'].get((r['Segment'], r['ProductCategory']), 0.0), axis=1
        )
        results['NewRevenue'] = results['BaseRevenue'] * (1 + results['PriceChange'])
        results['NewMargin']  = results['BaseMargin']  + results['PriceChange'] * results['BaseRevenue']

        total_new_revenue = results['NewRevenue'].sum()
        total_new_margin  = results['NewMargin'].sum()
        rev_delta = total_new_revenue - sr['base_revenue']
        mar_delta = total_new_margin  - sr['base_margin']
        rev_pct   = rev_delta / sr['base_revenue'] * 100 if sr['base_revenue'] else 0
        mar_pct   = mar_delta / sr['base_margin']  * 100 if sr['base_margin']  else 0

        st.markdown('<div class="section-header" style="font-size:1.1rem">Simulation Results</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("New Revenue",   fmt_currency(total_new_revenue),
                              f"{'+' if rev_pct >= 0 else ''}{rev_pct:.2f}% vs baseline")
        with c2: metric_card("New Margin",    fmt_currency(total_new_margin),
                              f"{'+' if mar_pct >= 0 else ''}{mar_pct:.2f}% vs baseline")
        with c3: metric_card("Revenue Delta", fmt_currency(rev_delta))
        with c4: metric_card("Margin Delta",  fmt_currency(mar_delta))

        # Per-segment breakdown
        st.markdown("**Impact by segment**")
        seg_results = (
            results.groupby('Segment')
            .agg(
                BaseRevenue=('BaseRevenue', 'sum'),
                NewRevenue =('NewRevenue',  'sum'),
                BaseMargin =('BaseMargin',  'sum'),
                NewMargin  =('NewMargin',   'sum'),
            )
            .reset_index()
        )
        seg_results['RevenueDelta'] = seg_results['NewRevenue'] - seg_results['BaseRevenue']
        seg_results['MarginDelta']  = seg_results['NewMargin']  - seg_results['BaseMargin']
        seg_results['Revenue∆%']    = (seg_results['RevenueDelta'] / seg_results['BaseRevenue'] * 100).map('{:+.2f}%'.format)
        seg_results['Margin∆%']     = (seg_results['MarginDelta']  / seg_results['BaseMargin']  * 100).map('{:+.2f}%'.format)

        seg_display = seg_results.copy()
        for col in ['BaseRevenue','NewRevenue','RevenueDelta','BaseMargin','NewMargin','MarginDelta']:
            seg_display[col] = seg_display[col].map(fmt_currency)
        st.dataframe(seg_display, use_container_width=True, hide_index=True)

        # Per-category breakdown
        st.markdown("**Impact by product category**")
        cat_results = (
            results.groupby('ProductCategory')
            .agg(
                BaseRevenue=('BaseRevenue', 'sum'),
                NewRevenue =('NewRevenue',  'sum'),
                BaseMargin =('BaseMargin',  'sum'),
                NewMargin  =('NewMargin',   'sum'),
            )
            .reset_index()
        )
        cat_results['RevenueDelta'] = cat_results['NewRevenue'] - cat_results['BaseRevenue']
        cat_results['MarginDelta']  = cat_results['NewMargin']  - cat_results['BaseMargin']
        cat_results['Revenue∆%']    = (cat_results['RevenueDelta'] / cat_results['BaseRevenue'] * 100).map('{:+.2f}%'.format)
        cat_results['Margin∆%']     = (cat_results['MarginDelta']  / cat_results['BaseMargin']  * 100).map('{:+.2f}%'.format)

        cat_display = cat_results.copy()
        for col in ['BaseRevenue','NewRevenue','RevenueDelta','BaseMargin','NewMargin','MarginDelta']:
            cat_display[col] = cat_display[col].map(fmt_currency)
        st.dataframe(cat_display, use_container_width=True, hide_index=True)

        # Charts
        st.markdown("**Revenue & Margin delta by segment**")
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, max(3, len(seg_results) * 0.45)))
            colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in seg_results['RevenueDelta']]
            ax.barh(seg_results['Segment'], seg_results['RevenueDelta'], color=colors, alpha=0.85)
            ax.axvline(0, color='#d4cfc7', linewidth=0.8)
            ax.set_xlabel("Revenue Delta (€)", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
            ax.tick_params(labelsize=8)
            ax.spines[['top','right','left']].set_visible(False)
            ax.set_title("Revenue impact by segment", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(6, max(3, len(seg_results) * 0.45)))
            colors = [PALETTE[1] if v >= 0 else PALETTE[3] for v in seg_results['MarginDelta']]
            ax.barh(seg_results['Segment'], seg_results['MarginDelta'], color=colors, alpha=0.85)
            ax.axvline(0, color='#d4cfc7', linewidth=0.8)
            ax.set_xlabel("Margin Delta (€)", fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(euro_axis_formatter))
            ax.tick_params(labelsize=8)
            ax.spines[['top','right','left']].set_visible(False)
            ax.set_title("Margin impact by segment", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        # ── Export ─────────────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("Export Simulation Results", key="sim_export_btn"):
            sim_output = io.BytesIO()
            results['RevenueDelta'] = results['NewRevenue'] - results['BaseRevenue']
            results['MarginDelta']  = results['NewMargin']  - results['BaseMargin']
            with pd.ExcelWriter(sim_output, engine='openpyxl') as writer:
                matrix_export = pd.DataFrame(
                    [[sr['price_changes'].get((seg, cat), 0.0) * 100 for cat in sr['prod_cats']] for seg in sr['seg_values']],
                    index=sr['seg_values'], columns=sr['prod_cats']
                )
                matrix_export.index.name = sr['seg_col']
                matrix_export.to_excel(writer, sheet_name='Pricing Rules')
                seg_results.to_excel(writer, sheet_name='By Segment', index=False)
                cat_results.to_excel(writer, sheet_name='By Product Category', index=False)
                results[['Segment','ProductCategory','BaseRevenue','NewRevenue',
                          'RevenueDelta','BaseMargin','NewMargin','MarginDelta','PriceChange']]\
                    .to_excel(writer, sheet_name='Full Detail', index=False)

            sim_output.seek(0)
            st.session_state['sim_export_bytes'] = sim_output.getvalue()
            st.session_state['sim_export_name'] = (
                f"pricing_simulation_{seg_col_sim}_{sim_dates[0]}_{sim_dates[1]}.xlsx"
                if sim_dates and len(sim_dates) == 2 else "pricing_simulation.xlsx"
            )

        if 'sim_export_bytes' in st.session_state:
            st.download_button(
                label="Download Simulation Excel",
                data=st.session_state['sim_export_bytes'],
                file_name=st.session_state['sim_export_name'],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="sim_download_btn"
            )
