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

def get_product_name_col(df):
    """Return the name of the product label column if available, else None."""
    for col in ['Name', 'Description']:
        if col in df.columns:
            return col
    return None

def enrich_with_product_name(result_df, source_df, id_col='ProductId'):
    """Insert a product name column after the ProductId column if Name/Description exists."""
    name_col = get_product_name_col(source_df)
    if name_col is None or id_col not in result_df.columns:
        return result_df
    name_map = source_df.drop_duplicates(id_col).set_index(id_col)[name_col]
    result_df = result_df.copy()
    result_df['ProductName'] = result_df[id_col].map(name_map)
    # Insert right after the id column
    idx = result_df.columns.tolist().index(id_col)
    cols = result_df.columns.tolist()
    cols.insert(idx + 1, cols.pop(cols.index('ProductName')))
    return result_df[cols]

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

def show_df(data, hide_index=True, currency_cols=None, percent_cols=None):
    """Display a full-width dataframe, compatible with both old Streamlit
    (use_container_width=True) and new Streamlit (width='stretch') APIs.

    currency_cols / percent_cols: optional list of column names that hold
    raw numeric values which should be *displayed* with € / % formatting
    while remaining numeric under the hood, so sorting/filtering in the
    dataframe widget stays numerically correct instead of alphabetical."""
    column_config = {}
    if currency_cols:
        for col in currency_cols:
            if col in data.columns:
                column_config[col] = st.column_config.NumberColumn(col, format="€ %,.2f")
    if percent_cols:
        for col in percent_cols:
            if col in data.columns:
                column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
    column_config = column_config or None
    try:
        st.dataframe(data, width='stretch', hide_index=hide_index, column_config=column_config)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=hide_index, column_config=column_config)

def show_chart(fig):
    """Display a full-width Plotly chart, compatible with both Streamlit width APIs."""
    try:
        st.plotly_chart(fig, width='stretch')
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes):
    df = pd.read_csv(file_bytes, low_memory=False) if file_bytes.name.endswith(".csv") else pd.read_excel(file_bytes)

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
def compute_base(df, _fingerprint):
    """_fingerprint is a cheap string used for cache invalidation instead
    of letting Streamlit hash the full 456k-row dataframe on every rerun."""
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
def compute_group_matrices(df, group_col, _fingerprint=''):
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
def run_kvi_classification(order_lines, kvi_score_threshold=2.0, core_percentile=75, _fingerprint=''):
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

    # Vectorized pair construction per invoice (numpy outer-product of indices,
    # instead of a nested Python double loop)
    row_chunks, col_chunks = [], []
    for products in grouped:
        idxs = np.array([product_to_idx[p] for p in products])
        n = len(idxs)
        if n == 0:
            continue
        row_chunks.append(np.repeat(idxs, n))
        col_chunks.append(np.tile(idxs, n))

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.ones(len(rows), dtype=np.float64)
        Q = coo_matrix((data, (rows, cols)), shape=(n_products, n_products)).tocsr()
        Q_j = np.asarray(Q.sum(axis=1)).flatten()
        Q_j[Q_j == 0] = 1
        from scipy.sparse import diags
        Q_norm = Q @ diags(1.0 / Q_j)
        score = np.asarray(Q_norm.sum(axis=1)).flatten() / n_products
        product_score_map = {p: score[i] for i, p in enumerate(product_ids)}
    else:
        product_score_map = {}

    dfAll['Corr_Score'] = dfAll['ProductId'].map(product_score_map).fillna(0)
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

uploaded = st.file_uploader("Upload order lines file", type=["csv", "xlsx"], label_visibility="collapsed")

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
    df_raw = load_and_prepare(uploaded)

with st.spinner("Computing metrics…"):
    rfm, prod_repeat = compute_base(df_raw, f"{len(df_raw)}_{df_raw['LineRevenue'].sum():.2f}")

# ── Cached sidebar helpers (run once, not on every click) ─────────────────────
@st.cache_data(show_spinner=False)
def get_sidebar_options(df):
    """Compute all sidebar filter options once from the raw df."""
    id_cols = {'CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate'}
    filter_cat_cols = [
        c for c in df.columns
        if c not in id_cols
        and 1 < df[c].nunique() < 1000
        and (
            str(df[c].dtype) in ('object', 'category', 'str', 'string')
            or str(df[c].dtype).startswith('str')
            or (df[c].dtype in ['int64', 'float64', 'Int64'] and df[c].nunique() < 50)
        )
    ]
    col_vals = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in filter_cat_cols
    }
    all_customers = sorted(df['CustomerId'].unique().tolist())
    min_date = df['CreatedDate'].min().date() if 'CreatedDate' in df.columns else None
    max_date = df['CreatedDate'].max().date() if 'CreatedDate' in df.columns else None
    n_rows = len(df)
    n_customers = df['CustomerId'].nunique()
    return filter_cat_cols, col_vals, all_customers, min_date, max_date, n_rows, n_customers

filter_cat_cols, col_vals, all_customers, min_date, max_date, n_rows, n_customers = get_sidebar_options(df_raw)

# ── Product aggregation (roll rows up to a chosen column, in place of ProductId) ─
@st.cache_data(show_spinner=False)
def aggregate_by_column(df_in, group_col):
    """
    Re-express order lines so that `group_col` acts as ProductId everywhere.
    Rows are aggregated to (InvoiceId, group_col); Quantity/Revenue/Cost are
    summed and per-unit prices are recomputed from those totals (weighted
    average), the same rollup logic used for the SubGroup case. Any other
    column is carried through with its first value per group, so it's only
    meaningful for attributes that are constant within the group (e.g. a
    parent category). Name/Description are dropped since they no longer
    identify a single product once rows are aggregated.
    """
    d = df_in.copy()
    has_cost = 'TotalCostPerUnit' in d.columns

    if 'LineRevenue' not in d.columns:
        d['LineRevenue'] = d['PricePerUnit'] * d['Quantity']
    if has_cost:
        d['LineCost'] = d['TotalCostPerUnit'] * d['Quantity']

    agg_dict = {'Quantity': 'sum', 'LineRevenue': 'sum'}
    if has_cost:
        agg_dict['LineCost'] = 'sum'

    skip = {
        'InvoiceId', group_col, 'ProductId', 'Quantity', 'LineRevenue', 'LineCost',
        'PricePerUnit', 'TotalCostPerUnit', 'Name', 'Description',
    }
    for c in d.columns:
        if c not in skip and c not in agg_dict:
            agg_dict[c] = 'first'

    grouped = d.groupby(['InvoiceId', group_col], as_index=False).agg(agg_dict)

    grouped['ProductId'] = grouped[group_col].astype(str)
    grouped['PricePerUnit'] = grouped['LineRevenue'] / grouped['Quantity']
    if has_cost:
        grouped['TotalCostPerUnit'] = grouped['LineCost'] / grouped['Quantity']
        grouped = grouped.drop(columns=['LineCost'])

    return grouped

# ── Sidebar ────────────────────────────────────────────────────────────────────
if 'filter_reset_counter' not in st.session_state:
    st.session_state['filter_reset_counter'] = 0

def _reset_filters():
    """Bump the reset counter so every filter widget below gets a brand-new
    key on the next run — Streamlit then renders them fresh instead of
    reusing (and visually retaining) the old widget instance."""
    st.session_state['filter_reset_counter'] += 1

with st.sidebar:
    st.markdown("### Explorer Controls")
    st.markdown("---")

    analysis = st.selectbox("Analysis view", [
        "Overview",
        "Category Breakdown",
        "Repeat Purchases",
        "Customer Specialty",
        "Time Analysis",
        "Basket Exploration",
        "Basket Segmentation",
        "KVI Classification",
        "Pricing Simulation",
    ])

    st.markdown("---")
    st.markdown("**Product Grouping**")
    agg_choices = ["— Off (use ProductId) —"] + filter_cat_cols
    agg_col_choice = st.selectbox(
        "Aggregate rows by", agg_choices, index=0, key="agg_col_choice",
        help=(
            "Roll order lines up to this column and treat it as ProductId "
            "everywhere in the app (e.g. SubGroup). KVI Classification and "
            "Pricing Simulation always use the real ProductId regardless of "
            "this setting, since KVI logic only makes sense at true product "
            "granularity."
        ),
    )

    st.markdown("---")
    filt_header_col, filt_reset_col = st.columns([2, 1])
    with filt_header_col:
        st.markdown("**Filters**")
    with filt_reset_col:
        st.button("Reset", key="reset_filters_btn", on_click=_reset_filters, help="Clear all filters below")

    _rk = st.session_state['filter_reset_counter']  # suffix appended to filter widget keys

    if min_date is not None:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"date_range_filter_{_rk}",
        )
    else:
        date_range = None

    selected_customers = st.multiselect(
        "Customers", all_customers, placeholder="All customers", key=f"customer_filter_{_rk}"
    )

    cat_filters = {}
    for col in filter_cat_cols:
        selected = st.multiselect(col, col_vals[col], placeholder=f"All {col}", key=f"filter_{col}_{_rk}")
        cat_filters[col] = selected

    st.markdown("---")
    st.markdown(f"<span style='font-size:0.75rem;color:#888'>{n_rows:,} rows · {n_customers:,} customers</span>", unsafe_allow_html=True)

# ── Apply product aggregation choice ────────────────────────────────────────────
if agg_col_choice != "— Off (use ProductId) —":
    df = aggregate_by_column(df_raw, agg_col_choice)
else:
    df = df_raw

# ── Apply filters ──────────────────────────────────────────────────────────────
cat_filters_tuple = tuple((k, tuple(v)) for k, v in cat_filters.items() if v)
_date_key = tuple(date_range) if date_range and len(date_range) == 2 else None
_cust_key  = tuple(selected_customers)
_seg_key   = str(st.session_state.get('confirmed_specialty', {}).get('n_customers', ''))

@st.cache_data(show_spinner=False)
def apply_filters(df, date_range, selected_customers, cat_filters_tuple, seg_key):
    fdf = df.copy()
    if date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        fdf = fdf[(fdf['CreatedDate'] >= start) & (fdf['CreatedDate'] <= end)]
    if selected_customers:
        fdf = fdf[fdf['CustomerId'].isin(selected_customers)]
    for col, vals in cat_filters_tuple:
        if vals:
            fdf = fdf[fdf[col].astype(str).isin(vals)]
    return fdf

fdf = apply_filters(df, _date_key, _cust_key, cat_filters_tuple, _seg_key)

# fdf_raw mirrors fdf but is always at true-ProductId granularity, regardless
# of the sidebar aggregation choice — used by KVI Classification and Pricing
# Simulation, which must never run on aggregated/synthetic ProductIds.
fdf_raw = apply_filters(df_raw, _date_key, _cust_key, cat_filters_tuple, _seg_key)

# Merge confirmed specialty labels — cached via a separate function so the
# 92ms merge doesn't run on every click
@st.cache_data(show_spinner=False)
def merge_labels(fdf, labels_df, seg_key):
    if labels_df is None:
        return fdf
    _labels = labels_df.copy()
    if 'Specialty' in _labels.columns and 'Customer Cluster' not in _labels.columns:
        _labels = _labels.rename(columns={'Specialty': 'Customer Cluster'})
    _labels['CustomerId'] = _labels['CustomerId'].astype(str)
    out = fdf.merge(_labels, on='CustomerId', how='left')
    out['Customer Cluster'] = out['Customer Cluster'].fillna('Unassigned')
    return out

if 'confirmed_specialty' in st.session_state:
    fdf = merge_labels(
        fdf,
        st.session_state['confirmed_specialty']['labels'],
        _seg_key
    )
    fdf_raw = merge_labels(
        fdf_raw,
        st.session_state['confirmed_specialty']['labels'],
        _seg_key
    )


@st.cache_data(show_spinner=False)
def get_cat_cols(df, _fp=''):
    id_cols = {'CustomerId', 'InvoiceId', 'ProductId', 'CreatedDate'}
    return [
        c for c in df.columns
        if c not in id_cols
        and 1 < df[c].nunique() < 1000
        and (
            str(df[c].dtype) in ('object', 'category', 'str', 'string')
            or str(df[c].dtype).startswith('str')
            or (df[c].dtype in ['int64', 'float64', 'Int64'] and df[c].nunique() < 50)
        )
    ]

# Robust fingerprint: keyed on what actually determines fdf content,
# not on the output data (avoids false cache hits from coincidental equality)
_fdf_fp = f"{_date_key}|{_cust_key}|{cat_filters_tuple}|{_seg_key}"

rfm, prod_repeat = compute_base(fdf, _fdf_fp)
cat_cols = get_cat_cols(fdf, _fdf_fp)

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
        show_df(grp_summary, currency_cols=['Revenue', 'AvgOrderValue'])

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
            'Spend': cust_grp.values,
            'Share': cust_grp_share.map('{:.1%}'.format).values,
        })
        show_df(cust_grp_df[cust_grp_df['Spend'] != 0], currency_cols=['Spend'])

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
        top_repeat['AvgOrderCount'] = top_repeat['AvgOrderCount'].map('{:.1f}'.format)
        show_df(enrich_with_product_name(top_repeat, fdf), currency_cols=['TotalSpend'])

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
            show_df(repeat_grp)
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
            show_df(
                enrich_with_product_name(cust_repeats[['ProductId','OrderCount','TotalQuantity','TotalSpend']], fdf),
                currency_cols=['TotalSpend']
            )

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# VIEW 5 — BASKET SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Basket Segmentation":
    st.markdown('<div class="section-header">Basket Segmentation</div>', unsafe_allow_html=True)

    # ── BASKET SEGMENTATION CONTENT ────────────────────────────────────────────

    st.markdown(
        "**Define baskets that represent customer archetypes, then automatically score and assign every customer to the basket they match best.**"
    )

    # ── Basket definition mode (global — applies to every basket below) ─────
    basket_mode_label = st.radio(
        "Definition mode",
        ["Basket — bought together in the same order", "Group of items — bought over time, any order"],
        horizontal=True, key="bseg_basket_mode",
        help=(
            "Basket mode requires all products to appear together on the same invoice — "
            "a true co-purchase bundle. Group of items mode is looser: a customer matches "
            "if they've bought any of these products at any point, even across separate "
            "orders — useful for archetypes defined by a category of goods rather than a "
            "literal cart."
        ),
    )
    is_group_mode = basket_mode_label.startswith("Group")
    basket_mode = "group" if is_group_mode else "basket"

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

    @st.cache_data(show_spinner=False)
    def get_cust_invoice_counts(fdf, _fp):
        return fdf.groupby('CustomerId')['InvoiceId'].nunique().sort_values(ascending=False)

    with col_exp2:
        with st.expander("Filter inactive customers", expanded=False):
            cust_invoice_counts = get_cust_invoice_counts(fdf, _fdf_fp)
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

    if top_pct < 100:
        bc_df = bc_df[bc_df['CustomerId'].isin(active_customers)]

    has_margin = 'TotalCostPerUnit' in bc_df.columns
    if has_margin:
        bc_df['LineMargin'] = (bc_df['PricePerUnit'] - bc_df['TotalCostPerUnit']) * bc_df['Quantity']

    @st.cache_data(show_spinner=False)
    def get_all_products(bc_df, _fp):
        return sorted(bc_df['ProductId'].astype(str).unique().tolist())

    _bc_fp = f"{_fdf_fp}|{tuple(sorted(bc_filters.items()))}|{top_pct}"
    all_products = get_all_products(bc_df, _bc_fp)

    # ── Helper: basket info (cached) ───────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def basket_info_cached(product_tuple, df, mode, _fp):
        """Return key stats for a basket/group.

        mode == 'basket': strict co-purchase — all products must appear
        together on the same invoice (original behaviour).
        mode == 'group':  loose category affinity — any line matching any
        of the products counts, regardless of invoice or timing.
        """
        prods = set(str(p) for p in product_tuple)
        if not prods:
            return None

        sub = df[df['ProductId'].astype(str).isin(prods)]
        customers_any = sub['CustomerId'].nunique()

        if mode == 'group':
            if sub.empty:
                return {'customers_all': 0, 'customers_any': 0, 'invoice_count': 0,
                        'total_rev': 0, 'total_mar': None}
            total_rev = sub['LineRevenue'].sum()
            total_mar = sub['LineMargin'].sum() if 'LineMargin' in sub.columns else None
            return {'customers_all': customers_any, 'customers_any': customers_any,
                    'invoice_count': sub['InvoiceId'].nunique(),
                    'total_rev': total_rev, 'total_mar': total_mar}

        # ── Basket (strict) mode — original co-occurrence logic ────────────
        prod_to_inv = sub.groupby('ProductId')['InvoiceId'].apply(set).to_dict()
        invoice_sets = list(prod_to_inv.values())
        basket_invoices = set.intersection(*invoice_sets) if invoice_sets else set()

        if not basket_invoices:
            return {'customers_all': 0, 'customers_any': customers_any, 'invoice_count': 0,
                    'total_rev': 0, 'total_mar': None}

        basket_lines = df[
            df['InvoiceId'].isin(basket_invoices) &
            df['ProductId'].astype(str).isin(prods)
        ]
        customers_all = df[df['InvoiceId'].isin(basket_invoices)]['CustomerId'].nunique()
        total_rev = basket_lines['LineRevenue'].sum()
        total_mar = basket_lines['LineMargin'].sum() if 'LineMargin' in basket_lines.columns else None

        return {'customers_all': customers_all, 'customers_any': customers_any,
                'invoice_count': len(basket_invoices),
                'total_rev': total_rev, 'total_mar': total_mar}

    def basket_info(product_list, df):
        return basket_info_cached(tuple(sorted(str(p) for p in product_list)), df, basket_mode, _bc_fp)

    def render_basket_metrics(info):
        """Draw the 4 metric cards, labelled according to the active mode."""
        c1, c2, c3, c4 = st.columns(4)
        if is_group_mode:
            with c1: metric_card("Group Customers", str(info["customers_any"]))
            with c2: metric_card("Orders Touching Group", str(info["invoice_count"]))
            with c3: metric_card("Group Revenue", fmt_currency(info["total_rev"]))
            with c4: metric_card("Group Margin", fmt_currency(info["total_mar"]) if info["total_mar"] is not None else "—")
        else:
            with c1: metric_card("Basket Customers", str(info["customers_all"]))
            with c2: metric_card("Buy Any Product", str(info["customers_any"]))
            with c3: metric_card("Basket Revenue", fmt_currency(info["total_rev"]))
            with c4: metric_card("Basket Margin", fmt_currency(info["total_mar"]) if info["total_mar"] is not None else "—")

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

        if inv_bin.shape[1] < 2 or inv_bin.shape[0] == 0:
            return []

        products = inv_bin.columns.tolist()
        mat = inv_bin.values

        # Vectorized pairwise co-occurrence via matrix multiplication —
        # replaces the row-by-row iterrows() + combinations() loop
        co_matrix = mat.T @ mat
        n = len(products)
        iu = np.triu_indices(n, k=1)
        pair_counts_arr = co_matrix[iu]
        order = np.argsort(-pair_counts_arr)

        suggestions = []
        used_products = set()

        for idx in order:
            count = pair_counts_arr[idx]
            if count == 0:
                break
            i, j = iu[0][idx], iu[1][idx]
            p1, p2 = products[i], products[j]
            if p1 in used_products or p2 in used_products:
                continue

            combo = tuple(sorted([p1, p2]))
            combo_set = set(combo)

            # Vectorized companion lookup — invoices containing both p1 and p2,
            # replaces the per-suggestion full iterrows() re-scan
            mask = (mat[:, i] == 1) & (mat[:, j] == 1)
            companion_counts = pd.Series(mat[mask].sum(axis=0), index=products)
            companion_counts = companion_counts.drop(labels=list(combo_set), errors='ignore')
            companion_counts = companion_counts[~companion_counts.index.isin(used_products)]
            companion_counts = companion_counts[companion_counts > 0].sort_values(ascending=False)
            top_companion = [companion_counts.index[0]] if len(companion_counts) > 0 else []

            basket_products_sug = list(combo_set) + top_companion
            used_products.update(basket_products_sug)

            suggestions.append({
                'name':          f"Basket {' + '.join(str(p) for p in combo)}",
                'products':      [str(p) for p in basket_products_sug],
                'invoice_count': int(count),
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
                    render_basket_metrics(info)
                st.markdown(f"**Products:** {', '.join(sug['products'])}")
                if st.button(f"Load into editor", key=f"load_sug_{i}"):
                    st.session_state['_basket_name_val'] = sug['name']
                    st.session_state['_basket_products_val'] = sug['products']
                    st.session_state['_basket_editor_version'] = st.session_state.get('_basket_editor_version', 0) + 1
                    st.rerun()

    # ── Basket editor ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.1rem">Step 2 — Define & Save Baskets</div>', unsafe_allow_html=True)
    st.caption("Name your basket and select the products that define it. Save multiple baskets to build a full segmentation.")

    col_name, col_add = st.columns([1, 1])
    # Use a counter in the key to force re-render when a suggestion is loaded
    editor_version = st.session_state.get('_basket_editor_version', 0)
    with col_name:
        basket_name = st.text_input(
            "Basket name (e.g. Ceiling Specialist)",
            value=st.session_state.get('_basket_name_val', ''),
            key=f"basket_name_input_{editor_version}"
        )
    with col_add:
        basket_products = st.multiselect(
            "Products in this basket",
            all_products,
            default=st.session_state.get('_basket_products_val', []),
            key=f"basket_products_input_{editor_version}"
        )

    # Live info for currently selected products
    if basket_products:
        info = basket_info(basket_products, bc_df)
        if info:
            st.markdown("**Current basket stats**")
            render_basket_metrics(info)

    col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
    with col_b1:
        if st.button("Save Basket", key="save_basket_btn"):
            if basket_name and basket_products:
                st.session_state['defined_baskets'][basket_name] = [str(p) for p in basket_products]
                st.session_state['_basket_name_val'] = ''
                st.session_state['_basket_products_val'] = []
                st.session_state['_basket_editor_version'] = st.session_state.get('_basket_editor_version', 0) + 1
                st.rerun()
            else:
                st.warning("Please enter a basket name and select at least one product.")

    # Show defined baskets with info
    if st.session_state['defined_baskets']:
        st.markdown("**Defined baskets**")
        for bname, bprods in list(st.session_state['defined_baskets'].items()):
            with st.expander(f"{bname} — {len(bprods)} products", expanded=False):
                info = basket_info(bprods, bc_df)
                if info:
                    render_basket_metrics(info)
                st.markdown(f"**Products:** {', '.join(bprods)}")
                if st.button("Remove", key=f"remove_basket_{bname}"):
                    del st.session_state['defined_baskets'][bname]
                    st.rerun()
    else:
        st.info("No baskets defined yet. Load a suggestion or build one manually above.")

    # ── Customer assignment ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header" style="font-size:1.1rem">Step 3 — Assign Customers</div>', unsafe_allow_html=True)
    st.caption(
        f"Definition mode: **{basket_mode_label}**. "
        + ("Thresholds below apply to any invoice touching any of a basket's products, anywhere in a customer's history."
           if is_group_mode else
           "Thresholds below apply only to invoices where the full basket appears together.")
    )

    assign_metric = st.selectbox(
        "Assign customers based on",
        ["Invoice Count", "Revenue", "Revenue Share of Customer Spend"],
        key="bseg_assign_metric",
        help=(
            ("Invoice Count — how many invoices touch any of the group's products. "
             "Revenue — total revenue from lines matching those products, anywhere in the customer's history. "
             "Revenue Share of Customer Spend — what share of a customer's overall spend "
             "comes from this group's products, which surfaces customers defined by the "
             "group even if they're not big spenders overall.")
            if is_group_mode else
            ("Invoice Count — how many invoices contain the full basket together. "
             "Revenue — total revenue from those invoices. "
             "Revenue Share of Customer Spend — what share of a customer's overall spend "
             "comes from the full basket, which surfaces customers defined by a basket "
             "even if they're not big spenders overall.")
        )
    )

    if assign_metric == "Invoice Count":
        assign_threshold = st.slider(
            "Min invoices containing the full basket to be assigned" if not is_group_mode
            else "Min invoices touching any group product to be assigned",
            1, 20, 3,
            help=(
                "Counts any invoice touching any of the group's products. Customers below this "
                "threshold for every group are labelled No Segmentation."
                if is_group_mode else
                "Counts only invoices where ALL basket products appear together. Customers below this threshold for every basket are labelled No Segmentation."
            )
        )
    elif assign_metric == "Revenue":
        assign_threshold = st.number_input(
            "Min revenue from the full basket to be assigned (€)" if not is_group_mode
            else "Min revenue from the group's products to be assigned (€)",
            min_value=0.0, value=100.0, step=25.0,
            help=(
                "Total revenue from lines matching any of the group's products, anywhere in the "
                "customer's history. Customers below this threshold for every group are labelled No Segmentation."
                if is_group_mode else
                "Total revenue from invoices where ALL basket products appear together. Customers below this threshold for every basket are labelled No Segmentation."
            )
        )
    else:
        assign_threshold = st.slider(
            "Min % of a customer's total spend that must come from the full basket" if not is_group_mode
            else "Min % of a customer's total spend that must come from the group's products",
            0, 100, 10,
            help=(
                "Share of the customer's overall spend that comes from any of the group's "
                "products, anywhere in their history. Customers below this threshold for every "
                "group are labelled No Segmentation."
                if is_group_mode else
                "Share of the customer's overall spend that comes from invoices containing the full basket. Customers below this threshold for every basket are labelled No Segmentation."
            )
        )

    if not st.session_state['defined_baskets']:
        st.info("Define at least one basket above to run assignment.")
    elif st.button("Run Customer Assignment", key="run_assignment_btn"):
        with st.spinner("Assigning customers..."):

            baskets = st.session_state['defined_baskets']
            all_custs = fdf['CustomerId'].unique()
            cust_total_spend = fdf.groupby('CustomerId')['LineRevenue'].sum()

            # For each basket, compute invoice count, revenue, and revenue share.
            # In "basket" mode this requires ALL basket products together on the
            # same invoice (and, matching the original behaviour, revenue is the
            # full invoice's revenue). In "group" mode it's a loose category
            # match — any invoice touching any of the group's products counts,
            # and revenue/share are based only on lines for those products.
            invoice_counts, revenue_stats, share_pct_stats = {}, {}, {}
            for bname, bprods in baskets.items():
                bprods_set = set(str(p) for p in bprods)

                if is_group_mode:
                    sub = fdf[fdf['ProductId'].astype(str).isin(bprods_set)]
                    inv_count = sub.groupby('CustomerId')['InvoiceId'].nunique().rename(bname)
                    revenue   = sub.groupby('CustomerId')['LineRevenue'].sum().rename(bname)
                else:
                    # Get invoices that contain ALL basket products
                    inv_prod = (
                        fdf[fdf['ProductId'].astype(str).isin(bprods_set)]
                        .groupby('InvoiceId')['ProductId']
                        .apply(lambda x: set(x.astype(str)))
                    )
                    full_basket_invoices = inv_prod[inv_prod.apply(lambda x: bprods_set.issubset(x))].index
                    basket_lines = fdf[fdf['InvoiceId'].isin(full_basket_invoices)]

                    inv_count = basket_lines.groupby('CustomerId')['InvoiceId'].nunique().rename(bname)
                    revenue   = basket_lines.groupby('CustomerId')['LineRevenue'].sum().rename(bname)

                share_pct = (revenue / cust_total_spend.reindex(revenue.index) * 100).fillna(0).rename(bname)

                invoice_counts[bname] = inv_count
                revenue_stats[bname]  = revenue
                share_pct_stats[bname] = share_pct

            basket_names = list(baskets.keys())

            def _to_matrix(stat_dict):
                m = pd.DataFrame(stat_dict).fillna(0)
                m.index.name = 'CustomerId'
                return m.reindex(all_custs, fill_value=0)

            invoices_df = _to_matrix(invoice_counts)
            revenue_df  = _to_matrix(revenue_stats)
            share_df    = _to_matrix(share_pct_stats)

            metric_df = {
                "Invoice Count":                  invoices_df,
                "Revenue":                         revenue_df,
                "Revenue Share of Customer Spend": share_df,
            }[assign_metric]

            # Vectorized assignment — no Python loop over customers
            max_stat    = metric_df[basket_names].max(axis=1)
            best_basket = metric_df[basket_names].idxmax(axis=1)

            assigned_series = best_basket.where(max_stat >= assign_threshold, 'No Segmentation')

            assignment_df = pd.DataFrame({
                'CustomerId':     metric_df.index,
                'AssignedBasket': assigned_series.values,
                'AssignmentStat': max_stat.values,
            })
            for b in basket_names:
                assignment_df[f'Invoices_{b}']     = invoices_df[b].values
                assignment_df[f'Revenue_{b}']      = revenue_df[b].values
                assignment_df[f'RevenueShare_{b}'] = share_df[b].values
            assignment_df = assignment_df.reset_index(drop=True)
            # Rename Insufficient Data to No Segmentation
            assignment_df['AssignedBasket'] = assignment_df['AssignedBasket'].replace(
                'Insufficient Data', 'No Segmentation'
            )
            st.session_state['basket_assignment'] = assignment_df
            st.session_state['basket_assignment_metric'] = assign_metric
            st.session_state['basket_assignment_threshold'] = assign_threshold
            st.session_state['basket_assignment_mode'] = basket_mode_label

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

        c1, c2, c3 = st.columns(3)
        assigned_n = (assignment_df['AssignedBasket'].isin(st.session_state['defined_baskets'])).sum()
        with c1: metric_card("Assigned customers", str(assigned_n))
        with c2: metric_card("Unclassified", str((assignment_df['AssignedBasket'] == 'Unclassified').sum()))
        with c3: metric_card("No Segmentation", str((assignment_df['AssignedBasket'] == 'No Segmentation').sum()))

        show_df(summary, currency_cols=['TotalSpend'])

        assign_metric_used = st.session_state.get('basket_assignment_metric', 'Invoice Count')
        assign_threshold_used = st.session_state.get('basket_assignment_threshold')
        assign_mode_used = st.session_state.get('basket_assignment_mode', basket_mode_label)
        # Derive basket names from whatever's actually in the stored table, so this
        # still works if baskets were redefined after the assignment last ran
        basket_names_disp = [c[len('Invoices_'):] for c in assignment_df.columns if c.startswith('Invoices_')]

        st.markdown("**Full customer assignment table**")
        st.caption(
            f"Assigned on **{assign_metric_used}** (threshold: {assign_threshold_used}), "
            f"definition mode: **{assign_mode_used}**. "
            "AssignmentStat is the value of that statistic for the customer's assigned basket."
        )
        table_currency_cols = [f'Revenue_{b}' for b in basket_names_disp]
        table_percent_cols  = [f'RevenueShare_{b}' for b in basket_names_disp]
        if assign_metric_used == "Revenue":
            table_currency_cols.append('AssignmentStat')
        elif assign_metric_used == "Revenue Share of Customer Spend":
            table_percent_cols.append('AssignmentStat')
        show_df(assignment_df, currency_cols=table_currency_cols, percent_cols=table_percent_cols)

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
            if assign_metric_used == "Invoice Count":
                threshold_label = f"min {assign_threshold_used} invoices"
            elif assign_metric_used == "Revenue":
                threshold_label = f"min {fmt_currency(assign_threshold_used)} revenue"
            else:
                threshold_label = f"min {assign_threshold_used:.0f}% revenue share"
            st.session_state['confirmed_specialty'] = {
                'labels':          labels,
                'col':             'Basket Segmentation',
                'threshold':       assign_threshold_used,
                'threshold_label': threshold_label,
                'n_customers':     n_segmented,
                'n_specialties':   labels['Customer Cluster'].nunique(),
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


# ══════════════════════════════════════════════════════════════════════════════
# VIEW 6 — BASKET EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Basket Exploration":
    st.markdown('<div class="section-header">Basket Exploration</div>', unsafe_allow_html=True)
    st.markdown("**Explore the most common product combinations in the data, with revenue and customer metrics.**")

    # ── Controls ──────────────────────────────────────────────────────────
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        top_n_exp = st.slider("Top N baskets to show", 5, 50, 15, key="exp_top_n")
    with col_c2:
        no_overlap = st.toggle("No overlapping products", value=False, key="exp_no_overlap",
                               help="Each product can only appear in one basket")
    with col_c3:
        no_overlap_customers = st.toggle("No overlapping customers", value=False, key="exp_no_overlap_customers",
                               help="Each customer can only be counted towards one basket")
    with col_c4:
        top_pct_prods = st.slider(
            "Top % products by invoice frequency", 10, 100, 50, 10,
            format="%d%%", key="exp_top_pct_prods",
            help="Only include the top X% most frequently purchased products"
        )

    basket_size = 3

    # Cache product frequency — avoids full groupby on every slider tweak
    @st.cache_data(show_spinner=False)
    def get_prod_inv_counts(fdf, _fp):
        return fdf.groupby('ProductId')['InvoiceId'].nunique().sort_values(ascending=False)

    prod_inv_counts = get_prod_inv_counts(fdf, _fdf_fp)
    n_prods_keep = max(1, int(np.ceil(len(prod_inv_counts) * top_pct_prods / 100)))
    top_products_exp = set(prod_inv_counts.head(n_prods_keep).index.tolist())
    exp_input_df = fdf[fdf['ProductId'].isin(top_products_exp)]
    st.caption(f"Using top {top_pct_prods}% → {n_prods_keep:,} of {len(prod_inv_counts):,} products")

    @st.cache_data(show_spinner=False)
    def compute_basket_exploration(df, size, n_results, pool_multiplier=3, _fp=''):
        from collections import Counter
        has_cost = 'TotalCostPerUnit' in df.columns
        if has_cost:
            df = df.copy()
            df['LineMargin'] = (df['PricePerUnit'] - df['TotalCostPerUnit']) * df['Quantity']

        # Build product list per invoice
        inv_products = (
            df.groupby('InvoiceId')['ProductId']
            .apply(lambda x: sorted(x.astype(str).unique().tolist()))
            .reset_index()
        )

        combo_counter = Counter()
        for prods in inv_products['ProductId']:
            if len(prods) >= size:
                for combo in combinations(prods, size):
                    combo_counter[combo] += 1

        if not combo_counter:
            return pd.DataFrame()

        top_combos = combo_counter.most_common(n_results * pool_multiplier)

        # ── One-time precomputation, reused for every candidate combo ──────────
        df = df.assign(_p=df['ProductId'].astype(str))

        # product -> set of invoices (already existed, keep it)
        prod_to_invoices = df.groupby('_p')['InvoiceId'].apply(set).to_dict()

        # invoice -> customer (single lookup, avoids re-filtering df per combo)
        inv_to_cust = df.drop_duplicates('InvoiceId').set_index('InvoiceId')['CustomerId']

        # (InvoiceId, ProductId) -> revenue/margin aggregated once
        line_agg_cols = {'LineRevenue': 'sum'}
        if has_cost:
            line_agg_cols['LineMargin'] = 'sum'
        line_agg = (
            df.groupby(['InvoiceId', '_p'])
            .agg(**{k: (k, v) for k, v in line_agg_cols.items()})
        )
        # MultiIndex (InvoiceId, ProductId) -> revenue / margin, for fast .loc slicing
        rev_by_invprod = line_agg['LineRevenue']
        mar_by_invprod = line_agg['LineMargin'] if has_cost else None

        rows = []
        for combo, inv_count in top_combos:
            combo_set = set(combo)
            invoice_sets = [prod_to_invoices.get(p, set()) for p in combo_set]
            basket_inv = set.intersection(*invoice_sets) if invoice_sets else set()

            if not basket_inv:
                rows.append({
                    'Combo': combo, 'Products': ' + '.join(str(p) for p in combo),
                    'InvoiceCount': inv_count, 'CustomerCount': 0,
                    'CustomerSet': frozenset(),
                    'BasketRevenue': 0, 'BasketMargin': 0 if has_cost else None,
                    'AvgRevenuePerInvoice': 0,
                })
                continue

            basket_inv_list = list(basket_inv)
            combo_list = list(combo_set)

            # Vectorized slice of the precomputed (invoice, product) aggregates —
            # no scan of the original dataframe
            idx = pd.MultiIndex.from_product([basket_inv_list, combo_list])
            rev_slice = rev_by_invprod.reindex(idx).dropna()
            revenue = rev_slice.sum()
            margin = mar_by_invprod.reindex(idx).dropna().sum() if has_cost else None

            basket_customers = frozenset(inv_to_cust.reindex(basket_inv_list).dropna().unique())
            cust_count = len(basket_customers)
            avg_rev = revenue / len(basket_inv) if len(basket_inv) > 0 else 0

            rows.append({
                'Combo':              combo,
                'Products':           ' + '.join(str(p) for p in combo),
                'InvoiceCount':       inv_count,
                'CustomerCount':      cust_count,
                'CustomerSet':        basket_customers,
                'BasketRevenue':      round(revenue, 0),
                'BasketMargin':       round(margin, 0) if margin is not None else None,
                'AvgRevenuePerInvoice': round(avg_rev, 0),
            })

        return pd.DataFrame(rows)

    with st.spinner("Computing basket exploration..."):
        exp_df = compute_basket_exploration(exp_input_df, basket_size, top_n_exp, 3, _fdf_fp)

    if exp_df.empty:
        st.info("Not enough data for this basket size. Try reducing the basket size or expanding the product universe.")
    else:
        # Apply no-overlap filter(s)
        if no_overlap or no_overlap_customers:
            used_products = set()
            used_customers = set()
            keep = []
            for _, row in exp_df.iterrows():
                combo_set = set(row['Combo'])
                cust_set = row['CustomerSet']
                product_conflict = no_overlap and bool(combo_set & used_products)
                customer_conflict = no_overlap_customers and bool(cust_set & used_customers)
                if not product_conflict and not customer_conflict:
                    keep.append(True)
                    if no_overlap:
                        used_products.update(combo_set)
                    if no_overlap_customers:
                        used_customers.update(cust_set)
                else:
                    keep.append(False)
            exp_df = exp_df[keep].head(top_n_exp)
        else:
            exp_df = exp_df.head(top_n_exp)

        display_df = exp_df.drop(columns=['Combo', 'CustomerSet']).copy()

        _overlap_bits = []
        if no_overlap:
            _overlap_bits.append("no product overlap")
        if no_overlap_customers:
            _overlap_bits.append("no customer overlap")
        overlap_suffix = f" ({', '.join(_overlap_bits)})" if _overlap_bits else ""
        if 'BasketMargin' in display_df.columns and display_df['BasketMargin'].isna().all():
            display_df = display_df.drop(columns=['BasketMargin'])

        # Add product names column if available
        name_col = get_product_name_col(fdf)
        if name_col is not None:
            name_map = fdf.drop_duplicates('ProductId').set_index('ProductId')[name_col]
            display_df.insert(
                display_df.columns.tolist().index('Products') + 1,
                'Product Names',
                exp_df['Combo'].apply(
                    lambda combo: ' + '.join(str(name_map.get(p, p)) for p in combo)
                )
            )

        # ── Assortment coverage ───────────────────────────────────────────
        # Unique customers covered by at least one basket in the displayed list
        # Track basket_inv sets computed during scoring so we don't redo it here
        # (do this by having compute_basket_exploration optionally return them, OR
        # recompute once cheaply using prod_to_invoices logic outside the cached fn)

        all_basket_products = set()
        for combo in exp_df['Combo']:
            all_basket_products.update(combo)

        # Build invoice->product-set only once, only over the relevant rows
        covered_invoices = (
            fdf[fdf['ProductId'].astype(str).isin(all_basket_products)]
            .assign(_p=lambda d: d['ProductId'].astype(str))
            .groupby('InvoiceId')['_p'].apply(frozenset)
        )

        covered_customers = set()
        inv_to_cust_full = fdf.drop_duplicates('InvoiceId').set_index('InvoiceId')['CustomerId']
        for combo in exp_df['Combo']:
            combo_set = frozenset(combo)
            mask = covered_invoices.apply(lambda s, cs=combo_set: cs <= s)  # subset check, same cost but frozenset is faster
            basket_inv_ids = covered_invoices[mask].index
            covered_customers.update(inv_to_cust_full.reindex(basket_inv_ids).dropna().unique())

        total_customers = fdf['CustomerId'].nunique()
        coverage_pct = len(covered_customers) / total_customers if total_customers > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Baskets shown", str(len(exp_df)))
        with c2: metric_card("Unique customers covered", f"{len(covered_customers):,}")
        with c3: metric_card("% of total customers", f"{coverage_pct:.1%}")

        show_df(display_df)

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
                text=f"Top {len(exp_df)} baskets of size {basket_size}" + overlap_suffix,
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

        show_chart(fig)

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
            new_prods = [str(p) for p in prefill_row['Combo']]
            if st.session_state.get('exp_prefill_last') != prefill:
                st.session_state['exp_prefill_last'] = prefill
                st.session_state['exp_explorer_prods'] = new_prods
                st.rerun()

        explorer_prods = st.multiselect(
            "Products in basket",
            all_exp_products,
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
            if has_cost_exp and 'LineMargin' not in fdf.columns:
                fdf = fdf.assign(LineMargin=(fdf['PricePerUnit'] - fdf['TotalCostPerUnit']) * fdf['Quantity'])

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

            st.markdown("**Customers who ordered this basket**")
            cust_agg_currency_cols = ['BasketRevenue', 'TotalSpend']
            if has_cost_exp and 'TotalMargin' in cust_agg.columns:
                cust_agg_currency_cols.append('TotalMargin')
            show_df(cust_agg, currency_cols=cust_agg_currency_cols)

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
            other_currency_cols = ['TotalRevenue']
            if has_cost_exp and 'TotalMargin' in other.columns:
                other_currency_cols.append('TotalMargin')
            show_df(other, currency_cols=other_currency_cols)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 6 — CUSTOMER SPECIALTY
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Customer Specialty":
    st.markdown('<div class="section-header">Customer Specialty</div>', unsafe_allow_html=True)

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
        display_summary['AvgShare']     = display_summary['AvgShare'].map('{:.1%}'.format)
        display_summary['AvgRecency']   = display_summary['AvgRecency'].map('{:.0f} days'.format)
        display_summary['AvgFrequency'] = display_summary['AvgFrequency'].map('{:.1f}'.format)
        show_df(display_summary, currency_cols=['TotalSpend', 'AvgSpend'])

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
        display_customers['SpecialtyShare']  = display_customers['SpecialtyShare'].map('{:.1%}'.format)
        display_customers['Recency']         = display_customers['Recency'].map('{:.0f} days'.format)

        show_df(
            display_customers[['CustomerId', 'Specialty', 'SpecialtyShare',
                                'TotalSpend', 'Frequency', 'Recency']],
            currency_cols=['TotalSpend']
        )

        st.markdown("---")

        # Show current confirmed segmentation status
        if 'confirmed_specialty' in st.session_state:
            cs = st.session_state['confirmed_specialty']
            st.success(
                f"Active segmentation: **{cs['col']}** — "
                f"{cs['n_specialties']} specialties, "
                f"{cs['n_customers']} customers, "
                + cs.get('threshold_label', f"threshold {cs['threshold']:.0%}")
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
                'labels':          labels,
                'col':             specialty_col,
                'threshold':       threshold,
                'threshold_label': f"threshold {threshold:.0%}",
                'n_customers':     len(specialty_df),
                'n_specialties':   specialty_df['Specialty'].nunique(),
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
            top_spec_custs['SpecialtyShare'] = top_spec_custs['SpecialtyShare'].map('{:.1%}'.format)
            show_df(
                top_spec_custs[['CustomerId', 'TotalSpend', 'SpecialtyShare', 'Frequency', 'Recency']],
                currency_cols=['TotalSpend']
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
            'Spend':       cust_grp_spend.values,
            'Share':       cust_grp_share.map('{:.1%}'.format).values,
        })
        show_df(cust_display[cust_display['Spend'] != 0], currency_cols=['Spend'])

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 7 — TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Time Analysis":
    st.markdown('<div class="section-header">Time Analysis</div>', unsafe_allow_html=True)

    if 'CreatedDate' not in fdf.columns or fdf['CreatedDate'].dropna().empty:
        st.info("No date column available to compare timeframes.")
    else:
        ta_min = fdf['CreatedDate'].min().date()
        ta_max = fdf['CreatedDate'].max().date()
        ta_mid = ta_min + (ta_max - ta_min) // 2

        st.markdown("Pick two date ranges to compare performance between them.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Period A**")
            period_a = st.date_input(
                "Period A range", value=(ta_min, ta_mid),
                min_value=ta_min, max_value=ta_max, key="ta_period_a"
            )
        with col_b:
            st.markdown("**Period B**")
            period_b = st.date_input(
                "Period B range", value=(min(ta_mid + pd.Timedelta(days=1), ta_max), ta_max),
                min_value=ta_min, max_value=ta_max, key="ta_period_b"
            )

        if len(period_a) != 2 or len(period_b) != 2:
            st.warning("Select a start and end date for both periods.")
        else:
            a_start, a_end = pd.Timestamp(period_a[0]), pd.Timestamp(period_a[1])
            b_start, b_end = pd.Timestamp(period_b[0]), pd.Timestamp(period_b[1])

            df_a = fdf[(fdf['CreatedDate'] >= a_start) & (fdf['CreatedDate'] <= a_end)]
            df_b = fdf[(fdf['CreatedDate'] >= b_start) & (fdf['CreatedDate'] <= b_end)]

            has_cost = 'TotalCostPerUnit' in fdf.columns
            if has_cost:
                df_a = df_a.assign(LineMargin=(df_a['PricePerUnit'] - df_a['TotalCostPerUnit']) * df_a['Quantity'])
                df_b = df_b.assign(LineMargin=(df_b['PricePerUnit'] - df_b['TotalCostPerUnit']) * df_b['Quantity'])

            if df_a.empty or df_b.empty:
                st.warning("One or both periods contain no data. Adjust the date ranges and try again.")
            else:
                if a_start <= b_end and b_start <= a_end:
                    st.caption("Note: the selected periods overlap.")

                # ── Headline comparison ─────────────────────────────────────────
                def _ta_summary(d):
                    n_lines = len(d)
                    out = {
                        'Revenue':   d['LineRevenue'].sum(),
                        'Quantity':  d['Quantity'].sum(),
                        'Orders':    d['InvoiceId'].nunique(),
                        'Customers': d['CustomerId'].nunique(),
                        'Lines':     n_lines,
                    }
                    out['AOV'] = out['Revenue'] / out['Orders'] if out['Orders'] else 0
                    out['AvgQtyPerLine'] = out['Quantity'] / n_lines if n_lines else 0
                    if has_cost:
                        out['Margin'] = d['LineMargin'].sum()
                    return out

                sum_a, sum_b = _ta_summary(df_a), _ta_summary(df_b)

                def _ta_delta_card(label, a_val, b_val, is_currency=True, decimals=0):
                    delta = b_val - a_val
                    pct = (delta / a_val * 100) if a_val else 0
                    sign = '+' if delta >= 0 else ''
                    if is_currency:
                        a_disp, b_disp, d_disp = fmt_currency(a_val), fmt_currency(b_val), fmt_currency(delta)
                    else:
                        a_disp, b_disp, d_disp = f"{a_val:,.{decimals}f}", f"{b_val:,.{decimals}f}", f"{delta:,.{decimals}f}"
                    metric_card(label, f"{a_disp} → {b_disp}", f"Δ {sign}{d_disp} ({sign}{pct:.1f}%)")

                st.markdown("---")
                st.markdown('<div class="section-header" style="font-size:1.1rem">Headline comparison</div>', unsafe_allow_html=True)
                st.caption("Each card shows A → B, with the difference (Δ) below.")

                headline_metrics = [
                    ("Revenue",                sum_a['Revenue'],        sum_b['Revenue'],        dict(is_currency=True)),
                    ("Orders",                 sum_a['Orders'],         sum_b['Orders'],         dict(is_currency=False)),
                    ("Active Customers",       sum_a['Customers'],      sum_b['Customers'],      dict(is_currency=False)),
                    ("Avg Order Value",        sum_a['AOV'],            sum_b['AOV'],            dict(is_currency=True)),
                    ("Avg Qty per Order Line", sum_a['AvgQtyPerLine'],  sum_b['AvgQtyPerLine'],  dict(is_currency=False, decimals=2)),
                ]
                if has_cost:
                    headline_metrics.append(("Margin", sum_a['Margin'], sum_b['Margin'], dict(is_currency=True)))

                for i in range(0, len(headline_metrics), 3):
                    row = headline_metrics[i:i + 3]
                    row_cols = st.columns(len(row))
                    for col, (label, a_val, b_val, kwargs) in zip(row_cols, row):
                        with col:
                            _ta_delta_card(label, a_val, b_val, **kwargs)

                st.caption(
                    f"Period A: {a_start.date()} – {a_end.date()}   ·   "
                    f"Period B: {b_start.date()} – {b_end.date()}"
                )

                # ── Shared helpers for the three breakdown tabs ─────────────────
                def _ta_build_comparison(d_a, d_b, group_col, only_both=False):
                    agg = {
                        'Revenue':  ('LineRevenue', 'sum'),
                        'Quantity': ('Quantity', 'sum'),
                        'Orders':   ('InvoiceId', 'nunique'),
                        'Lines':    ('Quantity', 'size'),
                    }
                    if has_cost:
                        agg['Margin'] = ('LineMargin', 'sum')
                    a_g = d_a.groupby(group_col).agg(**agg).reset_index()
                    b_g = d_b.groupby(group_col).agg(**agg).reset_index()
                    merged = a_g.merge(b_g, on=group_col, how=('inner' if only_both else 'outer'), suffixes=('_A', '_B')).fillna(0)
                    merged['Revenue Δ'] = merged['Revenue_B'] - merged['Revenue_A']
                    merged['Revenue Δ%'] = merged.apply(
                        lambda r: (r['Revenue Δ'] / r['Revenue_A'] * 100) if r['Revenue_A'] else np.nan, axis=1
                    )
                    merged['Lines Δ'] = merged['Lines_B'] - merged['Lines_A']
                    merged['AvgQtyPerLine_A'] = (merged['Quantity_A'] / merged['Lines_A'].replace(0, np.nan)).fillna(0)
                    merged['AvgQtyPerLine_B'] = (merged['Quantity_B'] / merged['Lines_B'].replace(0, np.nan)).fillna(0)
                    merged['AvgQtyPerLine Δ'] = merged['AvgQtyPerLine_B'] - merged['AvgQtyPerLine_A']
                    merged['AvgPrice_A'] = (merged['Revenue_A'] / merged['Quantity_A'].replace(0, np.nan)).fillna(0)
                    merged['AvgPrice_B'] = (merged['Revenue_B'] / merged['Quantity_B'].replace(0, np.nan)).fillna(0)
                    merged['AvgPrice Δ'] = merged['AvgPrice_B'] - merged['AvgPrice_A']
                    merged['AOV_A'] = (merged['Revenue_A'] / merged['Orders_A'].replace(0, np.nan)).fillna(0)
                    merged['AOV_B'] = (merged['Revenue_B'] / merged['Orders_B'].replace(0, np.nan)).fillna(0)
                    if has_cost:
                        merged['Margin Δ'] = merged['Margin_B'] - merged['Margin_A']
                    return merged.sort_values('Revenue Δ', ascending=False)

                def _ta_display_table(cmp_df, show_avg_qty=False, show_avg_price=False):
                    # Values stay numeric (not formatted strings) so the dataframe
                    # widget sorts correctly on magnitude rather than alphabetically.
                    cols = [c for c in cmp_df.columns if c not in ('AOV_A', 'AOV_B')]
                    if not show_avg_qty:
                        cols = [
                            c for c in cols
                            if c not in ('Lines_A', 'Lines_B', 'Lines Δ', 'AvgQtyPerLine_A', 'AvgQtyPerLine_B', 'AvgQtyPerLine Δ')
                        ]
                    if not show_avg_price:
                        cols = [c for c in cols if c not in ('AvgPrice_A', 'AvgPrice_B', 'AvgPrice Δ')]
                    display_df = cmp_df[cols]
                    currency_cols = ['Revenue_A', 'Revenue_B', 'Revenue Δ']
                    if has_cost:
                        currency_cols += ['Margin_A', 'Margin_B', 'Margin Δ']
                    if show_avg_price:
                        currency_cols += ['AvgPrice_A', 'AvgPrice_B', 'AvgPrice Δ']
                    show_df(display_df, currency_cols=currency_cols, percent_cols=['Revenue Δ%'])

                only_both = st.checkbox(
                    "Only compare items present in both periods",
                    value=False,
                    key="ta_only_both",
                    help="Excludes customers/products/groups that were only active in Period A or only in Period B."
                )
                presence_label = "both periods" if only_both else "either period"

                tab_cust, tab_prod, tab_group = st.tabs(["Per Customer", "Per Product", "Per Grouping"])

                # ── Per customer ─────────────────────────────────────────────────
                with tab_cust:
                    cust_cmp = _ta_build_comparison(df_a, df_b, 'CustomerId', only_both=only_both)
                    st.markdown(f"**{len(cust_cmp)} customers active in {presence_label}**")
                    _ta_display_table(cust_cmp, show_avg_qty=True)

                    if not cust_cmp.empty:
                        st.markdown("---")
                        st.markdown("**Drill into a single customer**")
                        cust_options = sorted(cust_cmp['CustomerId'].astype(str).unique().tolist())
                        pick_cust = st.selectbox("Customer", cust_options, key="ta_cust_pick")
                        row = cust_cmp[cust_cmp['CustomerId'].astype(str) == pick_cust]
                        if not row.empty:
                            r = row.iloc[0]
                            drill_metrics = [
                                ("Revenue",                r['Revenue_A'],        r['Revenue_B'],        dict(is_currency=True)),
                                ("Orders",                 r['Orders_A'],         r['Orders_B'],         dict(is_currency=False)),
                                ("Quantity",               r['Quantity_A'],       r['Quantity_B'],       dict(is_currency=False)),
                                ("Avg Order Value",        r['AOV_A'],            r['AOV_B'],            dict(is_currency=True)),
                                ("Avg Qty per Order Line", r['AvgQtyPerLine_A'],  r['AvgQtyPerLine_B'],  dict(is_currency=False, decimals=2)),
                            ]
                            if has_cost:
                                drill_metrics.append(("Margin", r['Margin_A'], r['Margin_B'], dict(is_currency=True)))

                            for i in range(0, len(drill_metrics), 3):
                                drow = drill_metrics[i:i + 3]
                                drow_cols = st.columns(len(drow))
                                for col, (label, a_val, b_val, kwargs) in zip(drow_cols, drow):
                                    with col:
                                        _ta_delta_card(label, a_val, b_val, **kwargs)

                # ── Per product ──────────────────────────────────────────────────
                with tab_prod:
                    prod_cmp = _ta_build_comparison(df_a, df_b, 'ProductId', only_both=only_both)
                    prod_cmp = enrich_with_product_name(prod_cmp, fdf, id_col='ProductId')
                    st.markdown(f"**{len(prod_cmp)} products sold in {presence_label}**")
                    _ta_display_table(prod_cmp, show_avg_qty=True, show_avg_price=True)

                # ── Per grouping ─────────────────────────────────────────────────
                with tab_group:
                    if not cat_cols:
                        st.info("No categorical columns available for grouping.")
                    else:
                        group_col_choice = st.selectbox("Group by", cat_cols, key="ta_group_col")
                        group_cmp = _ta_build_comparison(df_a, df_b, group_col_choice, only_both=only_both)
                        st.markdown(f"**{len(group_cmp)} groups active in {presence_label}**")
                        _ta_display_table(group_cmp, show_avg_qty=True)

# ══════════════════════════════════════════════════════════════════════════════
# VIEW 8 — KVI CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "KVI Classification":
    st.markdown('<div class="section-header">KVI Classification</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a8099;margin-top:-0.8rem;margin-bottom:1.2rem;font-size:0.85rem'>"
        "Classifies every product into KVI, Core, or Slow Mover based on demand, revenue, "
        "customer breadth, and basket co-occurrence."
        "</p>", unsafe_allow_html=True
    )
    if agg_col_choice != "— Off (use ProductId) —":
        st.caption(
            f"Note: the sidebar is set to aggregate by **{agg_col_choice}**, but KVI "
            "Classification always runs on the true ProductId."
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
            + cs.get('threshold_label', f"threshold {cs['threshold']:.0%}")
            + f", {cs['n_specialties']} groups across {cs['n_customers']} customers."
        )
        seg_source = st.radio(
            "Run on", ["Confirmed specialty segmentation", "Custom group", "All customers"],
            horizontal=True, key="kvi_seg_source"
        )
    else:
        st.caption("No segmentation confirmed yet. Go to Customer Specialty → Customer List or Basket Segmentation to confirm one, or select a group manually below.")
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
        group_vals = sorted(fdf_raw[group_col_kvi].dropna().astype(str).unique().tolist())
        selected_group_val = st.selectbox("Select group", group_vals, key="kvi_group_val")

    with st.spinner("Running KVI classification..."):
        if seg_source == "Confirmed specialty segmentation" and has_confirmed and selected_group_val:
            confirmed_labels = st.session_state['confirmed_specialty']['labels'].copy()
            if 'Specialty' in confirmed_labels.columns and 'Customer Cluster' not in confirmed_labels.columns:
                confirmed_labels = confirmed_labels.rename(columns={'Specialty': 'Customer Cluster'})
            group_customers = confirmed_labels[
                confirmed_labels['Customer Cluster'] == selected_group_val
            ]['CustomerId'].unique()
            kvi_input = fdf_raw[fdf_raw['CustomerId'].isin(group_customers)]
            scope_label = f"Specialty = {selected_group_val}"

        elif seg_source == "Custom group" and group_col_kvi and selected_group_val:
            group_customers = fdf_raw[fdf_raw[group_col_kvi].astype(str) == selected_group_val]['CustomerId'].unique()
            kvi_input = fdf_raw[fdf_raw['CustomerId'].isin(group_customers)]
            scope_label = f"{group_col_kvi} = {selected_group_val}"

        else:
            kvi_input = fdf_raw
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
        sub['Demand_Proportion']         = sub['Demand_Proportion'].map('{:.2%}'.format)
        sub['Revenue_Proportion']        = sub['Revenue_Proportion'].map('{:.2%}'.format)
        sub['UniqueCustomers_Proportion'] = sub['UniqueCustomers_Proportion'].map('{:.2%}'.format)
        sub['KVI_Score']                 = sub['KVI_Score'].map('{:.3f}'.format)
        sub['Corr_Score']                = sub['Corr_Score'].map('{:.4f}'.format)
        show_df(
            enrich_with_product_name(
                sub[['ProductId', 'Quantity', 'Price', 'Revenue', 'UniqueCustomers',
                     'PurchaseCount', 'Demand_Proportion', 'Revenue_Proportion',
                     'UniqueCustomers_Proportion', 'Corr_Score', 'KVI_Score']],
                fdf_raw
            ),
            currency_cols=['Price', 'Revenue']
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
        show_df(enrich_with_product_name(score_df, fdf_raw))

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
                        group_input = fdf_raw[fdf_raw['CustomerId'].isin(group_customers)]
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
# VIEW 9 — PRICING SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif analysis == "Pricing Simulation":
    st.markdown('<div class="section-header">Pricing Simulation</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#7a8099;margin-top:-0.8rem;margin-bottom:1.2rem;font-size:0.85rem'>"
        "Set price change rules per segment × product category. Compare simulated revenue "
        "and margin against the baseline for a selected time period."
        "</p>", unsafe_allow_html=True
    )
    if agg_col_choice != "— Off (use ProductId) —":
        st.caption(
            f"Note: the sidebar is set to aggregate by **{agg_col_choice}**, but Pricing "
            "Simulation always runs on the true ProductId."
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
    "Product category column (columns) — optional",
    ["— None (blanket rate) —", "KVI Classification"] + cat_cols,
    key="sim_kvi_col",
    help="Leave as 'None' for a single blanket rate per segment row, or pick a column to break down by product category"
)
    with col_s3:
        # Date range for baseline
        if 'CreatedDate' in fdf_raw.columns:
            min_d = fdf_raw['CreatedDate'].min().date()
            max_d = fdf_raw['CreatedDate'].max().date()
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
        sim_df    = fdf_raw[(fdf_raw['CreatedDate'] >= sim_start) & (fdf_raw['CreatedDate'] <= sim_end)].copy()
    else:
        sim_df = fdf_raw.copy()

    # ── Check required cost column ─────────────────────────────────────────────
    if 'TotalCostPerUnit' not in sim_df.columns:
        st.error("TotalCostPerUnit column not found in data. Margin calculation requires this column.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def compute_simulation_baseline(sim_df_hash, seg_col, kvi_col):
        _df = sim_df_hash.copy()
        _df['LineMargin'] = (_df['PricePerUnit'] - _df['TotalCostPerUnit']) * _df['Quantity']

        if kvi_col == "— None (blanket rate) —":
            _df['ProductCategory'] = 'All Products'
            cats = ['All Products']
        elif kvi_col == "KVI Classification":
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
        label = "**% Change**" if kvi_col_sim == "— None (blanket rate) —" else f"**{cat}**"
        header_cols[i + 1].markdown(label)

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
        seg_results['Revenue∆%']    = seg_results['RevenueDelta'] / seg_results['BaseRevenue'] * 100
        seg_results['Margin∆%']     = seg_results['MarginDelta']  / seg_results['BaseMargin']  * 100

        seg_display = seg_results.copy()
        show_df(
            seg_display,
            currency_cols=['BaseRevenue','NewRevenue','RevenueDelta','BaseMargin','NewMargin','MarginDelta'],
            percent_cols=['Revenue∆%', 'Margin∆%']
        )

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
        cat_results['Revenue∆%']    = cat_results['RevenueDelta'] / cat_results['BaseRevenue'] * 100
        cat_results['Margin∆%']     = cat_results['MarginDelta']  / cat_results['BaseMargin']  * 100

        cat_display = cat_results.copy()
        show_df(
            cat_display,
            currency_cols=['BaseRevenue','NewRevenue','RevenueDelta','BaseMargin','NewMargin','MarginDelta'],
            percent_cols=['Revenue∆%', 'Margin∆%']
        )

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
