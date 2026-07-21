

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def render_classify():

    # =================================================================
    # THEME (LIGHT / DARK TOGGLE)
    # =================================================================
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_theme():
        st.session_state.dark_mode = not st.session_state.dark_mode

    THEME = {
        True: {  # dark — deep space / neon aurora
            "bg1": "#0B0E1A", "bg2": "#151A2E", "bg3": "#1B1033",
            "card": "rgba(27, 31, 39, 0.65)", "card_border": "rgba(126, 232, 250, 0.25)",
            "text": "#F5F7FF", "accent": "#7EE8FA", "accent2": "#B388FF",
            "glow": "rgba(126, 232, 250, 0.45)", "plotly_template": "plotly_dark",
        },
        False: {  # light — soft dawn / lavender sky
            "bg1": "#F3F6FF", "bg2": "#EAF0FF", "bg3": "#F7ECFF",
            "card": "rgba(255, 255, 255, 0.75)", "card_border": "rgba(90, 100, 200, 0.18)",
            "text": "#151626", "accent": "#4A5CD6", "accent2": "#B23FA8",
            "glow": "rgba(74, 92, 214, 0.28)", "plotly_template": "plotly_white",
        },
    }
    t = THEME[st.session_state.dark_mode]
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', 'Poppins', sans-serif; }}

    .stApp {{
        background: linear-gradient(135deg, {t['bg1']} 0%, {t['bg2']} 45%, {t['bg3']} 100%);
        color: {t['text']};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['bg2']} 0%, {t['bg3']} 100%);
        border-right: 1px solid {t['card_border']};
    }}

    h1 {{
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }}
    h2, h3, h4, p, span, label, .stMarkdown {{ color: {t['text']} !important; }}
    h2, h3, h4 {{ font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; }}

    div[data-testid="stMetric"] {{
        background: {t['card']};
        backdrop-filter: blur(10px);
        border-radius: 14px; padding: 14px;
        border: 1px solid {t['card_border']};
    }}
    div[data-testid="stMetricValue"] {{ color: {t['text']} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {t['text']} !important; opacity: 0.85; }}
    div[data-testid="stMetricValue"] > div {{ color: {t['text']} !important; }}

    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
        color: #0B0E1A !important;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.55em 1.2em;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        filter: brightness(1.08);
    }}

    .stTabs [data-baseweb="tab"] {{
        font-size: 15px; font-weight: 600; border-radius: 10px 10px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']}) !important;
        color: #0B0E1A !important;
    }}

    .eda-card {{
        background: {t['card']};
        backdrop-filter: blur(10px);
        border-radius: 16px; padding: 14px 16px 4px 16px;
        border: 1px solid {t['card_border']};
        margin-bottom: 16px;
    }}
    .eda-card h4 {{ margin-top: 0; margin-bottom: 6px; }}

    div[data-testid="stDataFrame"] {{
        border-radius: 12px; overflow: hidden; border: 1px solid {t['card_border']};
    }}

    div[data-baseweb="select"] > div, div[data-baseweb="popover"], .stSelectbox div[data-baseweb="select"] {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border-color: {t['card_border']} !important;
    }}
    div[data-baseweb="popover"] li, ul[role="listbox"] li {{
        color: {t['text']} !important;
        background-color: {t['card']} !important;
    }}
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stCaptionContainer"] {{
        color: {t['text']} !important;
    }}
    .stAlert, div[data-testid="stNotification"] {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {t['accent']}, {t['accent2']});
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-track {{ background: transparent; }}

    .gradient-text {{
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 800;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Lighten Streamlit's default top decoration bar — light mode only, dark mode untouched
    if not st.session_state.dark_mode:
        st.markdown("""
        <style>
        div[data-testid="stDecoration"] {
            background-image: linear-gradient(90deg, #C7D2FE, #DDD6FE) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Disable Streamlit's default dimming/fade-out while a rerun is in progress
    st.markdown("""
    <style>
    [data-stale="true"] {
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    def apply_theme(fig):
        if not st.session_state.dark_mode:
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="black"),
                xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), automargin=True,
                           gridcolor="rgba(0,0,0,0.12)", zerolinecolor="rgba(0,0,0,0.25)"),
                yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"), automargin=True,
                           gridcolor="rgba(0,0,0,0.12)", zerolinecolor="rgba(0,0,0,0.25)"),
                margin=dict(l=50, r=30, t=50, b=90),
                legend=dict(font=dict(color="black")),
                coloraxis_colorbar=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
            )
        else:
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), automargin=True,
                           gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.3)"),
                yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), automargin=True,
                           gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.3)"),
                margin=dict(l=50, r=30, t=50, b=90),
                legend=dict(font=dict(color="white")),
                coloraxis_colorbar=dict(tickfont=dict(color="white"), title_font=dict(color="white")),
            )
        # Geo panels (choropleth) draw their own background separate from paper_bgcolor
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=True,
                         landcolor="rgba(128,128,128,0.15)" if st.session_state.dark_mode else "rgba(0,0,0,0.05)")
        # Polar charts (radar) also have their own background
        fig.update_polars(bgcolor="rgba(0,0,0,0)")
        return fig

    # Colorful, consistent palettes used throughout
    CLASS_ORDER = ["Unhappy", "Moderately Happy", "Happy", "Very Happy"]
    CLASS_COLORS = {"Unhappy": "#EF476F", "Moderately Happy": "#FFD166",
                    "Happy": "#06D6A0", "Very Happy": "#118AB2"}
    CLASS_COLOR_LIST = [CLASS_COLORS[c] for c in CLASS_ORDER]
    REGION_COLORS = px.colors.qualitative.Bold
    PLOTLY_TEMPLATE = t["plotly_template"]

    def card_header(emoji, number, title):
        st.markdown(f"<div class='eda-card'><h4><span class='card-icon'>{emoji}</span> {number}. {title}</h4>",
                    unsafe_allow_html=True)

    def card_footer():
        st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # LOAD DATA & MODEL
    # =================================================================
    @st.cache_data
    def load_data():
        df = pd.read_csv("cleaned_dataset.csv")
        df["Happiness Label"] = pd.Categorical(df["Happiness Label"], categories=CLASS_ORDER, ordered=True)
        if "Region" not in df.columns:
            import country_converter as coco
            df["Region"] = coco.convert(names=df["Country name"].tolist(), to="continent", not_found="Other")
        return df

    @st.cache_resource
    def load_model():
        return joblib.load("happyclassify_model.pkl")

    @st.cache_data
    def load_comparison():
        try:
            return pd.read_csv("model_comparison.csv")
        except FileNotFoundError:
            return None

    df = load_data()
    bundle = load_model()
    models = bundle["models"]
    best_model_name = bundle["best_model_name"]
    model = models[best_model_name]
    scaler = bundle["scaler"]
    features = bundle["features"]
    label_map = bundle["label_map"]
    countries = bundle["countries"]
    comparison_df = load_comparison()

    # =================================================================
    # SIDEBAR
    # =================================================================
    with st.sidebar:
        st.markdown("## <span class='bounce-emoji'>😊</span> <span class='gradient-text'>HappyClassify</span>",
                    unsafe_allow_html=True)
        st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode",
                  on_click=toggle_theme, use_container_width=True)
        st.markdown("---")
        st.caption("Machine Learning-Based Classification of World Happiness Levels Using Socio-Economic Indicators")
        st.caption("Dr. B. R. Ambedkar NIT Jalandhar — Summer Internship Project By Shaurya Rana under Dr. Rajneesh Rani")
        st.markdown("---")
        st.metric("Countries", df["Country name"].nunique())
        st.metric("Years", f"{df['year'].min()}–{df['year'].max()}")
        st.metric("Best model", best_model_name, f"{bundle['test_accuracy'][best_model_name]*100:.1f}% test acc.")

    # =================================================================
    # HEADER + TABS
    # =================================================================
    st.markdown("# <span class='bounce-emoji'>😊</span> HappyClassify Dashboard", unsafe_allow_html=True)
    st.caption("Explore the World Happiness Report, compare models, and predict happiness levels interactively.")

    tab_overview, tab_eda, tab_country, tab_models, tab_predict = st.tabs(
        ["🏠 Overview", "📊 EDA Gallery", "🌍 Country Explorer", "🏆 Model Comparison", "🎯 Predict"]
    )

    # =================================================================
    # TAB 1: OVERVIEW
    # =================================================================
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Countries", df["Country name"].nunique())
        col2.metric("Years covered", f"{df['year'].min()}–{df['year'].max()}")
        col3.metric("Total records", len(df))
        col4.metric(f"Best model: {best_model_name}", f"{bundle['test_accuracy'][best_model_name]*100:.1f}% test accuracy")

        st.markdown("""
        This project classifies countries into **4 happiness levels** based on the World Happiness Report's
        socio-economic indicators (GDP per capita, social support, healthy life expectancy, freedom, generosity,
        corruption perception, positive/negative affect).
        """)

        level_table = pd.DataFrame({
            "Level": [0, 1, 2, 3],
            "Label": CLASS_ORDER,
            "Life Ladder Range": ["score < 4", "4 ≤ score < 6", "6 ≤ score < 7", "score ≥ 7"],
            "Color": ["🔴", "🟡", "🟢", "🔵"],
        })
        st.dataframe(level_table, use_container_width=True, hide_index=True)

        st.subheader("Data sample")
        st.dataframe(df.head(10), use_container_width=True)

    # =================================================================
    # TAB 2: EDA GALLERY — all 9 charts, 2-column grid for bigger, readable plots
    # =================================================================
    with tab_eda:
        st.markdown("### 📊 Exploratory Data Analysis Gallery")
        st.caption("Nine perspectives on the World Happiness Report dataset — distribution, correlation, "
                   "geography, and composition, all in one place.")

        CHART_H = 480  # bigger charts across the board

        @st.cache_data
        def build_eda_figures(df):
            figs = {}
            country_avg = df.groupby("Country name", as_index=False)["Life Ladder"].mean()
            top5_countries = (df.groupby("Country name")["Life Ladder"].mean()
                               .sort_values(ascending=False).head(5).index.tolist())

            fig = px.histogram(df, x="Life Ladder", nbins=30, color_discrete_sequence=["#FF6B6B"],
                                template=PLOTLY_TEMPLATE, marginal="box")
            fig.update_layout(bargap=0.05, height=CHART_H, showlegend=False)
            figs["hist"] = apply_theme(fig)

            fig = px.box(df, x="Region", y="Life Ladder", color="Region",
                         color_discrete_sequence=REGION_COLORS, template=PLOTLY_TEMPLATE)
            fig.update_layout(xaxis_tickangle=-20, showlegend=False, height=CHART_H,
                               xaxis=dict(automargin=True), bargap=0.2)
            figs["box_region"] = apply_theme(fig)

            numeric_cols = ["Life Ladder"] + features
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             template=PLOTLY_TEMPLATE, aspect="auto")
            fig.update_layout(height=CHART_H, xaxis_tickangle=-35, font=dict(size=11))
            figs["heatmap"] = apply_theme(fig)

            counts = df["Happiness Label"].value_counts().reindex(CLASS_ORDER).reset_index()
            counts.columns = ["Happiness Label", "Count"]
            fig = px.bar(counts, x="Happiness Label", y="Count", color="Happiness Label",
                         color_discrete_map=CLASS_COLORS, template=PLOTLY_TEMPLATE, text_auto=True)
            fig.update_layout(showlegend=False, height=CHART_H, xaxis_title="", xaxis_tickangle=0,
                               xaxis=dict(automargin=True))
            figs["count"] = apply_theme(fig)

            pair_features = {
                "Log GDP per capita": "GDP", "Social support": "Support",
                "Healthy life expectancy at birth": "LifeExp", "Freedom to make life choices": "Freedom"
            }
            df_renamed = df.rename(columns=pair_features)
            selected_features = list(pair_features.values())
            fig = px.scatter_matrix(df_renamed, dimensions=selected_features, color="Happiness Label",
                                     color_discrete_map=CLASS_COLORS, template=PLOTLY_TEMPLATE, opacity=0.55)
            fig.update_traces(diagonal_visible=False, marker=dict(size=3))
            fig.update_layout(height=CHART_H + 60, showlegend=True, font=dict(size=11),
                               legend=dict(orientation="h", y=-0.12),
                               xaxis=dict(tickangle=-30, automargin=True), yaxis=dict(automargin=True))
            figs["pair"] = apply_theme(fig)

            top10 = country_avg.sort_values("Life Ladder", ascending=False).head(10).sort_values("Life Ladder")
            fig = px.bar(top10, x="Life Ladder", y="Country name", orientation="h",
                         color_discrete_sequence=["#3B6FE0"], template=PLOTLY_TEMPLATE, text_auto=".2f")
            fig.update_layout(height=CHART_H, xaxis_title="Happiness Score", yaxis_title="",
                               showlegend=False, yaxis=dict(automargin=True))
            figs["top10"] = apply_theme(fig)

            fig = px.choropleth(country_avg, locations="Country name", locationmode="country names",
                                 color="Life Ladder", color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE)
            fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
            fig.update_layout(height=CHART_H, margin=dict(l=0, r=0, t=30, b=0))
            figs["map"] = apply_theme(fig)

            radar_df = df[df["Country name"].isin(top5_countries)].groupby("Country name")[features].mean()
            radar_norm = (radar_df - df[features].min()) / (df[features].max() - df[features].min())
            fig = go.Figure()
            radar_colors = px.colors.qualitative.Set1
            for i, country in enumerate(top5_countries):
                vals = radar_norm.loc[country].tolist()
                vals += vals[:1]
                fig.add_trace(go.Scatterpolar(r=vals, theta=features + [features[0]], fill="toself",
                                               name=country, line=dict(color=radar_colors[i % len(radar_colors)])))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=CHART_H,
                               polar=dict(radialaxis=dict(visible=True, range=[0, 1]),
                                          angularaxis=dict(tickfont=dict(size=9))),
                               legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
            figs["radar"] = apply_theme(fig)

            fig = px.violin(df, x="Region", y="Life Ladder", color="Region", box=True, points=False,
                             color_discrete_sequence=REGION_COLORS, template=PLOTLY_TEMPLATE)
            fig.update_layout(showlegend=False, height=CHART_H, xaxis_tickangle=-20, xaxis=dict(automargin=True))
            figs["violin"] = apply_theme(fig)

            return figs, top5_countries

        eda_figs, top5_countries = build_eda_figures(df)

        row1c1, row1c2 = st.columns(2)
        row2c1, row2c2 = st.columns(2)
        row3c1, row3c2 = st.columns(2)
        row4c1, row4c2 = st.columns(2)
        row5c1, _ = st.columns(2)

        # ---- 1. Histogram — Happiness Score ----
        with row1c1:
            card_header("📈", 1, "Histogram (Happiness Score)")
            st.plotly_chart(eda_figs["hist"], use_container_width=True)
            card_footer()

        # ---- 2. Box Plot — by Region ----
        with row1c2:
            card_header("📦", 2, "Box Plot by Region")
            st.plotly_chart(eda_figs["box_region"], use_container_width=True)
            card_footer()

        # ---- 3. Correlation Heatmap ----
        with row2c1:
            card_header("🔥", 3, "Correlation Heatmap")
            st.plotly_chart(eda_figs["heatmap"], use_container_width=True)
            card_footer()

        # ---- 4. Count Plot — by Happiness Level ----
        with row2c2:
            card_header("📊", 4, "Count Plot (by Happiness Level)")
            st.plotly_chart(eda_figs["count"], use_container_width=True)
            card_footer()

        # ---- 5. Pair Plot (scatter matrix) ----
        with row3c1:
            card_header("🔬", 5, "Pair Plot")
            st.plotly_chart(eda_figs["pair"], use_container_width=True)
            card_footer()

        # ---- 6. Bar Plot — Top 10 Countries by Happiness Score ----
        with row3c2:
            card_header("📶", 6, "Bar Plot (Top 10 Countries)")
            st.plotly_chart(eda_figs["top10"], use_container_width=True)
            card_footer()

        # ---- 7. World Map — Avg Happiness Score ----
        with row4c1:
            card_header("🗺️", 7, "World Map (Avg Happiness Score)")
            st.plotly_chart(eda_figs["map"], use_container_width=True)
            card_footer()

        # ---- 8. Radar Chart — Top 5 Countries ----
        with row4c2:
            card_header("🕸️", 8, "Radar Chart (Top 5 Countries)")
            st.plotly_chart(eda_figs["radar"], use_container_width=True)
            card_footer()

        # ---- 9. Violin Plot — by Region ----
        with row5c1:
            card_header("🎻", 9, "Violin Plot (by Region)")
            st.plotly_chart(eda_figs["violin"], use_container_width=True)
            card_footer()

    # =================================================================
    # TAB 3: COUNTRY EXPLORER (interactive: pick a country, see its trend)
    # =================================================================
    with tab_country:
        st.markdown("### Pick a country to explore its happiness trend")
        col_select, col_info = st.columns([1, 2])
        with col_select:
            selected_country = st.selectbox("Country", countries,
                                             index=countries.index("India") if "India" in countries else 0)
            country_df = df[df["Country name"] == selected_country].sort_values("year")
            region_name = country_df["Region"].iloc[0] if len(country_df) else "—"
            st.metric("Region", region_name)
            st.metric("Avg Happiness (Life Ladder)", f"{country_df['Life Ladder'].mean():.2f}")
            st.metric("Latest Happiness Level", country_df.sort_values('year').iloc[-1]["Happiness Label"]
                       if len(country_df) else "—")

        with col_info:
            fig = px.line(country_df, x="year", y="Life Ladder", markers=True,
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=["#118AB2"],
                          title=f"{selected_country} — Happiness Score Over Time")
            region_avg = df[df["Region"] == region_name].groupby("year")["Life Ladder"].mean().reset_index()
            fig.add_trace(go.Scatter(x=region_avg["year"], y=region_avg["Life Ladder"],
                                      mode="lines", name=f"{region_name} avg",
                                      line=dict(dash="dash", color="#FFD166")))
            fig.update_layout(height=480, xaxis=dict(automargin=True, dtick=1))
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {selected_country}'s socio-economic profile vs. global average (latest year)")
        latest_year = country_df["year"].max()
        latest_row = country_df[country_df["year"] == latest_year][features].mean()
        global_avg = df[df["year"] == latest_year][features].mean()

        compare_df = pd.DataFrame({
            "Feature": features,
            selected_country: latest_row.values,
            "Global Average": global_avg.values,
        }).melt(id_vars="Feature", var_name="Series", value_name="Value")

        fig = px.bar(compare_df, x="Feature", y="Value", color="Series", barmode="group",
                     color_discrete_sequence=["#EF476F", "#118AB2"], template=PLOTLY_TEMPLATE)
        fig.update_layout(xaxis_tickangle=-20, height=520, xaxis=dict(automargin=True))
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # TAB 4: MODEL COMPARISON TABLE + DIAGNOSTICS
    # =================================================================
    with tab_models:
        st.markdown("### Model Comparison Table")
        if comparison_df is not None:
            styled = comparison_df.style.background_gradient(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                cmap="Greens"
            ).format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
                      "F1-Score": "{:.3f}", "ROC-AUC": "{:.3f}", "Training Time (s)": "{:.3f}"})
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("### Accuracy & F1-Score by Model")
            melt_df = comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score", "ROC-AUC"],
                                          var_name="Metric", value_name="Score")
            fig = px.bar(melt_df, x="Model", y="Score", color="Metric", barmode="group",
                         color_discrete_sequence=["#118AB2", "#06D6A0", "#FFD166"],
                         template=PLOTLY_TEMPLATE, range_y=[0, 1])
            fig.update_layout(xaxis_tickangle=-15, height=480, xaxis=dict(automargin=True))
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Training Time by Model")
            fig = px.bar(comparison_df.sort_values("Training Time (s)"), x="Model", y="Training Time (s)",
                         color="Model", color_discrete_sequence=px.colors.qualitative.Bold,
                         template=PLOTLY_TEMPLATE, text_auto=".3f")
            fig.update_layout(showlegend=False, xaxis_tickangle=-15, height=480, xaxis=dict(automargin=True))
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("model_comparison.csv not found — run train_models.py first.")

        st.markdown("---")
        diag_model_name = st.selectbox(
            "🔍 Check diagnostics for model",
            list(models.keys()),
            index=list(models.keys()).index(best_model_name),
            key="classify_diag_model_select"
        )
        st.info("Computed on the full dataset for illustration, since the model bundle doesn't store "
                "the original held-out test split. Treat these as indicative, not the official test-set metrics.")

        @st.cache_data
        def compute_diagnostics(model_name):
            diag_model = models[model_name]
            X_full = df[features]
            X_full_scaled = scaler.transform(X_full)
            y_true = df["Happiness Label"].cat.codes.to_numpy()
            y_pred = diag_model.predict(X_full_scaled)
            y_proba = diag_model.predict_proba(X_full_scaled) if hasattr(diag_model, "predict_proba") else None

            DIAG_H = 460
            out = {}

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=[0, 1, 2, 3], y=[0, 1, 2, 3], template=PLOTLY_TEMPLATE)
            fig.update_layout(height=DIAG_H, coloraxis_showscale=False)
            out["cm"] = apply_theme(fig)

            if y_proba is not None:
                y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
                fig = go.Figure()
                for i in range(4):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                              name=f"Class {i} (AUC={roc_auc:.2f})",
                                              line=dict(color=CLASS_COLOR_LIST[i])))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                          line=dict(dash="dash", color="gray")))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=DIAG_H,
                                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                   legend=dict(font=dict(size=10), orientation="h", y=-0.25))
                out["roc"] = apply_theme(fig)
            else:
                out["roc"] = None

            if hasattr(diag_model, "feature_importances_"):
                importances = diag_model.feature_importances_
            elif hasattr(diag_model, "coef_"):
                importances = np.abs(diag_model.coef_).mean(axis=0)
            else:
                importances = None
            if importances is not None:
                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
                    "Importance", ascending=True)
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues", template=PLOTLY_TEMPLATE)
                fig.update_layout(height=DIAG_H, coloraxis_showscale=False, yaxis=dict(automargin=True))
                out["importance"] = apply_theme(fig)
            else:
                out["importance"] = None

            class_counts = df["Happiness Label"].value_counts().reindex(CLASS_ORDER)
            fig = px.pie(names=class_counts.index, values=class_counts.values,
                         color=class_counts.index, color_discrete_map=CLASS_COLORS,
                         template=PLOTLY_TEMPLATE, hole=0.35)
            fig.update_layout(height=DIAG_H, legend=dict(font=dict(size=11), orientation="h", y=-0.15))
            out["pie"] = apply_theme(fig)

            return out

        diag = compute_diagnostics(diag_model_name)

        diag_r1c1, diag_r1c2 = st.columns(2)
        diag_r2c1, diag_r2c2 = st.columns(2)

        # ---- A. Confusion Matrix ----
        with diag_r1c1:
            card_header("🧮", "A", f"Confusion Matrix ({diag_model_name})")
            st.plotly_chart(diag["cm"], use_container_width=True)
            card_footer()

        # ---- B. ROC Curve (One-vs-Rest) ----
        with diag_r1c2:
            card_header("📉", "B", f"ROC Curve — One-vs-Rest ({diag_model_name})")
            if diag["roc"] is not None:
                st.plotly_chart(diag["roc"], use_container_width=True)
            else:
                st.warning(f"{diag_model_name} has no predict_proba — ROC curve unavailable.")
            card_footer()

        # ---- C. Feature Importance ----
        with diag_r2c1:
            card_header("🌟", "C", f"Feature Importance ({diag_model_name})")
            if diag["importance"] is not None:
                st.plotly_chart(diag["importance"], use_container_width=True)
            else:
                st.warning(f"{diag_model_name} doesn't expose feature importances.")
            card_footer()

        # ---- D. Class Distribution ----
        with diag_r2c2:
            card_header("🥧", "D", "Class Distribution")
            st.plotly_chart(diag["pie"], use_container_width=True)
            card_footer()

    # =================================================================
    # TAB 5: INTERACTIVE PREDICTION DASHBOARD
    # =================================================================
    with tab_predict:
        st.subheader("Adjust the socio-economic indicators below")

        selected_model_name = st.selectbox(
            "Choose model for prediction",
            list(models.keys()),
            index=list(models.keys()).index(best_model_name)
        )
        selected_model = models[selected_model_name]

        ranges = {
            "Log GDP per capita": (5.5, 12.0, 9.4, 0.01),
            "Social support": (0.0, 1.0, 0.81, 0.01),
            "Healthy life expectancy at birth": (20.0, 85.0, 63.5, 0.5),
            "Freedom to make life choices": (0.0, 1.0, 0.75, 0.01),
            "Generosity": (-0.5, 0.6, 0.0, 0.01),
            "Perceptions of corruption": (0.0, 1.0, 0.75, 0.01),
            "Positive affect": (0.0, 1.0, 0.65, 0.01),
            "Negative affect": (0.0, 1.0, 0.27, 0.01),
        }

        col_left, col_right = st.columns([1, 1])

        with col_left:
            user_values = {}
            for feat in features:
                lo, hi, default, step = ranges[feat]
                user_values[feat] = st.slider(feat, min_value=float(lo), max_value=float(hi),
                                               value=float(default), step=float(step))
            selected_ref_country = st.selectbox("Country (optional, for reference only)", ["—"] + countries)
            predict_clicked = st.button("▶ Predict Happiness Level", type="primary", use_container_width=True)

        with col_right:
            if predict_clicked:
                X_input = pd.DataFrame([user_values])[features]
                X_scaled = scaler.transform(X_input)
                pred_class = selected_model.predict(X_scaled)[0]
                pred_proba = selected_model.predict_proba(X_scaled)[0] if hasattr(selected_model, "predict_proba") else None

                pred_label = label_map[pred_class]
                st.markdown("### Predicted Happiness Level")
                st.markdown(
                    f"<h2 style='color:{CLASS_COLORS[pred_label]}'>{pred_class} — {pred_label}</h2>",
                    unsafe_allow_html=True
                )

                if pred_proba is not None:
                    proba_df = pd.DataFrame({
                        "Level": [f"{i} - {label_map[i]}" for i in range(4)],
                        "Probability": pred_proba
                    })
                    fig = px.bar(proba_df, x="Level", y="Probability", text_auto=".2f",
                                 color="Level",
                                 color_discrete_sequence=CLASS_COLOR_LIST,
                                 template=PLOTLY_TEMPLATE)
                    fig.update_layout(yaxis_range=[0, 1], showlegend=False, height=420,
                                       xaxis=dict(automargin=True, tickangle=-10))
                    fig = apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{selected_model_name} doesn't support probability outputs, showing class only.")

                acc = bundle["test_accuracy"][selected_model_name]
                st.caption(f"Model used: {selected_model_name} (test accuracy: {acc*100:.1f}%)")
                if selected_ref_country != "—":
                    st.caption(f"Reference country selected: {selected_ref_country} (not used as a model input)")
            else:
                st.info("Adjust the sliders on the left and click **Predict Happiness Level**.")

    st.markdown("---")
    st.caption("HappyClassify · World Happiness Report 2005–2023 · Built with Streamlit, scikit-learn & Plotly")