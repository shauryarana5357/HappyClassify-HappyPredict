
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve


def render_regression():

    # =================================================================
    # THEME (LIGHT / DARK TOGGLE) — identical to HappyClassify
    # =================================================================
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    def toggle_theme():
        st.session_state.dark_mode = not st.session_state.dark_mode

    THEME = {
        True: {
            "bg1": "#0B0E1A", "bg2": "#151A2E", "bg3": "#1B1033",
            "card": "rgba(27, 31, 39, 0.65)", "card_border": "rgba(126, 232, 250, 0.25)",
            "text": "#F5F7FF", "accent": "#7EE8FA", "accent2": "#B388FF",
            "glow": "rgba(126, 232, 250, 0.45)", "plotly_template": "plotly_dark",
        },
        False: {
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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
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
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), automargin=True,
                           gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.3)"),
                yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), automargin=True,
                           gridcolor="rgba(255,255,255,0.15)", zerolinecolor="rgba(255,255,255,0.3)"),
                margin=dict(l=50, r=30, t=50, b=90),
                legend=dict(font=dict(color="white")),
                coloraxis_colorbar=dict(tickfont=dict(color="white"), title_font=dict(color="white")),
            )
        fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=True,
                         landcolor="rgba(128,128,128,0.15)" if st.session_state.dark_mode else "rgba(0,0,0,0.05)")
        fig.update_polars(bgcolor="rgba(0,0,0,0)")
        return fig


    CLASS_ORDER = ["Unhappy", "Moderately Happy", "Happy", "Very Happy"]
    CLASS_COLORS = {"Unhappy": "#EF476F", "Moderately Happy": "#FFD166",
                    "Happy": "#06D6A0", "Very Happy": "#118AB2"}
    REGION_COLORS = px.colors.qualitative.Bold
    PLOTLY_TEMPLATE = t["plotly_template"]


    def score_to_label(score: float) -> str:
        if score < 4:
            return "Unhappy"
        elif score < 6:
            return "Moderately Happy"
        elif score < 7:
            return "Happy"
        else:
            return "Very Happy"


    def card_header(emoji, number, title):
        st.markdown(f"<div class='eda-card'><h4><span class='card-icon'>{emoji}</span> {number}. {title}</h4>",
                    unsafe_allow_html=True)

    def card_footer():
        st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # LOAD DATA & MODEL BUNDLE
    # =================================================================
    @st.cache_data
    def load_data():
        df = pd.read_csv("cleaned_dataset.csv")
        if "Region" not in df.columns:
            import country_converter as coco
            df["Region"] = coco.convert(names=df["Country name"].tolist(), to="continent", not_found="Other")
        return df

    @st.cache_resource
    def load_model():
        return joblib.load("happypredict_model.pkl")

    @st.cache_data
    def load_comparison():
        try:
            return pd.read_csv("model_comparison_regression.csv")
        except FileNotFoundError:
            return None

    df = load_data()
    bundle = load_model()
    models = bundle["models"]
    best_model_name = bundle["best_model_name"]
    model = models[best_model_name]
    scaler = bundle["scaler"]
    features = bundle["features"]
    countries = bundle["countries"]
    comparison_df = load_comparison()

    # =================================================================
    # SIDEBAR
    # =================================================================
    with st.sidebar:
        st.markdown("## 📈 <span class='gradient-text'>HappyPredict</span>", unsafe_allow_html=True)
        st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode",
                  on_click=toggle_theme, use_container_width=True)
        st.markdown("---")
        st.caption("Machine Learning-Based Regression of World Happiness Scores From Socio-Economic Indicators")
        st.caption("Dr. B. R. Ambedkar NIT Jalandhar — Summer Internship Project By Shaurya Rana under Dr. Rajneesh Rani")
        st.markdown("---")
        st.metric("Countries", df["Country name"].nunique())
        st.metric("Years", f"{df['year'].min()}–{df['year'].max()}")
        st.metric("Best model", best_model_name, f"R² {bundle['test_r2'][best_model_name]:.3f}")

    # =================================================================
    # HEADER + TABS
    # =================================================================
    st.markdown("# 📈 HappyPredict Dashboard", unsafe_allow_html=True)
    st.caption("Explore the World Happiness Report, compare regression models, and predict happiness scores interactively.")

    tab_overview, tab_eda, tab_models, tab_predict = st.tabs(
        ["🏠 Overview", "📊 EDA Gallery", "🏆 Model Comparison", "🎯 Predict"]
    )

    # =================================================================
    # TAB 1: OVERVIEW
    # =================================================================
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Countries", df["Country name"].nunique())
        col2.metric("Years covered", f"{df['year'].min()}–{df['year'].max()}")
        col3.metric("Total records", len(df))
        col4.metric(f"Best model: {best_model_name}", f"R² {bundle['test_r2'][best_model_name]:.3f}")

        st.markdown("""
        This project predicts a **continuous happiness score** (World Happiness Report "Life Ladder", 0–10)
        from the same socio-economic indicators used in HappyClassify (GDP per capita, social support,
        healthy life expectancy, freedom, generosity, corruption perception, positive/negative affect).
        """)

        band_table = pd.DataFrame({
            "Band": CLASS_ORDER,
            "Life Ladder Range": ["score < 4", "4 ≤ score < 6", "6 ≤ score < 7", "score ≥ 7"],
            "Color": ["🔴", "🟡", "🟢", "🔵"],
        })
        st.dataframe(band_table, use_container_width=True, hide_index=True)

        st.subheader("Data sample")
        st.dataframe(df.head(10), use_container_width=True)

    # =================================================================
    # TAB 2: EDA GALLERY
    # =================================================================
    with tab_eda:
        st.markdown("### 📊 Exploratory Data Analysis Gallery")
        st.caption("Nine perspectives on the World Happiness Report dataset — distribution, correlation, "
                   "geography, and composition, all in one place.")

        CHART_H = 480

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

            pair_features = {
                "Log GDP per capita": "GDP", "Social support": "Support",
                "Healthy life expectancy at birth": "LifeExp", "Freedom to make life choices": "Freedom"
            }
            df_renamed = df.rename(columns=pair_features)
            selected_features = list(pair_features.values())
            fig = px.scatter_matrix(df_renamed, dimensions=selected_features, color="Life Ladder",
                                     color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE, opacity=0.55)
            fig.update_traces(diagonal_visible=False, marker=dict(size=3))
            fig.update_layout(height=CHART_H + 60, font=dict(size=11),
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

            return figs

        eda_figs = build_eda_figures(df)

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

        # ---- 4. Scatter Plot — each feature vs Happiness Score (interactive, stays uncached) ----
        with row2c2:
            card_header("🎯", 4, "Scatter Plot (feature vs Happiness Score)")
            scatter_feat = st.selectbox("Choose feature", features, key="eda_scatter_feat")
            fig = px.scatter(df, x=scatter_feat, y="Life Ladder", color_discrete_sequence=["#3B6FE0"],
                              template=PLOTLY_TEMPLATE, trendline="ols",
                              trendline_color_override="#EF476F", opacity=0.5)
            fig.update_layout(height=CHART_H - 40, showlegend=False)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            card_footer()

        # ---- 5. Pair Plot (scatter matrix) ----
        with row3c1:
            card_header("🔬", 5, "Pair Plot")
            st.plotly_chart(eda_figs["pair"], use_container_width=True)
            card_footer()

        # ---- 6. Bar Plot — Top 10 Countries ----
        with row3c2:
            card_header("📶", 6, "Bar Plot (Top 10 Countries)")
            st.plotly_chart(eda_figs["top10"], use_container_width=True)
            card_footer()

        # ---- 7. World Map ----
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
    # TAB 3: MODEL COMPARISON + PER-MODEL DIAGNOSTICS
    # =================================================================
    with tab_models:
        st.markdown("### Model Comparison Table")
        if comparison_df is not None:
            styled = comparison_df.style.background_gradient(
                subset=["MAE", "MSE", "RMSE", "R2"], cmap="Greens_r"
            ).format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}",
                      "R2": "{:.3f}", "Training Time (s)": "{:.3f}"})
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("### R² Score by Model")
            fig = px.bar(comparison_df.sort_values("R2"), x="Model", y="R2", color="Model",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         template=PLOTLY_TEMPLATE, text_auto=".3f")
            fig.update_layout(showlegend=False, xaxis_tickangle=-15, height=440, xaxis=dict(automargin=True))
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Training Time by Model")
            fig = px.bar(comparison_df.sort_values("Training Time (s)"), x="Model", y="Training Time (s)",
                         color="Model", color_discrete_sequence=px.colors.qualitative.Bold,
                         template=PLOTLY_TEMPLATE, text_auto=".3f")
            fig.update_layout(showlegend=False, xaxis_tickangle=-15, height=440, xaxis=dict(automargin=True))
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("model_comparison_regression.csv not found — run the training notebook first.")

        st.markdown("---")
        diag_model_name = st.selectbox("🔍 Check diagnostics for model", list(models.keys()),
                                        index=list(models.keys()).index(best_model_name))
        st.info("Computed on the full dataset for illustration, since the model bundle doesn't store "
                "the original held-out test split. Treat these as indicative, not the official test-set metrics.")

        @st.cache_data
        def compute_reg_diagnostics(model_name):
            diag_model = models[model_name]
            X_full = df[features]
            X_full_scaled = scaler.transform(X_full)
            y_true = df["Life Ladder"].to_numpy()
            y_pred = diag_model.predict(X_full_scaled)

            DIAG_H = 420
            out = {}

            fig = px.scatter(x=y_pred, y=y_true, labels={"x": "Predicted", "y": "Actual"},
                              color_discrete_sequence=["#3B6FE0"], template=PLOTLY_TEMPLATE, opacity=0.4)
            lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                      line=dict(color="#EF476F", dash="dash"), name="Ideal"))
            fig.update_layout(height=DIAG_H, showlegend=False)
            out["avp"] = apply_theme(fig)

            residuals = y_true - y_pred
            fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residuals"},
                              color_discrete_sequence=["#06D6A0"], template=PLOTLY_TEMPLATE, opacity=0.4)
            fig.add_hline(y=0, line_dash="dash", line_color="#EF476F")
            fig.update_layout(height=DIAG_H, showlegend=False)
            out["resid"] = apply_theme(fig)

            if hasattr(diag_model, "feature_importances_"):
                importances = diag_model.feature_importances_
            elif hasattr(diag_model, "coef_"):
                importances = np.abs(diag_model.coef_)
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

            train_sizes, train_scores, val_scores = learning_curve(
                diag_model, X_full_scaled, y_true, cv=3, scoring="r2",
                train_sizes=np.linspace(0.2, 1.0, 6), random_state=42
            )
            lc_df = pd.DataFrame({
                "Training Samples": np.concatenate([train_sizes, train_sizes]),
                "R² Score": np.concatenate([train_scores.mean(axis=1), val_scores.mean(axis=1)]),
                "Set": ["Training"] * len(train_sizes) + ["Validation"] * len(train_sizes),
            })
            fig = px.line(lc_df, x="Training Samples", y="R² Score", color="Set", markers=True,
                          color_discrete_map={"Training": "#118AB2", "Validation": "#EF476F"},
                          template=PLOTLY_TEMPLATE)
            fig.update_layout(height=DIAG_H, legend=dict(orientation="h", y=-0.2))
            out["learning_curve"] = apply_theme(fig)

            return out

        with st.spinner(f"Computing diagnostics for {diag_model_name} (first time only, then cached)..."):
            diag = compute_reg_diagnostics(diag_model_name)

        diag_c1, diag_c2, diag_c3, diag_c4 = st.columns(4)

        # ---- A. Actual vs Predicted ----
        with diag_c1:
            card_header("🎯", "A", "Actual vs Predicted")
            st.plotly_chart(diag["avp"], use_container_width=True)
            card_footer()

        # ---- B. Residual Error Plot ----
        with diag_c2:
            card_header("📉", "B", "Residual Error Plot")
            st.plotly_chart(diag["resid"], use_container_width=True)
            card_footer()

        # ---- C. Feature Importance ----
        with diag_c3:
            card_header("🌟", "C", f"Feature Importance ({diag_model_name})")
            if diag["importance"] is not None:
                st.plotly_chart(diag["importance"], use_container_width=True)
            else:
                st.warning(f"{diag_model_name} doesn't expose feature importances (e.g. SVR with an RBF kernel).")
            card_footer()

        # ---- D. Learning Curve ----
        with diag_c4:
            card_header("📈", "D", "Learning Curve")
            st.plotly_chart(diag["learning_curve"], use_container_width=True)
            card_footer()

    # =================================================================
    # TAB 4: INTERACTIVE PREDICTION DASHBOARD
    # =================================================================
    with tab_predict:
        st.markdown("## 🎯 Interactive Prediction Dashboard")
        st.subheader("Input Features (adjust the values below)")

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
            selected_model_name = st.selectbox(
                "Choose model for prediction", list(models.keys()),
                index=list(models.keys()).index(best_model_name)
            )
            selected_model = models[selected_model_name]

            st.markdown("---")
            user_values = {}
            for feat in features:
                lo, hi, default, step = ranges[feat]
                user_values[feat] = st.slider(
                    f"{feat}  ({lo:g} – {hi:g})", min_value=float(lo), max_value=float(hi),
                    value=float(default), step=float(step)
                )

            selected_ref_country = st.selectbox("Country (optional, for reference only)", ["—"] + countries)
            predict_clicked = st.button("▶ Predict Happiness Score", type="primary", use_container_width=True)

        with col_right:
            if predict_clicked:
                X_input = pd.DataFrame([user_values])[features]
                X_scaled = scaler.transform(X_input)
                pred_score = float(selected_model.predict(X_scaled)[0])
                pred_score = max(0.0, min(10.0, pred_score))
                pred_label = score_to_label(pred_score)

                st.markdown("### Predicted Happiness Score")
                st.markdown(
                    f"<h1 style='color:{CLASS_COLORS[pred_label]}; margin-bottom:0;'>{pred_score:.2f}</h1>"
                    f"<p style='margin-top:0; font-size:18px;'>{pred_label}</p>",
                    unsafe_allow_html=True
                )

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pred_score,
                    number={"suffix": " / 10", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 10]},
                        "bar": {"color": CLASS_COLORS[pred_label]},
                        "steps": [
                            {"range": [0, 4], "color": "rgba(239,71,111,0.25)"},
                            {"range": [4, 6], "color": "rgba(255,209,102,0.25)"},
                            {"range": [6, 7], "color": "rgba(6,214,160,0.25)"},
                            {"range": [7, 10], "color": "rgba(17,138,178,0.25)"},
                        ],
                    }
                ))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(l=30, r=30, t=30, b=10))
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                r2 = bundle["test_r2"][selected_model_name]
                rmse = bundle["test_rmse"][selected_model_name]
                st.caption(f"Model used: {selected_model_name}  ·  test R²: {r2:.3f}  ·  test RMSE: {rmse:.3f}")
                if selected_ref_country != "—":
                    st.caption(f"Reference country selected: {selected_ref_country} (not used as a model input)")

                st.markdown("---")
                st.markdown(
                    "**About the prediction**  \n"
                    "The model predicts a continuous happiness score (World Happiness Report \"Life Ladder\" scale, "
                    "0–10) from the input socio-economic indicators. The colour bands on the gauge mark the same "
                    "four happiness levels used in HappyClassify: 🔴 Unhappy (<4) · 🟡 Moderately Happy (4–6) · "
                    "🟢 Happy (6–7) · 🔵 Very Happy (≥7)."
                )
            else:
                st.info("Adjust the sliders on the left and click **Predict Happiness Score**.")

    st.markdown("---")
    st.caption("HappyPredict · World Happiness Report 2005–2023 · Built with Streamlit, scikit-learn & Plotly")