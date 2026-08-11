# ============================================================
# TANGERANG RESIDENTIAL AVM
# Production Streamlit Application
# ============================================================

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tangerang Residential AVM",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

ENGINE_PATH = BASE_DIR / "avm_engine.pkl"
DATA_PATH = BASE_DIR / "avm_data.csv"
CONFIG_PATH = BASE_DIR / "avm_config.json"


# ============================================================
# LOAD DEPLOYMENT FILES
# ============================================================

@st.cache_resource
def load_engine():

    with open(ENGINE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


engine = load_engine()
df_prod = load_data()
config = load_config()


# ============================================================
# RESTORE PRODUCTION COMPONENTS
# ============================================================

n_imputer = engine["n_imputer"]
n_scaler = engine["n_scaler"]

X_n_imputed = engine["X_n_imputed"]
X_n_scaled = engine["X_n_scaled"]
y_n_log = engine["y_n_log"]

model_n_features = engine["model_n_features"]

MODEL_N_NEIGHBORS = engine["n_neighbors"]
MODEL_N_TREES = engine["n_trees"]

FINAL_WEIGHT_N = engine["weight_model_n"]
FINAL_WEIGHT_COMP = engine["weight_comparable"]


# ============================================================
# HELPER — FORMAT RUPIAH
# ============================================================

def rupiah(value):

    if pd.isna(value):
        return "-"

    return f"Rp {value:,.0f}".replace(",", ".")


# ============================================================
# MODEL N — EXACT PRODUCTION IMPLEMENTATION
# ============================================================

def predict_model_n(
    land_size_m2,
    building_size_m2,
    bedrooms,
    bathrooms,
    lat,
    long
):

    if land_size_m2 <= 0:
        raise ValueError(
            "Luas tanah harus lebih besar dari 0."
        )

    if building_size_m2 <= 0:
        raise ValueError(
            "Luas bangunan harus lebih besar dari 0."
        )

    building_land_ratio = (
        building_size_m2
        /
        land_size_m2
    )

    subject = pd.DataFrame([{

        "land_size_m2":
            land_size_m2,

        "building_size_m2":
            building_size_m2,

        "building_land_ratio":
            building_land_ratio,

        "bedrooms":
            bedrooms,

        "bathrooms":
            bathrooms,

        "lat":
            lat,

        "long":
            long
    }])


    # --------------------------------------------------------
    # IMPUTE + SCALE
    # --------------------------------------------------------

    subject_imputed = (
        n_imputer.transform(
            subject[
                model_n_features
            ]
        )
    )

    subject_scaled = (
        n_scaler.transform(
            subject_imputed
        )
    )[0]


    # --------------------------------------------------------
    # SIMILARITY DISTANCE
    # --------------------------------------------------------

    distances = np.sqrt(

        np.sum(

            (
                X_n_scaled
                -
                subject_scaled
            ) ** 2,

            axis=1
        )
    )


    # --------------------------------------------------------
    # SELECT 150 LOCAL PROPERTIES
    # --------------------------------------------------------

    n_local = min(
        MODEL_N_NEIGHBORS,
        len(df_prod)
    )

    nearest_idx = (
        np.argsort(
            distances
        )[:n_local]
    )

    local_distances = (
        distances[
            nearest_idx
        ]
    )

    X_local = (
        X_n_imputed[
            nearest_idx
        ]
    )

    y_local = (
        y_n_log[
            nearest_idx
        ]
    )


    # --------------------------------------------------------
    # SIMILARITY WEIGHTS
    # --------------------------------------------------------

    local_scale = np.median(
        local_distances
    )

    if (
        local_scale <= 0
        or
        not np.isfinite(
            local_scale
        )
    ):

        local_scale = 1.0


    weights = np.exp(

        -0.5

        *

        (
            local_distances
            /
            local_scale
        ) ** 2
    )

    weights = np.maximum(
        weights,
        0.01
    )


    # --------------------------------------------------------
    # LOCAL RANDOM FOREST
    # --------------------------------------------------------

    local_rf = RandomForestRegressor(

        n_estimators=
            MODEL_N_TREES,

        random_state=42,

        n_jobs=-1,

        min_samples_leaf=2,

        max_features=1.0
    )

    local_rf.fit(

        X_local,

        y_local,

        sample_weight=
            weights
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    pred_log = (

        local_rf.predict(
            subject_imputed
        )[0]
    )

    prediction = np.expm1(
        pred_log
    )


    nearest_properties = (

        df_prod
        .iloc[
            nearest_idx
        ]
        .copy()
    )

    nearest_properties[
        "similarity_distance"
    ] = local_distances

    nearest_properties[
        "similarity_weight"
    ] = weights


    return {

        "prediction":
            prediction,

        "building_land_ratio":
            building_land_ratio,

        "local_sample_size":
            n_local,

        "avg_similarity_distance":
            local_distances.mean(),

        "median_similarity_distance":
            np.median(
                local_distances
            ),

        "nearest_properties":
            nearest_properties
    }


# ============================================================
# COMPARABLE ENGINE — EXACT PRODUCTION IMPLEMENTATION
# ============================================================

def production_comparables(
    land_size_m2,
    building_size_m2,
    bedrooms,
    bathrooms,
    lat,
    long,
    n_comps=5
):

    data = df_prod.copy()


    # --------------------------------------------------------
    # EXCLUDE EXACT SUBJECT
    # --------------------------------------------------------

    exact_match = (

        np.isclose(
            data["land_size_m2"],
            land_size_m2
        )

        &

        np.isclose(
            data["building_size_m2"],
            building_size_m2
        )

        &

        np.isclose(
            data["lat"],
            lat,
            atol=1e-7
        )

        &

        np.isclose(
            data["long"],
            long,
            atol=1e-7
        )
    )

    data = (
        data
        .loc[
            ~exact_match
        ]
        .copy()
    )


    # --------------------------------------------------------
    # HAVERSINE DISTANCE
    # --------------------------------------------------------

    earth_radius = 6371.0088

    lat1 = np.radians(
        lat
    )

    lon1 = np.radians(
        long
    )

    lat2 = np.radians(
        data[
            "lat"
        ]
        .astype(float)
        .values
    )

    lon2 = np.radians(
        data[
            "long"
        ]
        .astype(float)
        .values
    )

    dlat = (
        lat2
        -
        lat1
    )

    dlon = (
        lon2
        -
        lon1
    )

    a = (

        np.sin(
            dlat / 2
        ) ** 2

        +

        np.cos(
            lat1
        )

        *

        np.cos(
            lat2
        )

        *

        np.sin(
            dlon / 2
        ) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    data[
        "distance_km"
    ] = (
        earth_radius
        *
        c
    )


    # --------------------------------------------------------
    # PHYSICAL DIFFERENCES
    # --------------------------------------------------------

    data[
        "land_diff_pct"
    ] = (

        np.abs(

            data[
                "land_size_m2"
            ]

            -

            land_size_m2
        )

        /

        land_size_m2

        *

        100
    )

    data[
        "building_diff_pct"
    ] = (

        np.abs(

            data[
                "building_size_m2"
            ]

            -

            building_size_m2
        )

        /

        building_size_m2

        *

        100
    )

    data[
        "bedroom_diff"
    ] = np.abs(

        data[
            "bedrooms"
        ]
        .fillna(
            bedrooms
        )

        -

        bedrooms
    )

    data[
        "bathroom_diff"
    ] = np.abs(

        data[
            "bathrooms"
        ]
        .fillna(
            bathrooms
        )

        -

        bathrooms
    )


    # --------------------------------------------------------
    # APPRAISAL-STYLE COMPARABLE SCORE
    # --------------------------------------------------------

    data[
        "comparable_score"
    ] = (

        0.40
        *
        np.minimum(

            data[
                "distance_km"
            ]
            /
            5,

            2
        )

        +

        0.25
        *
        np.minimum(

            data[
                "land_diff_pct"
            ]
            /
            50,

            2
        )

        +

        0.20
        *
        np.minimum(

            data[
                "building_diff_pct"
            ]
            /
            50,

            2
        )

        +

        0.10
        *
        np.minimum(

            data[
                "bedroom_diff"
            ]
            /
            2,

            2
        )

        +

        0.05
        *
        np.minimum(

            data[
                "bathroom_diff"
            ]
            /
            2,

            2
        )
    )


    # --------------------------------------------------------
    # BASIC FILTER
    # --------------------------------------------------------

    candidates = data[

        (
            data[
                "land_diff_pct"
            ]
            <=
            100
        )

        &

        (
            data[
                "building_diff_pct"
            ]
            <=
            100
        )

        &

        (
            data[
                "distance_km"
            ]
            <=
            15
        )

    ].copy()


    if len(
        candidates
    ) < n_comps:

        candidates = (
            data.copy()
        )


    # --------------------------------------------------------
    # SELECT BEST COMPARABLES
    # --------------------------------------------------------

    comps = (

        candidates

        .sort_values(
            "comparable_score"
        )

        .head(
            n_comps
        )

        .copy()
    )


    # --------------------------------------------------------
    # MARKET EVIDENCE
    # --------------------------------------------------------

    prices = (
        comps[
            "price_in_rp"
        ]
        .astype(float)
    )

    comp_median = (
        prices.median()
    )

    comp_mean = (
        prices.mean()
    )

    comp_q1 = (
        prices.quantile(
            0.25
        )
    )

    comp_q3 = (
        prices.quantile(
            0.75
        )
    )

    comp_spread_pct = (

        (
            comp_q3
            -
            comp_q1
        )

        /

        comp_median

        *

        100

        if comp_median > 0

        else np.nan
    )

    avg_distance = (
        comps[
            "distance_km"
        ]
        .mean()
    )


    return {

        "comparables":
            comps,

        "count":
            len(comps),

        "median":
            comp_median,

        "mean":
            comp_mean,

        "q1":
            comp_q1,

        "q3":
            comp_q3,

        "spread_pct":
            comp_spread_pct,

        "avg_distance_km":
            avg_distance
    }


# ============================================================
# CONFIDENCE ENGINE
# ============================================================

def assign_confidence(
    model_comp_gap_pct,
    comp_spread_pct
):

    if pd.isna(
        model_comp_gap_pct
    ):

        level = "REVIEW"

    elif (
        model_comp_gap_pct
        <=
        10
    ):

        level = "HIGH"

    elif (
        model_comp_gap_pct
        <=
        30
    ):

        level = "MEDIUM"

    else:

        level = "REVIEW"


    if pd.isna(
        comp_spread_pct
    ):

        return "REVIEW"


    if (
        comp_spread_pct
        >
        40
    ):

        if level == "HIGH":
            level = "MEDIUM"

        elif level == "MEDIUM":
            level = "REVIEW"


    return level


# ============================================================
# FINAL AVM
# ============================================================

def run_avm(
    land_size_m2,
    building_size_m2,
    bedrooms,
    bathrooms,
    lat,
    long,
    n_comps=5
):

    result_n = predict_model_n(

        land_size_m2=
            land_size_m2,

        building_size_m2=
            building_size_m2,

        bedrooms=
            bedrooms,

        bathrooms=
            bathrooms,

        lat=
            lat,

        long=
            long
    )


    model_n_value = (
        result_n[
            "prediction"
        ]
    )


    result_comp = production_comparables(

        land_size_m2=
            land_size_m2,

        building_size_m2=
            building_size_m2,

        bedrooms=
            bedrooms,

        bathrooms=
            bathrooms,

        lat=
            lat,

        long=
            long,

        n_comps=
            n_comps
    )


    comp_value = (
        result_comp[
            "median"
        ]
    )


    final_value = (

        FINAL_WEIGHT_N
        *
        model_n_value

        +

        FINAL_WEIGHT_COMP
        *
        comp_value
    )


    model_comp_gap_pct = (

        abs(
            model_n_value
            -
            comp_value
        )

        /

        comp_value

        *

        100
    )


    evidence_lower = min(

        result_comp[
            "q1"
        ],

        final_value
    )

    evidence_upper = max(

        result_comp[
            "q3"
        ],

        final_value
    )


    confidence = assign_confidence(

        model_comp_gap_pct,

        result_comp[
            "spread_pct"
        ]
    )


    avg_distance = (
        result_comp[
            "avg_distance_km"
        ]
    )


    if avg_distance > 5:

        distance_flag = (
            "DISTANT COMPARABLES"
        )

    else:

        distance_flag = "OK"


    comps = (
        result_comp[
            "comparables"
        ]
    )


    zero_distance_count = (

        comps[
            "distance_km"
        ]

        .round(3)

        .eq(0)

        .sum()
    )


    if (
        zero_distance_count
        >=
        3
    ):

        coordinate_flag = (
            "SHARED-AREA COORDINATES"
        )

    else:

        coordinate_flag = "OK"


    return {

        "model_n_value":
            model_n_value,

        "comparable_value":
            comp_value,

        "final_value":
            final_value,

        "evidence_lower":
            evidence_lower,

        "evidence_upper":
            evidence_upper,

        "model_comp_gap_pct":
            model_comp_gap_pct,

        "comp_spread_pct":
            result_comp[
                "spread_pct"
            ],

        "avg_comp_distance_km":
            avg_distance,

        "similarity_distance":
            result_n[
                "avg_similarity_distance"
            ],

        "confidence":
            confidence,

        "distance_flag":
            distance_flag,

        "coordinate_flag":
            coordinate_flag,

        "comparables":
            comps
    }


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.72;
        margin-bottom: 1.5rem;
    }

    .value-box {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .final-value {
        font-size: 2rem;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🏠 Tangerang Residential AVM
    </div>

    <div class="subtitle">
    Similarity-Based Automated Valuation Model
    with Comparable Market Evidence
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header(
    "Subject Property"
)

st.sidebar.caption(
    "Masukkan karakteristik rumah yang akan dianalisis."
)


land_size = st.sidebar.number_input(
    "Luas Tanah (m²)",
    min_value=1.0,
    value=72.0,
    step=1.0
)

building_size = st.sidebar.number_input(
    "Luas Bangunan (m²)",
    min_value=1.0,
    value=67.0,
    step=1.0
)

bedrooms = st.sidebar.number_input(
    "Jumlah Kamar Tidur",
    min_value=0,
    value=3,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Jumlah Kamar Mandi",
    min_value=0,
    value=2,
    step=1
)

lat = st.sidebar.number_input(
    "Latitude",
    value=-6.2368145,
    format="%.7f"
)

long = st.sidebar.number_input(
    "Longitude",
    value=106.5663750,
    format="%.7f"
)

calculate = st.sidebar.button(
    "🔍 Hitung Indikasi AVM",
    type="primary",
    use_container_width=True
)


# ============================================================
# INFORMATION
# ============================================================

if not calculate:

    st.info(
        "Masukkan data properti pada panel sebelah kiri, "
        "kemudian klik **Hitung Indikasi AVM**."
    )

    st.markdown(
        """
        **Production architecture**

        `Subject Property`
        → `Similarity Model`
        → `Local Random Forest`
        → `Comparable Engine`
        → `80:20 Reconciliation`
        → `Confidence Assessment`
        """
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if calculate:

    try:

        with st.spinner(
            "Menganalisis properti dan mencari comparable..."
        ):

            result = run_avm(

                land_size_m2=
                    land_size,

                building_size_m2=
                    building_size,

                bedrooms=
                    bedrooms,

                bathrooms=
                    bathrooms,

                lat=
                    lat,

                long=
                    long,

                n_comps=5
            )


        # ====================================================
        # FINAL VALUE
        # ====================================================

        st.subheader(
            "AVM Asking-Price Indication"
        )

        st.markdown(
            f"""
            <div class="value-box">
                <div>Final Reconciled Indication</div>
                <div class="final-value">
                    {rupiah(result["final_value"])}
                </div>
                <div>
                    Evidence Range:
                    {rupiah(result["evidence_lower"])}
                    –
                    {rupiah(result["evidence_upper"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # MAIN METRICS
        # ====================================================

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model N",
            rupiah(
                result[
                    "model_n_value"
                ]
            )
        )

        c2.metric(
            "Comparable Median",
            rupiah(
                result[
                    "comparable_value"
                ]
            )
        )

        c3.metric(
            "Confidence",
            result[
                "confidence"
            ]
        )


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        st.subheader(
            "Diagnostics"
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Model–Comparable Gap",
            f'{result["model_comp_gap_pct"]:.2f}%'
        )

        d2.metric(
            "Comparable Spread",
            f'{result["comp_spread_pct"]:.2f}%'
        )

        d3.metric(
            "Avg. Comparable Distance",
            f'{result["avg_comp_distance_km"]:.2f} km'
        )

        d4.metric(
            "Similarity Distance",
            f'{result["similarity_distance"]:.3f}'
        )


        # ====================================================
        # FLAGS
        # ====================================================

        st.subheader(
            "Quality Flags"
        )

        if (
            result[
                "confidence"
            ]
            ==
            "HIGH"
        ):

            st.success(
                "Confidence Level: HIGH"
            )

        elif (
            result[
                "confidence"
            ]
            ==
            "MEDIUM"
        ):

            st.warning(
                "Confidence Level: MEDIUM"
            )

        else:

            st.error(
                "Confidence Level: REVIEW"
            )


        if (
            result[
                "distance_flag"
            ]
            !=
            "OK"
        ):

            st.warning(
                result[
                    "distance_flag"
                ]
            )


        if (
            result[
                "coordinate_flag"
            ]
            !=
            "OK"
        ):

            st.info(
                "Coordinate Flag: "
                +
                result[
                    "coordinate_flag"
                ]
            )


        # ====================================================
        # COMPARABLE TABLE
        # ====================================================

        st.subheader(
            "Comparable Market Evidence"
        )

        comps = (
            result[
                "comparables"
            ]
            .copy()
        )


        display_cols = [

            "district",

            "land_size_m2",

            "building_size_m2",

            "bedrooms",

            "bathrooms",

            "distance_km",

            "price_in_rp",

            "comparable_score"
        ]


        display_cols = [

            c
            for c in display_cols
            if c in comps.columns
        ]


        comp_display = (

            comps[
                display_cols
            ]
            .copy()
        )


        if (
            "price_in_rp"
            in
            comp_display.columns
        ):

            comp_display[
                "price_in_rp"
            ] = (

                comp_display[
                    "price_in_rp"
                ]

                .apply(
                    rupiah
                )
            )


        comp_display = (
            comp_display.rename(
                columns={

                    "district":
                        "District",

                    "land_size_m2":
                        "Land (m²)",

                    "building_size_m2":
                        "Building (m²)",

                    "bedrooms":
                        "Bedrooms",

                    "bathrooms":
                        "Bathrooms",

                    "distance_km":
                        "Distance (km)",

                    "price_in_rp":
                        "Asking Price",

                    "comparable_score":
                        "Comparable Score"
                }
            )
        )


        st.dataframe(
            comp_display,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # METHODOLOGY
        # ====================================================

        with st.expander(
            "How does this AVM work?"
        ):

            st.markdown(
                f"""
                The production AVM uses **{MODEL_N_NEIGHBORS}
                similarity-selected observations** to estimate
                the subject property's asking-price indication.

                A local Random Forest model is combined with
                comparable market evidence using the validated
                reconciliation weights:

                **{FINAL_WEIGHT_N:.0%} Model N +
                {FINAL_WEIGHT_COMP:.0%} Comparable Evidence.**

                The confidence indicator reflects agreement
                between model and comparable evidence and the
                dispersion of comparable asking prices.
                """
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.divider()

        st.caption(
            "Important: This output is an AVM asking-price "
            "indication based on listing data. It is not a "
            "final Market Value conclusion. Professional "
            "appraisal review remains required."
        )


    except Exception as e:

        st.error(
            "AVM calculation failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tangerang Residential AVM • Production v1.0"
)
