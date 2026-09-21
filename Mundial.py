
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="Predictor de Partidos Internacionales de Fútbol",
    page_icon="⚽",
    layout="centered"
)


RANDOM_STATE = 42


# ==========================================================
# CARGA DEL FEATURE DATASET
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("FeatureDataset.csv")

    df["date"] = pd.to_datetime(df["date"])

    return df


feature_dataset = load_data()


# ==========================================================
# CONFIGURACIÓN DE VARIABLES
# ==========================================================

COLUMNS_TO_DROP = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "target",
    "target_label",
    "tournament"
]


TARGET_COL = "target_label"


CLASS_LABELS = {
    0: "Victoria visitante",
    1: "Empate",
    2: "Victoria local"
}


# ==========================================================
# OBTENER COLUMNAS PREDICTORAS
# ==========================================================

feature_cols = [
    col
    for col in feature_dataset.columns
    if col not in COLUMNS_TO_DROP
]


# ==========================================================
# PREPROCESADOR
# ==========================================================

def build_numeric_preprocessor(feature_cols):

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, feature_cols)
        ],
        remainder="drop"
    )

    return preprocessor


# ==========================================================
# CONSTRUCCIÓN DEL MODELO XGBOOST
# ==========================================================

def build_xgboost(feature_cols):

    preprocessor = build_numeric_preprocessor(feature_cols)

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=3,

        n_estimators=300,

        learning_rate=0.03,

        max_depth=3,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_lambda=2.0,

        reg_alpha=0.1,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


# ==========================================================
# ENTRENAMIENTO
# ==========================================================

@st.cache_resource
def train_model(df, prediction_year):

    # Utilizar únicamente información anterior
    # al año que queremos predecir

    train_df = df[
        df["date"] < f"{prediction_year}-01-01"
    ].copy()

    X_train = train_df[feature_cols]

    y_train = train_df[TARGET_COL]

    model = build_xgboost(feature_cols)

    model.fit(
        X_train,
        y_train
    )

    return model


# ==========================================================
# PESO DEL TORNEO
# ==========================================================

def get_tournament_weight(tournament):

    if tournament == "FIFA World Cup":
        return 1.0

    elif tournament == "Friendly":
        return 0.5

    else:
        return 0.8


# ==========================================================
# OBTENER ÚLTIMO PARTIDO DE UN EQUIPO
# ==========================================================

def get_team_history(df, team, cutoff_date):

    team_matches = df[
        (
            (df["home_team"] == team) |
            (df["away_team"] == team)
        )
        &
        (df["date"] < cutoff_date)
    ].copy()

    team_matches = team_matches.sort_values("date")

    return team_matches


# ==========================================================
# CONSTRUIR SNAPSHOT DEL EQUIPO
# ==========================================================

def get_team_snapshot(df, team, cutoff_date):

    history = get_team_history(
        df,
        team,
        cutoff_date
    )

    if len(history) == 0:

        return {
            "elo": 1500,
            "win_rate": 0,
            "avg_goals": 0,
            "avg_goals_against": 0,
            "last5_win_rate": 0,
            "last5_goal_diff": 0
        }


    # ------------------------------------------------------
    # Último registro disponible
    # ------------------------------------------------------

    last_match = history.iloc[-1]

    if last_match["home_team"] == team:

        elo = last_match["home_elo"]

    else:

        elo = last_match["away_elo"]


    # ------------------------------------------------------
    # Estadísticas históricas
    # ------------------------------------------------------

    matches_played = len(history)

    wins = 0
    goals_for = 0
    goals_against = 0

    for _, match in history.iterrows():

        if match["home_team"] == team:

            gf = match["home_score"]
            ga = match["away_score"]

        else:

            gf = match["away_score"]
            ga = match["home_score"]


        goals_for += gf

        goals_against += ga


        if gf > ga:

            wins += 1


    win_rate = wins / matches_played

    avg_goals = goals_for / matches_played

    avg_goals_against = (
        goals_against /
        matches_played
    )


    # ------------------------------------------------------
    # Últimos 5 partidos
    # ------------------------------------------------------

    last5 = history.tail(5)

    recent_wins = 0

    recent_goal_diff = 0

    for _, match in last5.iterrows():

        if match["home_team"] == team:

            gf = match["home_score"]

            ga = match["away_score"]

        else:

            gf = match["away_score"]

            ga = match["home_score"]


        recent_goal_diff += gf - ga


        if gf > ga:

            recent_wins += 1


    last5_win_rate = (
        recent_wins / len(last5)
        if len(last5) > 0
        else 0
    )


    last5_goal_diff = (
        recent_goal_diff / len(last5)
        if len(last5) > 0
        else 0
    )


    return {

        "elo": elo,

        "win_rate": win_rate,

        "avg_goals": avg_goals,

        "avg_goals_against": avg_goals_against,

        "last5_win_rate": last5_win_rate,

        "last5_goal_diff": last5_goal_diff
    }


# ==========================================================
# HEAD TO HEAD
# ==========================================================

def get_h2h_features(
    df,
    home_team,
    away_team,
    cutoff_date
):

    h2h = df[
        (
            (
                (df["home_team"] == home_team) &
                (df["away_team"] == away_team)
            )
            |
            (
                (df["home_team"] == away_team) &
                (df["away_team"] == home_team)
            )
        )
        &
        (df["date"] < cutoff_date)
    ].copy()


    home_wins = 0
    away_wins = 0
    draws = 0

    home_goals = 0
    away_goals = 0


    for _, match in h2h.iterrows():

        if match["home_team"] == home_team:

            gf = match["home_score"]
            ga = match["away_score"]

        else:

            gf = match["away_score"]
            ga = match["home_score"]


        home_goals += gf

        away_goals += ga


        if gf > ga:

            home_wins += 1

        elif gf < ga:

            away_wins += 1

        else:

            draws += 1


    return {

        "h2h_matches": len(h2h),

        "h2h_home_wins": home_wins,

        "h2h_away_wins": away_wins,

        "h2h_draws": draws,

        "h2h_home_goals": home_goals,

        "h2h_away_goals": away_goals
    }


# ==========================================================
# CONSTRUIR FEATURES PARA UN PARTIDO
# ==========================================================

def build_match_features(
    df,
    home_team,
    away_team,
    prediction_year,
    tournament
):

    cutoff_date = pd.Timestamp(
        f"{prediction_year}-01-01"
    )


    home = get_team_snapshot(
        df,
        home_team,
        cutoff_date
    )


    away = get_team_snapshot(
        df,
        away_team,
        cutoff_date
    )


    h2h = get_h2h_features(
        df,
        home_team,
        away_team,
        cutoff_date
    )


    tournament_weight = get_tournament_weight(
        tournament
    )


    features = {

        # Elo
        "home_elo": home["elo"],

        "away_elo": away["elo"],

        "elo_diff":
            home["elo"] - away["elo"],


        # Win rate
        "home_win_rate":
            home["win_rate"],

        "away_win_rate":
            away["win_rate"],

        "win_rate_diff":
            home["win_rate"] - away["win_rate"],


        # Goles
        "home_avg_goals":
            home["avg_goals"],

        "away_avg_goals":
            away["avg_goals"],

        "avg_goals_diff":
            home["avg_goals"] - away["avg_goals"],


        "home_avg_goals_against":
            home["avg_goals_against"],

        "away_avg_goals_against":
            away["avg_goals_against"],

        "avg_goals_against_diff":
            home["avg_goals_against"]
            - away["avg_goals_against"],


        # Últimos 5
        "home_last5_win_rate":
            home["last5_win_rate"],

        "away_last5_win_rate":
            away["last5_win_rate"],

        "last5_win_rate_diff":
            home["last5_win_rate"]
            - away["last5_win_rate"],


        "home_last5_goal_diff":
            home["last5_goal_diff"],

        "away_last5_goal_diff":
            away["last5_goal_diff"],

        "last5_goal_diff_diff":
            home["last5_goal_diff"]
            - away["last5_goal_diff"],


        # H2H
        "h2h_matches":
            h2h["h2h_matches"],

        "h2h_home_wins":
            h2h["h2h_home_wins"],

        "h2h_away_wins":
            h2h["h2h_away_wins"],

        "h2h_draws":
            h2h["h2h_draws"],

        "h2h_home_goals":
            h2h["h2h_home_goals"],

        "h2h_away_goals":
            h2h["h2h_away_goals"],


        # Partido
        "neutral": 1,

        "tournament_weight":
            tournament_weight
    }


    return pd.DataFrame(
        [features],
        columns=feature_cols
    )


# ==========================================================
# INTERFAZ
# ==========================================================

st.title("⚽ Predictor de partidos de fútbol")



st.write(
    """
    Este modelo predice el resultado de un partido de fútbol internacional basándose en el historial de ambos equipos hasta la fecha del encuentro establecida.
    Recuerda que el modelo puede cometer errores, no vayas corriendo a meterle toda la quincena al Caliente por el resultado de la predicción.
    Tiempo de predicción: ~1 minuto. 
    """
)

st.image("soccer ex.jpg", caption="Fútbol")


st.header("Datos del encuentro")


# ----------------------------------------------------------
# Equipos
# ----------------------------------------------------------

teams = sorted(
    set(feature_dataset["home_team"])
    |
    set(feature_dataset["away_team"])
)


home_team = st.selectbox(
    "Equipo local",
    teams
)


away_team = st.selectbox(
    "Equipo visitante",
    teams
)


# ----------------------------------------------------------
# Año
# ----------------------------------------------------------

min_year = (
    feature_dataset["date"]
    .dt.year
    .min()
)

max_year = (
    feature_dataset["date"]
    .dt.year
    .max()
)


prediction_year = st.number_input(
    "Año del encuentro",
    min_value=int(min_year),
    max_value=int(max_year) + 1,
    value=2026,
    step=1
)


# ----------------------------------------------------------
# Torneo
# ----------------------------------------------------------

tournament = st.selectbox(
    "Tipo de torneo",
    [
        "FIFA World Cup",
        "Friendly",
        "Other Tournaments"
    ]
)


# ==========================================================
# VALIDACIÓN
# ==========================================================

if home_team == away_team:

    st.error(
        "El equipo local y visitante deben ser diferentes."
    )

else:

    if st.button("⚽ Predecir resultado"):

        with st.spinner("Entrenando modelo y generando predicción..."):

            model = train_model(
                feature_dataset,
                prediction_year
            )


            match_features = build_match_features(

                feature_dataset,

                home_team,

                away_team,

                prediction_year,

                tournament
            )


            probabilities = model.predict_proba(
                match_features
            )[0]


            prediction = model.predict(
                match_features
            )[0]


        # ==================================================
        # RESULTADO
        # ==================================================

        st.subheader("Resultado de la predicción")


        st.write(
            f"### {home_team} vs {away_team}"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Victoria local",
                f"{probabilities[2] * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Empate",
                f"{probabilities[1] * 100:.2f}%"
            )


        with col3:

            st.metric(
                "Victoria visitante",
                f"{probabilities[0] * 100:.2f}%"
            )


        st.divider()


        st.success(
            f"Predicción: **{CLASS_LABELS[prediction]}**"
        )


        # --------------------------------------------------
        # Probabilidades
        # --------------------------------------------------

        probability_df = pd.DataFrame({

            "Resultado": [
                "Victoria visitante",
                "Empate",
                "Victoria local"
            ],

            "Probabilidad": [

                probabilities[0],

                probabilities[1],

                probabilities[2]
            ]
        })


        st.bar_chart(
            probability_df.set_index("Resultado")
        )
