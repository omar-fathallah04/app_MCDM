import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Application aide à la décision - MCDM", layout="wide"
)

# --- STYLES CSS ET ANIMATIONS PERSONNALISÉS ---
st.markdown(
    """
    <style>
    /* Animation d'apparition globale */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-out;
    }

    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        font-weight: 300;
    }

    /* Style des conteneurs de résultats avec effet de survol */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-out;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        border-color: #2a5298;
    }

    /* Style des boutons Streamlit */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(42, 82, 152, 0.2);
    }

    .stButton>button:hover {
        opacity: 0.95;
        box-shadow: 0 6px 15px rgba(42, 82, 152, 0.35);
        transform: translateY(-2px);
    }

    /* Personnalisation de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- EN-TÊTE DESIGN ---
st.markdown(
    """
    <div class="main-header">
        <h1>Bienvenue à l'application d'aide à la décision multicritère</h1>
        <p>Plateforme interactive d'évaluation, de pondération et de classement des alternatives</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR : Paramètres ---
st.sidebar.markdown("### Configuration du Problème")

num_alt = st.sidebar.number_input(
    "Nombre d'alternatives (m)", min_value=2, max_value=10, value=3
)
num_crit = st.sidebar.number_input(
    "Nombre de critères (n)", min_value=2, max_value=10, value=3
)

alt_names = [
    st.sidebar.text_input(f"Nom Alternative {i+1}", f"A{i+1}")
    for i in range(num_alt)
]
crit_names = [
    st.sidebar.text_input(f"Nom Critère {j+1}", f"C{j+1}")
    for j in range(num_crit)
]

st.sidebar.markdown("---")
st.sidebar.markdown("### Type des Critères")
crit_types = []
for j in range(num_crit):
  t = st.sidebar.selectbox(
      f"Sens pour {crit_names[j]}", ["Max (Bénéfice)", "Min (Coût)"], key=f"t_{j}"
  )
  crit_types.append(1 if "Max" in t else -1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Poids des Critères")
weights = []
st.sidebar.markdown(
    "<small>Les poids saisis seront normalisés automatiquement.</small>",
    unsafe_allow_html=True,
)
for j in range(num_crit):
  w = st.sidebar.number_input(
      f"Poids {crit_names[j]}",
      min_value=0.01,
      max_value=1.0,
      value=float(1.0 / num_crit),
      key=f"w_{j}",
  )
  weights.append(w)
weights = np.array(weights)
weights = weights / np.sum(weights)  # Normalisation

# --- CORPS PRINCIPAL : Matrice de décision ---
st.markdown("### Saisie de la matrice de décision")
st.markdown(
    "Renseignez ci-dessous les performances quantitatives ou scores de chaque"
    " alternative par rapport à chaque critère :"
)

matrix_data = []
for i, alt in enumerate(alt_names):
  row = []
  cols = st.columns(num_crit)
  for j, crit in enumerate(crit_names):
    val = cols[j].number_input(
        f"{alt} / {crit}", value=float((i + 1) * (j + 1)), key=f"val_{i}_{j}"
    )
    row.append(val)
  matrix_data.append(row)

X = np.array(matrix_data)

# Affichage de la matrice sous forme de DataFrame dans un conteneur stylisé
st.markdown("<br>", unsafe_allow_html=True)
df_decision = pd.DataFrame(X, index=alt_names, columns=crit_names)
st.dataframe(df_decision, use_container_width=True)

# --- CHOIX DU MODELE ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Choix du Modèle de Classement")
model_choice = st.selectbox(
    "Sélectionnez la méthode d'analyse :",
    ["WASPAS", "TOPSIS", "VIKOR", "Tous les modèles (Comparaison)"],
)


# --- FONCTIONS DES METHODES ---
def run_waspas(X, weights, crit_types):
  m, n = X.shape
  r = np.zeros((m, n))
  for j in range(n):
    col = X[:, j]
    if crit_types[j] == 1:  # Bénéfice
      max_val = np.max(col)
      r[:, j] = col / max_val if max_val != 0 else col
    else:  # Coût
      min_val = np.min(col)
      r[:, j] = min_val / col if min_val != 0 else np.zeros_like(col)

  q1 = np.dot(r, weights)
  q2 = np.prod(r**weights, axis=1)
  Q = 0.5 * q1 + 0.5 * q2
  return Q, q1, q2


def run_topsis(X, weights, crit_types):
  m, n = X.shape
  norm_X = np.zeros((m, n))
  for j in range(n):
    denom = np.sqrt(np.sum(X[:, j] ** 2))
    norm_X[:, j] = X[:, j] / denom if denom != 0 else X[:, j]

  v = norm_X * weights

  ideal_pos = np.zeros(n)
  ideal_neg = np.zeros(n)
  for j in range(n):
    if crit_types[j] == 1:
      ideal_pos[j] = np.max(v[:, j])
      ideal_neg[j] = np.min(v[:, j])
    else:
      ideal_pos[j] = np.min(v[:, j])
      ideal_neg[j] = np.max(v[:, j])

  S_pos = np.sqrt(np.sum((v - ideal_pos) ** 2, axis=1))
  S_neg = np.sqrt(np.sum((v - ideal_neg) ** 2, axis=1))

  RC = S_neg / (S_neg + S_pos) if (S_neg + S_pos).any() != 0 else np.zeros(m)
  return RC


def run_vikor(X, weights, crit_types, v_strat=0.5):
  m, n = X.shape
  f_star = np.zeros(n)
  f_min = np.zeros(n)

  for j in range(n):
    if crit_types[j] == 1:
      f_star[j] = np.max(X[:, j])
      f_min[j] = np.min(X[:, j])
    else:
      f_star[j] = np.min(X[:, j])
      f_min[j] = np.max(X[:, j])

  S = np.zeros(m)
  R = np.zeros(m)
  for i in range(m):
    terms = np.zeros(n)
    for j in range(n):
      diff = (
          (f_star[j] - X[i, j]) / (f_star[j] - f_min[j])
          if (f_star[j] - f_min[j]) != 0
          else 0
      )
      terms[j] = weights[j] * diff
    S[i] = np.sum(terms)
    R[i] = np.max(terms)

  S_star, S_minus = np.min(S), np.max(S)
  R_star, R_minus = np.min(R), np.max(R)

  Q = np.zeros(m)
  for i in range(m):
    term1 = (
        (S[i] - S_star) / (S_minus - S_star) if (S_minus - S_star) != 0 else 0
    )
    term2 = (
        (R[i] - R_star) / (R_minus - R_star) if (R_minus - R_star) != 0 else 0
    )
    Q[i] = v_strat * term1 + (1 - v_strat) * term2

  return Q, S, R


# --- EXECUTION ET AFFICHAGE ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Lancer l'Évaluation et le Classement"):
  st.markdown("---")
  st.markdown("### Résultats de l'Analyse")

  if model_choice == "WASPAS" or model_choice == "Tous les modèles (Comparaison)":
    Q_w, _, _ = run_waspas(X, weights, crit_types)
    df_waspas = pd.DataFrame(
        {"Score WASPAS (Q)": Q_w}, index=alt_names
    ).sort_values(by="Score WASPAS (Q)", ascending=False)

    st.markdown(
        """
        <div class="result-card">
            <h4>Méthode WASPAS (Weighted Aggregates Sum Product Assessment)</h4>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df_waspas, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  if model_choice == "TOPSIS" or model_choice == "Tous les modèles (Comparaison)":
    RC_t = run_topsis(X, weights, crit_types)
    df_topsis = pd.DataFrame(
        {"Proximité Idéale (RC*)": RC_t}, index=alt_names
    ).sort_values(by="Proximité Idéale (RC*)", ascending=False)

    st.markdown(
        """
        <div class="result-card">
            <h4>Méthode TOPSIS (Technique for Order Preference by Similarity to an Ideal Solution)</h4>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df_topsis, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

  if model_choice == "VIKOR" or model_choice == "Tous les modèles (Comparaison)":
    Q_v, S, R = run_vikor(X, weights, crit_types)
    df_vikor = pd.DataFrame(
        {"S (Utilité)": S, "R (Regret)": R, "Q (Compromis)": Q_v}, index=alt_names
    ).sort_values(
        by="Q (Compromis)", ascending=True
    )

    st.markdown(
        """
        <div class="result-card">
            <h4>Méthode VIKOR (Compromise Ranking Method)</h4>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df_vikor, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Copyright en bas de page ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem; padding:"
    " 1rem;'>© Omar FATHALLAH - Tous droits réservés</div>",
    unsafe_allow_html=True,
)