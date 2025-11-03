import streamlit as st
import numpy as np
import plotly.graph_objects as go

EPS = 1e-12

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PK-SIMple",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PREAMBLE ----------------
st.markdown(
    """
    <div style="
        width:100%; 
        padding:20px; 
        border-radius:10px; 
        background-color:#e8f0fe; 
        border: 1px solid #c3d1f0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align:left;
        color:#000000;
    ">
        <h2 style="margin-top:0; text-align:center;">💊 PK-SIMple 💊</h2>
        <p style="margin-bottom:10px;">
        This interactive tool was developed by pharmaceutics staff and researchers at the <strong>School of Pharmacy, University College Cork (UCC)</strong> 
        to help teach the fundamental principles of pharmacokinetics. It implements a one-compartment model with first-order absorption and elimination. Features include:
        </p>
        <ul style="margin-top:0; margin-bottom:0;">
            <li>Options for IV bolus, IV infusion, and extravascular (oral) dosing.</li>
            <li>Simulations of repeated dosing and chronic accumulation.</li>
            <li>Adjustable glomerular filtration rate, renal secretion/reabsorption, hepatic blood flow, and intrinsic clearance.</li>
            <li>User-defined enzymes to simulate drug-drug interactions and genetic variability.</li>
            <li>Calculation of volume of distribution based on fractions unbound in plasma and tissues.</li>
            <li>Automatic detection of flip-flop pharmacokinetics for drugs with low absorption rate constants.</li>
            <li>Downloadable plots of plasma concentration and ln(Cp).</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- LAYOUT ----------------
left_col, right_col = st.columns([1,1])

# ---------------- LEFT PANEL (INPUTS) ----------------
with left_col:
    st.header("Simulation Inputs")
    
    # ---------- ABSORPTION ----------
    st.subheader("Absorption")
    route = st.selectbox("Route of administration", ["IV bolus", "IV infusion", "Extravascular (oral/other)"])

    if route in ["IV bolus", "IV infusion"]:
        F = 1.0
        ka = None
        if route=="IV infusion":
            Tinf = st.number_input("Infusion duration $T_{inf}$ (h)", 0.01, 24.0, 1.0)
        else:
            Tinf = None
    else:
        bio_method = st.radio(
            "Specify bioavailability directly or calculate from $F_a$, $F_g$, and $E_h$?",
            ["Directly specify F", "Calculate F"]
        )
        if bio_method == "Directly specify F":
            F = st.slider("Bioavailability $F$", 0.0, 1.0, 1.0, step=0.05)
        else:
            Fa = st.slider("Fraction absorbed $F_a$", 0.0, 1.0, 1.0, step=0.05)
            Fg = st.slider(r"Fraction escaping gut metabolism $F_g$", 0.0, 1.0, 1.0, step=0.05)
            st.markdown("**Note:** Hepatic extraction $E_h$ will only be applied if calculated from $CL_{int}$ in the Metabolism & Elimination section.")
            F = Fa*Fg # Eh applied later if available
        ka = st.number_input(r"Absorption rate constant $k_a$ (h⁻¹)", 0.0, 5.0, 1.0)

    # ---------- DISTRIBUTION ----------
    st.subheader("Distribution")
    Vd_mode = st.radio("Volume of distribution $V_d$ mode", ["Fixed $V_d$", "Calculate from $f_u/f_{ut}$"])
    if Vd_mode == "Fixed $V_d$":
        Vd = st.number_input("Volume of distribution $V_d$ (L)", 1.0, 500.0, 50.0, step=1.0)
        fu = 1.0
        st.warning(
        "⚠️ Fraction unbound ($f_u$) not specified — assuming $f_u = 1.0$ (complete plasma unbinding). "
        "This will affect calculation of renal clearance from glomerular filtration in Renal-Hepatic Mode."
        )
    else:
        Vp = st.number_input("Plasma volume $V_p$ (L)", 1.0, 10.0, 3.0, step=0.1)
        Vt = st.number_input("Tissue volume $V_t$ (L)", 1.0, 500.0, 37.0, step=1.0)
        fu = st.slider(r"Fraction unbound in plasma $f_u$", 0.0, 1.0, 0.9, step=0.01)
        fut = st.slider(r"Fraction unbound in tissue $f_{ut}$", 0.0, 1.0, 0.8, step=0.01)
        Vd = Vp + Vt * (fu / max(fut, EPS))
    st.write(f"Calculated $V_d$ = {Vd:.2f} L")

    # ---------- METABOLISM & ELIMINATION ----------
    st.subheader("Metabolism & Elimination")
    use_global_pk = st.radio("Input mode:", ["Global Parameter Mode", "Renal-Hepatic Mode"])
    CL, k, t_half, Eh, total_enzyme_CL, CL_renal, CL_hepatic = 0.0, 0.0, float('inf'), None, 0.0, 0.0, 0.0

    if use_global_pk=="Global Parameter Mode":
        pk_choice = st.selectbox("Specify parameter", ["Clearance", "Half-life", "Elimination rate constant"])
        if pk_choice=="Clearance":
            CL = st.number_input(r"Clearance $CL$ (L/h)", 0.0, 100.0, 5.0, step=0.1)
            k = CL / max(Vd, EPS)
            t_half = float("inf") if k<=EPS else np.log(2)/k
        elif pk_choice==r"Half-life":
            t_half = st.number_input(r"Half-life $t_{1/2}$ (h)", 0.0, 1000.0, 8.0)
            k = np.log(2)/max(t_half, EPS) if t_half>0 else 0.0
            CL = k * Vd
        else:
            k = st.number_input(r"Elimination rate constant $k_{el}$ (h⁻¹)", 0.0, 5.0, 0.1)
            CL = k*Vd
            t_half = np.log(2)/k if k>EPS else float("inf")
    else:
        include_renal = st.checkbox("Include renal clearance", value=True)
        include_hepatic = st.checkbox("Calculate hepatic clearance via $CL_{int}$ and $Q_h$", value=True)
        include_other = st.checkbox("Calculate hepatic clearance by individual enzyme contributions", value=False)
        if include_hepatic and include_other:
            st.warning("Cannot select both hepatic intrinsic clearance and other enzymes. Choose one.")
            st.stop()
# ---------- RENAL CLEARANCE BASED ON GFR ----------
        if include_renal:
            st.subheader("Renal Clearance via GFR")
            
            # Slider for GFR in mL/min
            GFR_mL_min = st.slider(
                r"Glomerular filtration rate $GFR$ (mL/min)", 
                0.0, 120.0, 100.0, step=1.0
            )
            
            # Slider for secretion/reabsorption adjustment in mL/min
            CL_adj_mL_min = st.slider(
                "Adjust renal clearance for secretion (+) / reabsorption (-) (mL/min)",
                -120.0, 600.0, 0.0, step=1.0
            )
            
            # Convert to L/h
            GFR_L_h = GFR_mL_min * 60 / 1000
            CL_adj_L_h = CL_adj_mL_min * 60 / 1000
            
            # Calculate renal clearance
            CL_renal = max(fu * GFR_L_h + CL_adj_L_h, 0.0)  # cannot be negative
            
            st.write(f"Calculated renal clearance $CL_{{renal}}$ = {CL_renal:.2f} L/h")

        if include_hepatic:
            st.subheader(r"Hepatic Clearance via $Q_h$ and $CL_{int}$")
            Qh = st.number_input(r"Hepatic blood flow $Q_h$ (L/h)", 0.1, 150.0, 90.0)
            CLint = st.number_input(r"Intrinsic clearance $CL_{int}$ (L/h)", 0.1, 500.0, 200.0)
            CL_hepatic = (Qh * fu * CLint)/(Qh + fu*CLint)
            Eh = CL_hepatic / Qh
            st.write(f"Calculated hepatic extraction ratio $E_h$ = {Eh:.2f}")
        if include_other:
            st.subheader("Hepatic Clearance by Individual Enzyme Contributions")
            if "enzymes" not in st.session_state: st.session_state["enzymes"] = {}
            with st.expander("Add/remove enzyme contributions (L/h)"):
                enzyme_name = st.text_input("Enzyme name", key="enz_name")
                enzyme_cl = st.number_input("Enzyme CL (L/h)", 0.0, 50.0, 0.0, key="enz_cl")
                if st.button("Add enzyme"):
                    if enzyme_name:
                        st.session_state["enzymes"][enzyme_name] = float(enzyme_cl)
                        st.rerun()
                if st.session_state["enzymes"]:
                    for enz, val in list(st.session_state["enzymes"].items()):
                        cols = st.columns([3,1])
                        cols[0].write(f"{enz}: {val:.2f} L/h")
                        if cols[1].button(f"Remove {enz}"):
                            st.session_state["enzymes"].pop(enz)
                            st.rerun()
            total_enzyme_CL = sum(st.session_state["enzymes"].values())
        CL = CL_renal + CL_hepatic + total_enzyme_CL
        k = CL / max(Vd, EPS)
        t_half = float("inf") if k<=EPS else np.log(2)/k

    # ---------- SIMULATION DETAILS ----------
    st.subheader("Dosing")
    dose = st.number_input("Dose (mg)", 0.0, 20000.0, 100.0)
    tau = st.number_input("Dosing interval (h) (simulation time if single dose)", 0.01, 168.0, 12.0)
    n_doses = int(st.number_input("Number of doses", 1, 100, 5))
    t_end = tau*n_doses
    if route=="IV infusion" and Tinf is None:
        Tinf = 1.0

# ---------------- RIGHT PANEL (OUTPUTS) ----------------
with right_col:
    st.header("Simulation Results")
    time = np.linspace(0, t_end, 500)
    
    def conc_iv_bolus(dose,Vd,k,t,tau,n_doses):
        Cp = np.zeros_like(t)
        for n in range(n_doses):
            td = np.maximum(t - n*tau, 0.0)
            Cp += (dose/Vd)*np.exp(-k*td)*(t>=n*tau)
        return Cp

    def conc_iv_infusion(dose,Vd,k,t,tau,n_doses,Tinf):
        R = dose/max(Tinf, EPS)
        Cp = np.zeros_like(t)
        for n in range(n_doses):
            t_start = n*tau
            t_end_inf = t_start + Tinf
            during = (t>=t_start) & (t<t_end_inf)
            after = t>=t_end_inf
            if k>EPS:
                Cp[during] += (R/(Vd*k))*(1-np.exp(-k*(t[during]-t_start)))
                Cp[after] += (R/(Vd*k))*(1-np.exp(-k*Tinf))*np.exp(-k*(t[after]-t_end_inf))
            else:
                Cp[during] += (R/Vd)*(t[during]-t_start)
                Cp[after] += (R/Vd)*Tinf
        return Cp

    def conc_extravascular(dose,Vd,k,ka,F,t,tau,n_doses):
        Cp = np.zeros_like(t)
        for n in range(n_doses):
            td = np.maximum(t - n*tau,0.0)
            mask = (t>=n*tau)
            if ka is None: raise ValueError(r"$k_a$ must be provided")
            if abs(ka-k)>EPS:
                term = (F*dose*ka)/(Vd*(ka-k))
                Cp += term*(np.exp(-k*td)-np.exp(-ka*td))*mask
            else:
                term = (F*dose*ka)/Vd
                Cp += term*td*np.exp(-k*td)*mask
        return Cp

    if route=="Extravascular (oral/other)" and bio_method=="Calculate F" and Eh is not None:
        F = Fa * Fg * (1 - Eh)
    
    if route=="IV bolus": Cp = conc_iv_bolus(dose,Vd,k,time,tau,n_doses)
    elif route=="IV infusion": Cp = conc_iv_infusion(dose,Vd,k,time,tau,n_doses,Tinf)
    else: Cp = conc_extravascular(dose,Vd,k,ka,F,time,tau,n_doses)
    
    # ---------- PLOT ----------
    plot_type = st.radio(
    "Select plot type",
    ["Concentration (Cp)", "ln(Cp)"]
    )
    fig = go.Figure()
    if plot_type == "Concentration (Cp)":
        y_data = Cp
        y_label = "Concentration (mg/L)"
    else:
        # Use natural log, avoid log(0)
        y_data = np.log(np.maximum(Cp, EPS))
        y_label = "ln(Concentration)"

    fig.add_trace(go.Scatter(x=time, y=y_data, mode="lines", name="Central Compartment"))
    fig.update_layout(
        title="Concentration–Time Profile",
        xaxis_title="Time (h)",
        yaxis_title=y_label,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
        # ---------- FLIP-FLOP WARNING ----------
    if route=="Extravascular (oral/other)" and ka is not None and ka < k:
        st.warning(
            "⚠️ Flip-flop kinetics detected ($k_a$ < $k_{el}$). Apparent terminal decline in plot reflects $k_a$. Below parameters are calculated using $k_{el}$."
        )
    # ---------- TEXTUAL RESULTS ----------
    empty1, metrics_col, empty2 = st.columns([1, 6, 1])
    if CL > 0:
    # fe as % of dose reaching systemic circulation
        fe_percent = 100 * CL_renal / CL
    else:
        fe_percent = 0.0
    with metrics_col:
        col1, col2 = st.columns(2)
        col1.metric(r"Elimination rate constant $k_{el}$ (h⁻¹)", f"{k:.3f}")
        col1.metric(r"Half-life $t_{1/2}$ (h)", "∞" if t_half==float('inf') else f"{t_half:.3f}")
        col1.metric(r"Clearance $CL$ (L/h)", f"{CL:.3f}")
        col1.metric(r"Bioavailability $F$ (%)", f"{F*100:.1f}%")

        col2.metric("$C_{max}$ (mg/L)", f"{np.max(Cp):.4f}")
        col2.metric("$T_{max}$ (h)", f"{float(time[np.argmax(Cp)]):.2f}")
        col2.metric(f"AUC (0–{t_end:.1f} h) (mg·h/L)", f"{np.trapz(Cp,time):.3f}")
        col2.metric("Absorbed dose cleared renally (%)", f"{fe_percent:.2f}%")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #333;
        text-align: center;
        padding: 5px 0;
        font-size: 12px;
        opacity: 0.8;
    }
    </style>
    <div class="footer">
        &copy; 2025  <a href="https://www.ucc.ie/en/pharmacy/" target="_blank">School of Pharmacy, University College Cork</a>.
    </div>
    """,
    unsafe_allow_html=True
)
