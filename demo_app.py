"""
Stellar Spectra Visualizer
Based on: Deep-Generative-Model-Stellar-Spectra by Tirthesh Jani
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stellar Spectra Visualizer", page_icon="🌌", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .star-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .spectrum-container {
        background: #0a0a0a;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌌 Stellar Spectra Visualizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Exploring the APOGEE Survey - Deep Generative Models for Astronomy</div>', unsafe_allow_html=True)

# Generate synthetic stellar spectra for demonstration
def generate_stellar_spectrum(spectral_type, wavelength_range=(15000, 17000), n_points=1000):
    """Generate synthetic stellar spectrum based on spectral type"""
    wavelength = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
    
    # Base continuum
    continuum = np.ones(n_points)
    
    # Add absorption features based on spectral type
    flux = continuum.copy()
    
    if spectral_type == "O":  # Hot, massive stars
        # Strong ionized lines
        lines = [(15400, 0.3), (15600, 0.25), (16400, 0.35)]
        temperature = 40000
        color = "#9bb0ff"  # Blue
    elif spectral_type == "B":
        lines = [(15350, 0.25), (15550, 0.2), (16100, 0.3)]
        temperature = 20000
        color = "#aabfff"
    elif spectral_type == "A":  # Sirius-like
        lines = [(15250, 0.4), (15800, 0.3), (16700, 0.35)]
        temperature = 8500
        color = "#cad7ff"
    elif spectral_type == "F":  # Procyon-like
        lines = [(15300, 0.5), (15650, 0.4), (16250, 0.45)]
        temperature = 6500
        color = "#f8f7ff"
    elif spectral_type == "G":  # Sun-like
        lines = [(15150, 0.6), (15400, 0.55), (15890, 0.7), (16500, 0.5)]
        temperature = 5778
        color = "#fff4ea"
    elif spectral_type == "K":  # Cooler, orange
        lines = [(15120, 0.7), (15450, 0.65), (15950, 0.75), (16650, 0.6)]
        temperature = 4500
        color = "#ffd2a1"
    else:  # M - Coolest, red
        lines = [(15100, 0.8), (15500, 0.75), (16000, 0.85), (16700, 0.7)]
        temperature = 3000
        color = "#ffcc6f"
    
    # Generate absorption lines
    for center, depth in lines:
        width = 20 + np.random.rand() * 30
        flux -= depth * np.exp(-((wavelength - center) ** 2) / (2 * width ** 2))
    
    # Add noise
    noise = np.random.normal(0, 0.02, n_points)
    flux += noise
    
    # Ensure flux stays positive
    flux = np.clip(flux, 0.1, 1.0)
    
    return wavelength, flux, temperature, color

def get_spectral_info(spectral_type):
    """Get information about spectral type"""
    info = {
        "O": {
            "name": "Blue Giant",
            "examples": "Mintaka, Naos, Alnitak",
            "mass": "16-90 M☉",
            "lifespan": "1-10 million years",
            "features": "Ionized helium lines, weak hydrogen"
        },
        "B": {
            "name": "Blue-White Star",
            "examples": "Rigel, Spica, Regulus",
            "mass": "2.1-16 M☉",
            "lifespan": "10-100 million years",
            "features": "Neutral helium, stronger hydrogen"
        },
        "A": {
            "name": "White Star",
            "examples": "Sirius, Vega, Altair",
            "mass": "1.4-2.1 M☉",
            "lifespan": "100 million - 1 billion years",
            "features": "Strong hydrogen lines"
        },
        "F": {
            "name": "Yellow-White Star",
            "examples": "Procyon, Canopus",
            "mass": "1.04-1.4 M☉",
            "lifespan": "1-4 billion years",
            "features": "Weaker hydrogen, ionized metals"
        },
        "G": {
            "name": "Yellow Star",
            "examples": "Sun, Alpha Centauri A",
            "mass": "0.8-1.04 M☉",
            "lifespan": "4-17 billion years",
            "features": "Strong ionized calcium, sodium"
        },
        "K": {
            "name": "Orange Star",
            "examples": "Arcturus, Aldebaran",
            "mass": "0.45-0.8 M☉",
            "lifespan": "17-70 billion years",
            "features": "Strong neutral metals, molecular bands"
        },
        "M": {
            "name": "Red Dwarf",
            "examples": "Proxima Centauri, Betelgeuse",
            "mass": "0.08-0.45 M☉",
            "lifespan": "70+ billion years",
            "features": "Strong molecular bands (TiO, VO)"
        }
    }
    return info.get(spectral_type, {})

# Spectral lines database
SPECTRAL_LINES = {
    "H-alpha": (6563, "Hydrogen", "#ff6b6b"),
    "H-beta": (4861, "Hydrogen", "#ff8787"),
    "Ca II K": (3933, "Ionized Calcium", "#ffd43b"),
    "Ca II H": (3968, "Ionized Calcium", "#ffd43b"),
    "Na I D": (5890, "Neutral Sodium", "#ffa94d"),
    "Mg I b": (5167, "Neutral Magnesium", "#69db7c"),
    "Fe I": (5270, "Neutral Iron", "#74c0fc"),
    "TiO": (7054, "Titanium Oxide", "#da77f2"),
}

# Sidebar controls
st.sidebar.header("🔭 Observation Settings")
spectral_types_selected = st.sidebar.multiselect(
    "Spectral Types to Display",
    ["O", "B", "A", "F", "G", "K", "M"],
    default=["G", "K", "M"]
)

wavelength_min = st.sidebar.slider("Min Wavelength (Å)", 15000, 16000, 15100)
wavelength_max = st.sidebar.slider("Max Wavelength (Å)", 16000, 17000, 17000)

show_lines = st.sidebar.checkbox("Show Spectral Lines", value=True)
normalize = st.sidebar.checkbox("Normalize Spectra", value=True)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Spectra Viewer", "⭐ Star Catalog", "🧬 Spectral Types", "🔬 Analysis"])

with tab1:
    st.subheader("Stellar Spectra Visualization")
    
    if spectral_types_selected:
        fig = go.Figure()
        
        for i, spec_type in enumerate(spectral_types_selected):
            wave, flux, temp, color = generate_stellar_spectrum(
                spec_type, 
                wavelength_range=(wavelength_min, wavelength_max)
            )
            
            if normalize:
                flux = flux / np.max(flux)
            
            # Offset for visibility
            offset = i * 0.3
            
            fig.add_trace(go.Scatter(
                x=wave,
                y=flux + offset,
                mode='lines',
                name=f'Type {spec_type} (T={temp}K)',
                line=dict(color=color, width=1.5),
                hovertemplate=f'<b>Type {spec_type}</b><br>' +
                             'Wavelength: %{x:.1f} Å<br>' +
                             'Flux: %{y:.3f}<extra></extra>'
            ))
        
        # Add spectral line markers
        if show_lines:
            for name, (wavelength, element, color) in SPECTRAL_LINES.items():
                if wavelength_min <= wavelength <= wavelength_max:
                    fig.add_vline(
                        x=wavelength,
                        line=dict(color=color, dash='dash', width=1),
                        annotation=dict(text=f'{name}', textangle=-90, font=dict(size=10))
                    )
        
        fig.update_layout(
            title="APOGEE Spectral Region (H-band)",
            xaxis_title="Wavelength (Å)",
            yaxis_title="Normalized Flux + Offset",
            template="plotly_dark",
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend for spectral lines
        if show_lines:
            st.subheader("Spectral Lines Reference")
            cols = st.columns(4)
            for i, (name, (wavelength, element, color)) in enumerate(SPECTRAL_LINES.items()):
                with cols[i % 4]:
                    st.markdown(
                        f"<div style='color: {color}; font-weight: bold;'>"
                        f"{name} ({wavelength}Å)<br>"
                        f"<span style='font-size: 0.8em; color: #888;'>{element}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("👆 Select at least one spectral type from the sidebar to view spectra.")

with tab2:
    st.subheader("⭐ Simulated Star Catalog")
    
    # Generate simulated catalog
    np.random.seed(42)
    n_stars = 100
    
    spectral_types = np.random.choice(["O", "B", "A", "F", "G", "K", "M"], n_stars, 
                                      p=[0.02, 0.05, 0.1, 0.15, 0.3, 0.25, 0.13])
    
    catalog_data = []
    for spec_type in spectral_types:
        _, _, temp, _ = generate_stellar_spectrum(spec_type)
        catalog_data.append({
            "Spectral Type": spec_type,
            "Temperature (K)": temp + np.random.randint(-200, 200),
            "Metallicity [Fe/H]": np.random.normal(-0.5, 0.5),
            "Surface Gravity (log g)": np.random.uniform(0, 5),
            "Distance (pc)": np.random.exponential(500),
            "RA (deg)": np.random.uniform(0, 360),
            "Dec (deg)": np.random.uniform(-90, 90)
        })
    
    df = pd.DataFrame(catalog_data)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        temp_range = st.slider("Temperature Range (K)", 2000, 45000, (2000, 45000), 1000)
    with col2:
        feh_range = st.slider("Metallicity [Fe/H]", -3.0, 1.0, (-3.0, 1.0), 0.1)
    
    filtered_df = df[
        (df["Temperature (K)"].between(*temp_range)) &
        (df["Metallicity [Fe/H]"].between(*feh_range))
    ]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} stars")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hr = px.scatter(
            filtered_df, x="Temperature (K)", y="Surface Gravity (log g)",
            color="Spectral Type",
            color_discrete_map={"O": "#9bb0ff", "B": "#aabfff", "A": "#cad7ff",
                              "F": "#f8f7ff", "G": "#fff4ea", "K": "#ffd2a1", "M": "#ffcc6f"},
            title="Hertzsprung-Russell Diagram (Simplified)",
            log_x=True,
            height=400
        )
        fig_hr.update_traces(marker=dict(size=8, opacity=0.7))
        fig_hr.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_hr, use_container_width=True)
    
    with col2:
        fig_sky = px.scatter(
            filtered_df, x="RA (deg)", y="Dec (deg)",
            color="Spectral Type",
            color_discrete_map={"O": "#9bb0ff", "B": "#aabfff", "A": "#cad7ff",
                              "F": "#f8f7ff", "G": "#fff4ea", "K": "#ffd2a1", "M": "#ffcc6f"},
            title="Sky Distribution",
            height=400
        )
        fig_sky.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig_sky, use_container_width=True)
    
    # Histogram
    fig_hist = px.histogram(
        filtered_df, x="Spectral Type",
        category_orders={"Spectral Type": ["O", "B", "A", "F", "G", "K", "M"]},
        color="Spectral Type",
        color_discrete_map={"O": "#9bb0ff", "B": "#aabfff", "A": "#cad7ff",
                          "F": "#f8f7ff", "G": "#fff4ea", "K": "#ffd2a1", "M": "#ffcc6f"},
        title="Stellar Population Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("🧬 Spectral Classification Guide")
    
    selected_type = st.selectbox(
        "Select Spectral Type to Learn More",
        ["O", "B", "A", "F", "G", "K", "M"]
    )
    
    info = get_spectral_info(selected_type)
    
    if info:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="star-card">
                <h2>Type {selected_type} Star</h2>
                <h4>{info['name']}</h4>
                <hr style="border-color: rgba(255,255,255,0.2);">
                <p><strong>💫 Examples:</strong><br>{info['examples']}</p>
                <p><strong>⚖️ Mass:</strong><br>{info['mass']}</p>
                <p><strong>⏱️ Lifespan:</strong><br>{info['lifespan']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Generate spectrum for this type
            wave, flux, temp, color = generate_stellar_spectrum(selected_type)
            
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=wave,
                y=flux,
                mode='lines',
                fill='tozeroy',
                fillcolor=color.replace('#', 'rgba(') + ', 0.3)',
                line=dict(color=color, width=2),
                name=f'Type {selected_type}'
            ))
            
            fig_spec.update_layout(
                title=f"Typical Type {selected_type} Spectrum (T ≈ {temp}K)",
                xaxis_title="Wavelength (Å)",
                yaxis_title="Normalized Flux",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_spec, use_container_width=True)
            
            st.info(f"**Spectral Features:** {info['features']}")
    
    # Temperature scale visualization
    st.subheader("🌡️ Stellar Temperature Scale")
    
    temps = [40000, 20000, 8500, 6500, 5778, 4500, 3000]
    types = ["O", "B", "A", "F", "G", "K", "M"]
    colors = ["#9bb0ff", "#aabfff", "#cad7ff", "#f8f7ff", "#fff4ea", "#ffd2a1", "#ffcc6f"]
    
    fig_scale = go.Figure()
    
    for i, (t, typ, col) in enumerate(zip(temps, types, colors)):
        fig_scale.add_trace(go.Bar(
            x=[typ],
            y=[np.log10(t)],
            marker_color=col,
            name=f'{typ} ({t:,}K)',
            text=f'{t:,}K',
            textposition='outside'
        ))
    
    fig_scale.update_layout(
        title="Surface Temperature by Spectral Type",
        xaxis_title="Spectral Type",
        yaxis_title="Log₁₀(Temperature K)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_scale, use_container_width=True)

with tab4:
    st.subheader("🔬 Spectral Analysis Tools")
    
    analysis_type = st.selectbox(
        "Select Analysis",
        ["Line Depth Measurement", "Continuum Fitting", "Equivalent Width"]
    )
    
    if analysis_type == "Line Depth Measurement":
        st.markdown("""
        ### Measuring Spectral Line Depth
        
        The depth of an absorption line indicates the abundance of the element and 
        the physical conditions in the stellar atmosphere.
        
        **Formula:**
        ```
        Line Depth = 1 - (F_line / F_continuum)
        ```
        """)
        
        # Interactive line measurement
        wave, flux, _, _ = generate_stellar_spectrum("G")
        
        fig_analysis = go.Figure()
        fig_analysis.add_trace(go.Scatter(
            x=wave, y=flux,
            mode='lines',
            name='Spectrum',
            line=dict(color='white')
        ))
        
        # Add measurement lines
        line_center = 15890
        line_idx = np.argmin(np.abs(wave - line_center))
        line_depth = flux[line_idx]
        continuum_level = 1.0
        
        fig_analysis.add_hline(y=continuum_level, line_dash="dash", 
                              annotation_text="Continuum", line_color="green")
        fig_analysis.add_vline(x=line_center, line_dash="dash",
                              annotation_text="Line Center", line_color="red")
        
        # Add measurement annotations
        fig_analysis.add_annotation(
            x=line_center + 50, y=(continuum_level + line_depth) / 2,
            text=f"Depth: {continuum_level - line_depth:.3f}",
            showarrow=True,
            arrowhead=2
        )
        
        fig_analysis.update_layout(
            title="Interactive Line Depth Measurement",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_analysis, use_container_width=True)
    
    elif analysis_type == "Continuum Fitting":
        st.markdown("""
        ### Continuum Normalization
        
        Stellar spectra are normalized by fitting a smooth curve (continuum) through 
        the peaks of the spectrum, then dividing the spectrum by this continuum.
        """)
        
        wave, flux, _, _ = generate_stellar_spectrum("K")
        
        # Simple polynomial continuum fit
        coeffs = np.polyfit(wave, flux, 5)
        continuum = np.polyval(coeffs, wave)
        normalized = flux / continuum
        
        fig_cont = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Raw Spectrum with Continuum", "Normalized Spectrum"))
        
        fig_cont.add_trace(go.Scatter(x=wave, y=flux, mode='lines', name='Raw',
                                     line=dict(color='white')), row=1, col=1)
        fig_cont.add_trace(go.Scatter(x=wave, y=continuum, mode='lines', name='Continuum',
                                     line=dict(color='red', dash='dash')), row=1, col=1)
        
        fig_cont.add_trace(go.Scatter(x=wave, y=normalized, mode='lines', name='Normalized',
                                     line=dict(color='cyan')), row=2, col=1)
        fig_cont.add_hline(y=1, line_dash="dash", row=2, col=1)
        
        fig_cont.update_layout(height=600, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_cont, use_container_width=True)
    
    else:  # Equivalent Width
        st.markdown("""
        ### Equivalent Width
        
        The equivalent width measures the total strength of a spectral line, 
        representing the width of a rectangular line with the same total absorption.
        
        **Formula:**
        ```
        EW = ∫(1 - F(λ)/F_c) dλ
        ```
        """)
        
        wave, flux, _, _ = generate_stellar_spectrum("M")
        
        # Calculate EW for a specific line
        line_center = 16000
        line_width = 100
        mask = (wave > line_center - line_width) & (wave < line_center + line_width)
        
        ew = np.trapz(1 - flux[mask], wave[mask])
        
        fig_ew = go.Figure()
        fig_ew.add_trace(go.Scatter(x=wave, y=flux, mode='lines', name='Spectrum',
                                   line=dict(color='white')))
        fig_ew.add_trace(go.Scatter(
            x=wave[mask], y=flux[mask],
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            mode='none',
            name=f'EW = {ew:.2f} Å'
        ))
        
        fig_ew.update_layout(
            title=f"Equivalent Width Measurement (EW = {ew:.2f} Å)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_ew, use_container_width=True)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌌 Based on <a href="https://github.com/TirtheshJani/Deep-Generative-Model-Stellar-Spectra">
    Deep-Generative-Model-Stellar-Spectra</a> by Tirthesh Jani</p>
    <p>Data: APOGEE Survey (SDSS) 🔭</p>
</div>
""", unsafe_allow_html=True)
