import streamlit as st
import pickle
import pandas as pd
from main import get_clean_data
import plotly.graph_objects as go
import numpy as np


def add_sidebar():
    st.sidebar.title("Cell Nuclei Measurements:")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"), # se = standard error
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {} # store the avg values to help create the chart and the output window

    for lables, keys in slider_labels:
        input_dict[keys] = st.sidebar.slider(
        lables,
        min_value=float(0),
        max_value=float(data[keys].max()),
        value=float(data[keys].mean()), # default value
        )

    return input_dict

def get_scaled_values(input_dict):
    # data = get_clean_data()
    # scaler = StandardScaler()
    # scaled_values = scaler.fit_transform(data)
    # return scaled_values
  data = get_clean_data()
  
  x = data.drop(columns=['diagnosis'])
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = x[key].max()
    min_val = x[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict



def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
            'Smoothness', 'Compactness', 
            'Concavity', 'Concave Points',
            'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[ input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    # fig.show() not gonna be used as streamlit has its own way of displaying the chart
    return fig

def add_precictions(input_data):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    if prediction == 0:
        st.write("The breast mass is benign.")
    else:
        st.write("The breast mass is malignant.")

    # st.write(prediction)
    st.write(f"the probability of the mass being benign is {model.predict_proba(input_array_scaled)[0][0]:.2f}")
    st.write(f"the probability of the mass being Maliganant is {model.predict_proba(input_array_scaled)[0][1]:.2f}")

def main():
    
    st.set_page_config(
        page_title="Breast Cancer Predictor App",
        page_icon="Female Doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
        )
    
    input_data = add_sidebar()
    # st.write(input_data)
    
    
    with st.container():
        st.title("Breast Cancer Predictor App")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.write("this is column 1")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        st.subheader("Cell Cluster Predictor:")
        add_precictions(input_data)

    st.warning("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


if __name__ == "__main__":
    main()