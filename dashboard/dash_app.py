#dash_app.py
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

from functools import lru_cache
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import time
import pandas as pd
import shap
import sys
import pickle
import base64
from PIL import Image
import os

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))



sys.path.append('./utils')
from dashboard_helper_functions import Decode_shap_values, Get_Target_Decision

#logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session_state attributes
if "top_features" not in st.session_state:
    st.session_state.top_features = None
if "fig" not in st.session_state:
    st.session_state.fig = None

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

#def Get_Target_Decision(client_data, feature_importance, decision, cluster):
#    client_data['Cluster'] = int(cluster)
#    client_data['TARGET'] = int(decision)
#    top_features_names = feature_importance.col_name.tolist()
#    return top_features_names, client_data

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def fetch_shap_values():
    api_url = "http://localhost:8001/shap_values"  # Update with the correct port
    return requests.get(api_url)

#@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def run_api(SK_ID_CURR):
    api_url = "http://localhost:8001/predict"  # Update with the correct port
    json_data = json.dumps({"data": {"SK_ID_CURR": str(SK_ID_CURR)}})
    return requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def get_clients_data(selected_features):
    api_url = "http://localhost:8001/all_clients"  # Get all clients data
    selected_features= ['TARGET', 'Cluster'] + selected_features

    json_data = json.dumps({"data": {"Features": str(selected_features)}})
    response = requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        print('***********************************')
        print('response_data', type(response_data))
        print('***********************************')


        # Access the 'clients_data' key in the response
        df_clients = pd.DataFrame(response_data['clients_data'])
        df_clients = df_clients.reindex(columns=selected_features)

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}\n{response.text}")

    return df_clients

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def fetch_client_info(SK_ID_CURR):
    try:
        # Call the API to get the Scores
        api_url = "http://localhost:8001/client"  # Update with the correct port
        json_data = json.dumps({"data": {"SK_ID_CURR": str(SK_ID_CURR)}})
        response = requests.post(api_url, data=json_data, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            # Access the JSON content of the response
            json_response = response.json()

            # Create a dictionary to hold the data
            client_info = {
                'Gender': json_response['Gender'],
                'Pronoun': json_response['Pronoun'],
                'Age': json_response['Age'],
                'Family Status': json_response['Family Status'],
                'Housing Type': json_response['Housing Type'],
                'Income Total': json_response['Income Total'],
                'Income Type': json_response['Income Type'],
                'Employed since': json_response['Employed since'],
                'Occupation Type': json_response['Occupation Type'],
                'Credit Amount': json_response['Credit Amount'],
                'Annuity Amount': json_response['Annuity Amount']
            }

            return client_info

    except Exception as e:
        return print('error in fetch_client_info', str(e))

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def global_feature_importance_bar_chart(sorted_features):
    st.write("Global Feature Importance:")
        # Create a bar chart using Plotly
    fig = go.Figure(go.Bar(
        x=sorted_features['Global Importance'],
        y=sorted_features['Feature'],
        orientation='h'
    ))

    fig.update_layout(
        title_text="Top 10 Global Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        width=600,
    )   

@st.cache_data(hash_funcs={np.ndarray: lambda x: x.view(dtype=np.uint8)})
def create_credit_score_gauge(predicted_prob):
    try:
        # Convert probability to a score between 0 and 100
        credit_score = int(predicted_prob[0] * 100)
        Threshold = 100*(1 - 0.4801)

        # Define labels for likely to pay or not
        pay_labels = ["Not Likely to Pay", "Likely to Pay"]
        pay_label = pay_labels[(credit_score >= Threshold)]
        if pay_label == "Not Likely to Pay":
            Decision_txt = "Rejected"
        else:
            Decision_txt = "Accepted"

        # Define ranking labels with four levels
        ranking_labels = ["Needs work", "Needs some work", "Good", "Excellent"]
        ranking = ranking_labels[min(3, credit_score // 25)]  # Adjusted based on four levels

        # Determine colors based on the position of the credit score relative to the threshold

        # Create a gauge chart
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 0.5], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickmode': 'array', 'tickvals': [0, int(Threshold / 2), Threshold, int(1.5 * Threshold), 100]},
                'bar': {'color': 'purple'},
                'steps': [
                    {'range': [0, Threshold / 2], 'color': '#FFA500'},
                    {'range': [Threshold / 2, Threshold], 'color': '#fafa6e'},
                    {'range': [Threshold, 1.5 * Threshold], 'color': '#59c187'},
                    {'range': [1.5 * Threshold, 100], 'color': '#208f8b'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': Threshold,
                }
            }
        ))

        # Update the plot with the final title
        fig.update_layout(
            title_text=f"Credit Score",
            title_x=0.19,
            title_y=0.5,
            title_font_size=24,  # Increased font size
            width=1000,  # Increased width
            height=300,  # Increased height
            margin=dict(l=10, r=10, b=10, t=10)  # Adjusted margin
        )

        for i in range(0, credit_score + 1, 2):
            fig.update_traces(value=i)
            time.sleep(0.01)

        # Add ranking and likelihood annotations in the middle of the plot
        annotation_text = f"<b>{ranking}</b><br>{  pay_label}"
        fig.add_annotation(
            x=0.185,
            y=-0.04,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=16),
        )

        return fig, ranking, Decision_txt, credit_score, Threshold

    except Exception as e:
        return print('error in create_credit_score_gauge', str(e))

#@st.cache_data()
def distri_plot(merged_df, feature, title):
    # Plot position of the client's value
    fig_value_position = go.Figure()
    # Add histogram trace for the selected feature
    fig_value_position.add_trace(go.Histogram(x=merged_df[merged_df['TARGET'] == 0][feature], name=feature + ' destribution on class 0', nbinsx=100, histnorm='percent', marker=dict(color='green')))
    fig_value_position.add_trace(go.Histogram(x=merged_df[merged_df['TARGET'] == 1][feature], name=feature + ' destribution on class 1', nbinsx=100, histnorm='percent', marker=dict(color='orange')))
    
    # Add vertical line for the client's value position
    client_value_position = merged_df[feature].iloc[-1]

    fig_value_position.add_shape(go.layout.Shape(type='line', x0=client_value_position, x1=client_value_position,
            y0=0, y1=1, yref='paper', line=dict(color='red', width=2, dash='dot'), name=f'Client {feature}'))

    # Add text annotation to describe the vertical line
    fig_value_position.add_annotation(go.layout.Annotation(text=f'Client {feature} Value', x=client_value_position,
            y=1.02, xref='x', yref='paper', showarrow=False, font=dict(size=10),))
   
    fig_value_position.update_layout(title_text=title, title=dict(x=0.15),  legend=dict(x=0, y=-0.2, orientation="h"))
    
    return fig_value_position

#@st.cache_data()
def get_shap_values():
    response = fetch_shap_values()

    if response.status_code == 200:
        result = response.json()
        feature_names = result['feature_names']

        # Deserialize shap values using pickle
        encoded_shap_values = result['global_shap_values']

        serialized_global_shap_values = base64.b64decode(encoded_shap_values)
        global_shap_values = pickle.loads(serialized_global_shap_values)

        # Deserialize Explanation using pickle
        encoded_exp_0 = result['exp_0']

        serialized_exp_0= base64.b64decode(encoded_exp_0)
        exp_0 = pickle.loads(serialized_exp_0)

        #Deserialize Feature importance
        encoded_featue_importance = result['featue_importance']
        serialized_featue_importance = base64.b64decode(encoded_featue_importance)
        featue_importance = pickle.loads(serialized_featue_importance)


    #return Decode_shap_values(encoded_shap_values)
    return global_shap_values, feature_names, exp_0, featue_importance

def main():
    st.set_page_config(page_title="P7_Client_Dashboard", page_icon="📊", layout="wide")

    # Specify the absolute path to the logo image
    #st.session_state.logo1 = st.session_state.logo.crop((left, top, right, bottom)).resize(newsize)
    ## Open the image using the absolute path
    ##st.session_state.resized_logo = st.session_state.logo.resize((20, 20), Image.Resampling.BICUBIC)
    #st.image(st.session_state.logo1, caption='Prêt à dépenser')
    st.markdown("<h1 style='text-align: center;'>CREDIT SCORE ANALYZER2</h1>", unsafe_allow_html=True)


    global_shap_values, feature_names, exp_0, featue_importance = get_shap_values()

    # Get client ID from the user
    with st.sidebar:
        
        ## Specify the absolute path to the logo image
        #st.session_state.logo_path = os.path.join(current_dir, '../img/logo.png')
        ## Open the image using the absolute path
        #st.session_state.logo = Image.open(st.session_state.logo_path)
#
        #st.image(st.session_state.logo, caption='Prêt à dépenser')
        analysis_type = st.radio(
            "Please select an analysis to perform",
            ('Client analysis', 'Global analysis'))
        
        if analysis_type == 'Client analysis':
            SK_ID_CURR = st.text_input("Enter loan request ID:")
            # Convert the input to numpy.int64
            try:
                SK_ID_CURR = np.int64(SK_ID_CURR)
            except ValueError:
                st.sidebar.warning("Please enter a valid loan request ID.")
                st.stop()  # Stop further execution if input is invalid
        #else:
        #    with st.sidebar:
        #        st.warning(f"Sorry! Unable to get Scores from the API. Client not found.")
    
    if analysis_type == 'Client analysis':
        # # Create columns for layout
        col1, col2, col3 = st.columns([2, 1, 2])

        # Call the API to get the Scores
        response = run_api(SK_ID_CURR)

        if response.status_code == 200:
            ## Fetch client Info
            client_info = fetch_client_info(SK_ID_CURR)

            #st.session_state.shap_values= client_data

            result = response.json()
            probability = result['probability']
            shap_values = Decode_shap_values(result['global_shap_values'])

            with col1:
                container1 = st.container(border=True)               
                # Appliquez le style HTML en utilisant st.markdown
                container1.markdown(
                    f"""
                    <div>
                        <h3 style="font-size: 1.2em; text-align: center;">Client Information</h3>
                        <p style="font-size: 1.1em;"><b>The client is</b> a {client_info['Gender']}, {client_info['Age']} and {client_info['Pronoun']} is {client_info['Family Status']}.</p>
                        <p style="font-size: 1.1em;"><b>Occupation:</b> {client_info['Pronoun']} is a {client_info['Occupation Type']}, {client_info['Income Type']} since {client_info['Employed since']}.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            ## Display performance graph
            with col3:
                fig, ranking, pay_label, credit_score, Threshold =create_credit_score_gauge(probability)
                st.plotly_chart(fig)

            with col1:
                # Condition pour définir la couleur de fond
                background_color = "#00FF7F" if credit_score > Threshold else "#CD5C5C"

                st.markdown(
                            f"""
                            <div style="background-color: {background_color}; padding: 13px; border-radius: 5px;">
                                <h3 style="text-align: center;">Outcome</h3>
                                <p style="font-size: 1.2em;"><b>Client's ranking:</b> {ranking}</p>
                                <p style="font-size: 1.2em;"><b>Credit score:</b> {credit_score}</p>
                                <p style="font-size: 1.2em;"><b>Decision:</b> {pay_label}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                            )




            ## Create tabs
            tabs = st.tabs(["Client Features importance", "Features distribution", "Additional Client Information"])

            with tabs[0]:
                # Fetch client Info           
                st.session_state.top_features  = st.slider("You can select up to 10 features", min_value=1, max_value=10, value=5)
                cluster = np.int64(result["cluster"])
                decision = np.int64(result["decision"])
                idx = result['idx']             

                
                tab0_col1, tab0_col2, tab0_col3 = st.columns([1, 4, 1])
                with tab0_col2:
                    st.pyplot(shap.plots.waterfall(exp_0[idx], max_display=st.session_state.top_features))

            with tabs[1]:

                encoded_client_data = result['client_data']
                serialized_client_data= base64.b64decode(encoded_client_data)
                client_data = pickle.loads(serialized_client_data)
                
                
                ##waterfall_plot
                top_features_names, new_client_data = Get_Target_Decision(client_data, featue_importance, decision, cluster)

                selected_features = st.multiselect('Select features to analyse', top_features_names)
                 # Merge the two DataFrames
                st.session_state.df_clients = get_clients_data(top_features_names)

                #features_client_df = pd.DataFrame(sorted_client_features_values, columns=st.session_state.df_clients.columns.tolist())
                features_client_df = new_client_data[st.session_state.df_clients.columns.tolist()]


                merged_df = pd.concat([st.session_state.df_clients, features_client_df])
                merged_df['Cluster'] = merged_df['Cluster'].astype(int)

                for feature in selected_features:
                    col11, col12 = st.columns([1, 1])
                    with col11:
                        title = f'Distribution of feature {feature} on all the exisiting clients in the DB'
                        fig_value_position = distri_plot(merged_df, feature, title)
                        st.plotly_chart(fig_value_position)
                    with col12:
                        title = f'Distribution of feature {feature} on similar group of clients'
                        fig_value_position = distri_plot(merged_df.iloc[np.where(merged_df.Cluster == cluster)[0]], feature, title)
                        st.plotly_chart(fig_value_position)
            with tabs[2]:
                    client_info = pd.DataFrame({
                        "Details": ['Details'],
                        "Housing Type": [client_info['Housing Type']],
                        "Income Total": [client_info['Income Total']],
                        "Credit Amount": [client_info['Credit Amount']],
                        "Annuity Amount": [client_info['Annuity Amount']]
                    })

                    # Set "Housing Type" as the index
                    client_info.set_index("Details", inplace=True)

                    st.table(client_info)

            #with tabs[3]:
            #    st.pyplot(shap.summary_plot(shap_values, top_features_names, plot_type='bar', show=True, color_bar=True, max_display=15))
            #    st.pyplot(shap.summary_plot(shap_values[0], top_features_names, plot_type='violin', show=True, color_bar=True, max_display=15))
            #    st.pyplot(shap.summary_plot(shap_values[1], top_features_names, plot_type='violin', show=True, color_bar=True, max_display=15))
            #    st.pyplot(shap.summary_plot(shap_values[0], top_features_names, plot_type='dot', show=True, color_bar=True, max_display=15))
            #    st.pyplot(shap.summary_plot(shap_values[1], top_features_names, plot_type='dot', show=True, color_bar=True, max_display=15))
            #    #shap.summary_plot(shap_values, st.session_state.df_clients, plot_type='bar', show=False, color_bar=False, max_display=15)

        
    if analysis_type == 'Global analysis':
        st.session_state.top_global_features = st.slider("You can select up to 20 features", min_value=1, max_value=20, value=10)
        with st.container(height = 600, border=True):
            st.title("Average features impact on the prediction")
            st.pyplot(shap.summary_plot(global_shap_values, feature_names, plot_type='bar', show=True, color_bar=True, plot_size=0.18,
                                    title='Average features impact on the prediction', max_display=st.session_state.top_global_features))
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.title("Definition of a successful loan request")
            st.session_state.violin_blue = shap.summary_plot(global_shap_values[0], feature_names, plot_type='violin', 
                                            show=True, title='Definition of a successful client', cmap='cool', #color='cool',
                                            color_bar=True, max_display=st.session_state.top_global_features)
            
            st.pyplot(st.session_state.violin_blue)
        with col_g2:
            st.title("Definition of a refused loan request")
            st.pyplot(shap.summary_plot(global_shap_values[1], feature_names, plot_type='violin', show=True, title='Definition of a successful client', color='red', color_bar=True, max_display=st.session_state.top_global_features))
        #st.pyplot(shap.summary_plot(global_shap_values, feature_names, plot_type='dot', show=True, color_bar=True, max_display=st.session_state.top_global_features))


if __name__ == "__main__":
    main()