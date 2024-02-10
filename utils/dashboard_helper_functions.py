# dashboard_helper_functions.py
import pickle
import gc
import base64
import os

# Print the current working directory
current_directory = os.getcwd()
print(f"Current Directory: {current_directory}")

def Decode_shap_values(base64_global_shap_values):
    # Decode from base64
    serialized_global_shap_values = base64.b64decode(base64_global_shap_values)

    # Deserialize using pickle
    global_shap_values = pickle.loads(serialized_global_shap_values)
    gc.collect()

    return global_shap_values

def Get_Target_Decision(client_data, feature_importance, decision, cluster):
    client_data['Cluster'] = int(cluster)
    client_data['TARGET'] = int(decision)
    top_features_names = feature_importance.col_name.tolist()
    gc.collect()
    return top_features_names, client_data