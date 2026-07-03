import streamlit as st
import requests

st.set_page_config(page_title="FaceProtect", layout="wide")

st.title("FaceProtect Dashboard")
st.subheader("AI-Powered Privacy & Content Moderation Gateway")
st.divider()

# --- 1. Top Metrics (Live from API) ---
col1, col2, col3 = st.columns(3)

try:
    stats_response = requests.get("http://host.docker.internal:8000/stats", timeout=5)
    
    if stats_response.status_code == 200:
        stats_data = stats_response.json()
        
        total_embeddings = stats_data.get("total_embeddings", 0)
        blocked_count = stats_data.get("blocked_count", 0)
        trusted_count = stats_data.get("trusted_count", 0) 
    else:
        total_embeddings, blocked_count, trusted_count = 0, 0, 0
        st.warning(f"Failed to load stats. Status: {stats_response.status_code}")

except requests.exceptions.ConnectionError:
    total_embeddings, blocked_count, trusted_count = 0, 0, 0
    st.error("API Connection Error: Stats could not be loaded.")

col1.metric(label="Total Embeddings", value=total_embeddings)
col2.metric(label="Blocked Profiles", value=blocked_count)
col3.metric(label="Trusted Profiles", value=trusted_count)

st.divider()

# --- 2. Image Analysis Section ---
uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_col, _ = st.columns([1, 2])
    with img_col:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Scanning face against Vector DB..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://host.docker.internal:8000/verify", files=files)

                if response.status_code == 200:
                    result = response.json()

                    decision = result.get("decision")
                    similarity = round(result.get("similarity_percent", 0.0), 2)
                    person_id = result.get("person_id")
                    recommendation = result.get("recommendation")

                    display_person = person_id if person_id else "Unknown"

                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Identity", display_person)
                    res_col2.metric("Similarity", f"{similarity}%")
                    
                    if decision == "ACCEPT":
                        res_col3.metric("Decision", "ACCEPT")
                        st.success("Face verified successfully. No unauthorized match found or user is trusted.")
                    elif decision == "REJECT":
                        res_col3.metric("Decision", "REJECT")
                        st.error("Unauthorized face detected. Block policy applied.")
                    else:
                        st.warning(f"Unexpected decision value: {decision}")

                    if recommendation:
                        st.info(f"**Action Taken:** {recommendation.get('primary_action')} — {recommendation.get('reason')}")
                else:
                    st.error(f"API request failed with status code {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Unable to reach the API. Please check if the backend is running on port 8000.")

st.divider()

# --- 3. Trust Lists Management ---
st.subheader("Trust Lists Management")

try:
    trust_response = requests.get("http://host.docker.internal:8000/trust", timeout=5)
    
    if trust_response.status_code == 200:
        trust_data = trust_response.json()
        
        if trust_data:
            st.dataframe(trust_data, use_container_width=True)
        else:
            st.info("No records found in the Trust List.")
    else:
        st.error(f"Failed to load Trust Lists. Status code: {trust_response.status_code}")

except requests.exceptions.ConnectionError:
    st.error("Connection Error: Unable to load Trust Lists. Check if the API is running.")

st.divider()

st.subheader("Manage Trust Status")

action_col1, action_col2, action_col3 = st.columns([2, 1, 1])

with action_col1:
    target_person_id = st.text_input("Enter Person ID to modify:")

with action_col2:
    st.write("") 
    st.write("")
    if st.button("Block Person", type="primary", use_container_width=True):
        if target_person_id:
            try:
                res = requests.patch(f"http://host.docker.internal:8000/trust/{target_person_id}/block")
                if res.status_code == 200:
                    st.success(f"Successfully blocked {target_person_id}. Please refresh to see changes.")
                else:
                    st.error(f"Failed to block. Status: {res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("API Connection Error.")
        else:
            st.warning("Please enter a Person ID first.")

with action_col3:
    st.write("") 
    st.write("")
    if st.button("Unblock Person", use_container_width=True):
        if target_person_id:
            try:
                res = requests.patch(f"http://host.docker.internal:8000/trust/{target_person_id}/unblock")
                if res.status_code == 200:
                    st.success(f"Successfully unblocked {target_person_id}. Please refresh to see changes.")
                else:
                    st.error(f"Failed to unblock. Status: {res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("API Connection Error.")
        else:
            st.warning("Please enter a Person ID first.")

st.divider()

# --- 4. Smart Protect (Auto-Blur) ---
st.subheader("Smart Protect (Auto-Blur)")
st.markdown("Upload an image to automatically blur unauthorized faces based on privacy policies.")

smart_file = st.file_uploader("Upload image for Smart Protect", type=["jpg", "jpeg", "png"], key="smart_protect_uploader")

if smart_file is not None:
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.image(smart_file, caption="Original Image", use_container_width=True)

    if st.button("Run Smart Protect", type="primary", use_container_width=True):
        with st.spinner("Applying privacy filters..."):
            try:
                files = {"file": (smart_file.name, smart_file.getvalue(), smart_file.type)}
                smart_response = requests.post("http://host.docker.internal:8000/smart_protect", files=files)

                if smart_response.status_code == 200:
                    sp_decision = smart_response.headers.get("X-Decision", "UNKNOWN")
                    sp_person_id = smart_response.headers.get("X-Person-Id", "Unknown")
                    sp_similarity = smart_response.headers.get("X-Similarity", "0.0")

                    with img_col2:
                        st.image(smart_response.content, caption=f"Processed Image", use_container_width=True)
                    
                    sp_info1, sp_info2, sp_info3 = st.columns(3)
                    sp_info1.metric("Smart Decision", sp_decision)
                    sp_info2.metric("Identity", sp_person_id)
                    sp_info3.metric("Similarity", f"{sp_similarity}%")
                    
                    if sp_decision == "REJECT":
                        st.info("Face was rejected and automatically blurred to protect privacy.")
                    elif sp_decision == "ACCEPT":
                        st.success("Face accepted. No privacy filters applied.")

                else:
                    st.error(f"Smart Protect request failed. Status: {smart_response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Unable to reach the API.")

st.divider()