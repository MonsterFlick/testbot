import streamlit as st

def check_and_download():
    st.title("No Found")
    st.error("Your system is not capable of downloading the AI model.")
    st.write("Please ensure your system meets the following requirements:")
    st.markdown("""
    - **GPU:** NVIDIA RTX 3080 or higher
    - **VRAM:** 10GB or more
    - **Storage:** 20GB free space
    - **RAM:** 16GB or more
    """)

# Call the function to display the message
check_and_download()
