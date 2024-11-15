import streamlit as st
import subprocess
from pathlib import Path
import time
import os

def main():
    st.title("LDR to HDR Converter")
    st.write("Drag and drop a .jpg file to convert it to .hdr format.")

    # File uploader for drag-and-drop
    uploaded_file = st.file_uploader("Choose a LDR file", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded file
        st.image(uploaded_file, caption="Uploaded LDR image.", use_container_width=True)

        # Save the uploaded file to a unique temporary location
        temp_filename = f"temp_{int(time.time())}.jpg"  # Unique filename based on timestamp
        input_path = Path(temp_filename)

        # Save the uploaded file locally
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display a message while converting
        st.write("Converting image...")

        # Run the conversion script
        conversion_command = ["python", "conversion.py", str(input_path)]
        result = subprocess.run(conversion_command, capture_output=True, text=True)

        # Display any output or errors from the conversion process
        if result.stdout:
            st.write("Conversion output:", result.stdout)
        
        if result.stderr:
            st.error("Conversion errors:")
            st.write(result.stderr)

        # Check for the .hdr file in the current directory
        output_files = list(Path.cwd().glob("*.hdr"))

        if output_files:
            output_path = output_files[0]  # Assuming the first .hdr file is the result
            st.success("Conversion complete! Here is your .hdr file:")

            # Provide a download button for the .hdr file
            st.download_button(
                label="Download .hdr file",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="application/octet-stream"
            )

            # Clean up temporary files (input .jpg and output .hdr)
            try:
                input_path.unlink()  # Delete the temporary .jpg file
                output_path.unlink()  # Delete the .hdr file after download
            except Exception as e:
                st.error(f"Error cleaning up files: {str(e)}")
        else:
            st.error("Conversion failed. No .hdr file found.")

if __name__ == "__main__":
    main()
