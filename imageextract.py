import streamlit as st 
import pytesseract
from PIL import Image 

def extract_text(image):
    try:
        text=pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Extract text from images:",page_icon=":camera:")
    st.header("Extract Text from images ")

    images=st.file_uploader("Upload your images here:",type=['jpg','png','jpeg'],accept_multiple_files=True)

    if images is not None: 
        for i in images:
            image = Image.open(i)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Extract", key="extract_button"): 
                extracted_text=extract_text(image)
                st.write('The text in image says:')
                st.write(extracted_text)
        


if __name__=="__main__":
    main()
