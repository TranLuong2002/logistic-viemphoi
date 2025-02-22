import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# Load models
with open('/Pneumonia_Detection/models/logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

cnn_model = load_model('/Pneumonia_Detection/models/cnn_model.h5')

# Streamlit UI
st.title("Chương trình phát hiện bệnh viêm phổi")

st.markdown("""
    Tải lên một hoặc nhiều hình ảnh chụp X-quang và chương trình này sẽ dự đoán **Viêm phổi** hay **Bình thường**. 
""")

# Chức năng xử lý trước ảnh đã upload
def preprocess_image(image):
    try:
        # Chuyển đổi hình ảnh sang thang độ xám nếu là RGB
        if image.mode == "RGB":
            image = image.convert("L")
        # Thay đổi kích thước hình ảnh thành 150x150 (kích thước đầu vào cho các mô hình)
        image = image.resize((150, 150))
        # Chuyển đổi hình ảnh thành một mảng gọn gàng và chuẩn hóa các giá trị pixel
        img_array = np.array(image) / 255.0
        img_cnn = img_array.reshape(1, 150, 150, 1)

        # Làm phẳng hình ảnh cho hồi quy logistic
        img_flat = img_array.flatten().reshape(1, -1)
        return img_cnn, img_flat
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None

# upload file 
uploaded_files = st.file_uploader("Tải lên 1 hoặc nhiều hình ảnh X-quang hỗ trợ(JPG,PNG,JPEG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# kiểm tra hình ảnh đã upload chưa
if uploaded_files:
    images = []  # List to hold loaded images
    for file in uploaded_files:
        try:
            img = Image.open(file)
            images.append(img)
        except Exception as e:
            st.error(f"Error loading file {file.name}: {e}")

    # Hiển thị bản xem trước của tất cả hình ảnh
    st.subheader("Xem trước hình ảnh đã tải lên")
    cols = st.columns(min(len(images), 4))  # Display up to 4 images per row
    for i, img in enumerate(images):
        with cols[i % len(cols)]:
            st.image(img, caption=f"Image {i + 1}", use_column_width=True)

    # nút button xác nhận
    if st.button("Phân loại hình ảnh"):
        results = []  # To store results
        for i, img in enumerate(images):
            st.write(f"Phân loại hình ảnh {i + 1}...")
            img_cnn, img_flat = preprocess_image(img)
            if img_cnn is not None and img_flat is not None:
                # mô hình dự đoán CNN
                cnn_preds = cnn_model.predict(img_cnn)
                cnn_preds = (cnn_preds > 0.5).astype(int)
                # dự đoán logistic
                lr_preds = lr_model.predict(img_flat)

                # Append results
                results.append({
                    "Image": f"Hình ảnh {i + 1}",
                    "CNN": "Viêm phổi" if cnn_preds == 0 else "Bình thường",
                    "Logistic Regression": "Viêm phổi" if lr_preds == 0 else "Bình thường"
                })
            else:
                results.append({
                    "Image": f"Image {i + 1}",
                    "CNN": "Error in processing",
                    "Logistic Regression": "Error in processing"
                })

        # Display results
        st.subheader("Prediction Results")
        for res in results:
            st.write(f"**{res['Image']}**")
            st.write(f"- CNN Prediction: {res['CNN']}")
            st.write(f"- Logistic Regression Prediction: {res['Logistic Regression']}")

else:
    st.info("Vui lòng tải lên một hoặc nhiều hình ảnh để phân loại.")
