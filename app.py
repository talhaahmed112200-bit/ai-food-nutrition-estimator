
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="AI Food Nutrition Estimator",
    page_icon="🍎",
    layout="centered"
)

# Approximate nutrition per 100 grams.
# These are illustrative estimates for common foods.
NUTRITION = {
    "apple": {"calories": 52, "protein": 0.3, "carbs": 13.8, "fat": 0.2},
    "banana": {"calories": 89, "protein": 1.1, "carbs": 22.8, "fat": 0.3},
    "pizza": {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    "hamburger": {"calories": 295, "protein": 17.0, "carbs": 24.0, "fat": 14.0},
    "cheeseburger": {"calories": 303, "protein": 17.0, "carbs": 30.0, "fat": 14.0},
    "hot dog": {"calories": 290, "protein": 10.0, "carbs": 24.0, "fat": 18.0},
    "french fries": {"calories": 312, "protein": 3.4, "carbs": 41.0, "fat": 15.0},
    "orange": {"calories": 47, "protein": 0.9, "carbs": 11.8, "fat": 0.1},
    "broccoli": {"calories": 34, "protein": 2.8, "carbs": 6.6, "fat": 0.4},
    "carrot": {"calories": 41, "protein": 0.9, "carbs": 9.6, "fat": 0.2},
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3},
    "chicken": {"calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6},
    "egg": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0},
    "ice cream": {"calories": 207, "protein": 3.5, "carbs": 24.0, "fat": 11.0},
    "cake": {"calories": 350, "protein": 5.0, "carbs": 50.0, "fat": 15.0},
    "donut": {"calories": 452, "protein": 4.9, "carbs": 51.0, "fat": 25.0},
    "strawberry": {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3},
    "granny smith": {"calories": 52, "protein": 0.3, "carbs": 13.8, "fat": 0.2},
}

# Load the pretrained MobileNetV2 model only when needed
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

st.title("🍎 AI Food Image and Nutrition Estimator")
st.write(
    "Upload a food image to let an AI model predict the food "
    "and estimate its nutritional values."
)

st.info(
    "This is a student demonstration. Recognition and nutrition "
    "estimates may be inaccurate, especially for mixed dishes."
)

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded food image", width="stretch")

        # Choose the serving size
        grams = st.number_input(
            "Estimated serving size (grams)",
            min_value=1,
            max_value=2000,
            value=100,
            step=25
        )

        if st.button("Analyze Food"):
            with st.spinner("AI is analyzing your image..."):
                model = load_model()

                # Prepare image for MobileNetV2
                resized = image.resize((224, 224))
                image_array = np.array(resized, dtype=np.float32)
                image_array = np.expand_dims(image_array, axis=0)
                image_array = (
                    tf.keras.applications.mobilenet_v2.preprocess_input(
                        image_array
                    )
                )

                # Predict the most likely ImageNet classes
                predictions = model.predict(image_array, verbose=0)
                decoded = (
                    tf.keras.applications.mobilenet_v2
                    .decode_predictions(predictions, top=5)[0]
                )

                st.subheader("AI Predictions")

                for _, label, score in decoded:
                    st.write(
                        f"**{label.replace('_', ' ').title()}** "
                        f"— {score * 100:.1f}% model score"
                    )

                # Match the predicted labels with our nutrition table
                all_labels = [
                    label.lower().replace("_", " ")
                    for _, label, _ in decoded
                ]

                matched_food = None

                for label in all_labels:
                    for food in NUTRITION:
                        if food in label or label in food:
                            matched_food = food
                            break
                    if matched_food:
                        break

                st.subheader("Estimated Nutrition")

                if matched_food:
                    # Calculate nutrition for the selected serving size
                    nutrition = NUTRITION[matched_food]
                    factor = grams / 100

                    st.write(f"Food matched: **{matched_food.title()}**")
                    st.write(f"Serving size: **{grams} g**")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Calories",
                        f"{nutrition['calories'] * factor:.1f} kcal"
                    )
                    col2.metric(
                        "Protein",
                        f"{nutrition['protein'] * factor:.1f} g"
                    )

                    col3, col4 = st.columns(2)

                    col3.metric(
                        "Carbohydrates",
                        f"{nutrition['carbs'] * factor:.1f} g"
                    )
                    col4.metric(
                        "Fat",
                        f"{nutrition['fat'] * factor:.1f} g"
                    )

                    st.caption(
                        "Nutrition values are approximate estimates "
                        "based on the matched food and serving size."
                    )
                else:
                    st.warning(
                        "No matching food was found in the built-in "
                        "nutrition table. The AI prediction may still "
                        "be useful, but nutrition was not calculated."
                    )

    except Exception as error:
        st.error(f"Something went wrong: {error}")
        st.write(
            "Check that the image is valid and the required "
            "Python libraries are installed."
        )
