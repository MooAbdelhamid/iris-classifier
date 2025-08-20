import streamlit as st
import pandas as pd
import joblib

#Load the pre-trained model
model = joblib.load('..\models\decision_tree_iris_model.pkl')

#Title of the app
st.title("Iris Flower Species Prediction")

#User input for features
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=5.0)


if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
        
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    st.write(f"The predicted species is: {prediction[0]}")
    # Display image
    if prediction[0] == 'Iris-setosa':
        st.image('images/iris-setosa.jpg', caption='Iris Setosa')
    elif prediction[0] == 'Iris-versicolor':
        st.image('images/iris-versicolor.jpg', caption='Iris Versicolor')
    else :
        st.image('images/iris-virginica.jpg', caption='Iris Virginica')