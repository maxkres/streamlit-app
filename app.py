import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Car Price Prediction", layout="wide")

@st.cache_resource
def load_model():
    with open('model_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts['model']
ohe = artifacts['ohe']
num_cols = artifacts['num_cols']
cat_cols = artifacts['cat_cols']
medians = artifacts['medians']
y_train = artifacts['y_train']
X_train_full = artifacts['X_train_full']

st.title("Прогнозирование цены автомобиля")

tab1, tab2, tab3 = st.tabs(["EDA", "Прогноз", "Веса модели"])

with tab1:
    st.header("Exploratory Data Analysis")
    
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Распределение цены")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_train['selling_price'], bins=50, edgecolor='black')
        ax.set_xlabel('Цена')
        ax.set_ylabel('Количество')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Цена vs Год выпуска")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df_train['year'], df_train['selling_price'], alpha=0.5)
        ax.set_xlabel('Год')
        ax.set_ylabel('Цена')
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Распределение по типу топлива")
        fig, ax = plt.subplots(figsize=(8, 5))
        df_train['fuel'].value_counts().plot(kind='barh', ax=ax)
        ax.set_xlabel('Количество')
        st.pyplot(fig)
    
    with col4:
        st.subheader("Корреляция числовых признаков")
        numeric_cols = ['year', 'selling_price', 'km_driven']
        corr = df_train[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("Прогнозирование цены")
    
    input_method = st.radio("Выберите способ ввода данных:", ["Ручной ввод", "Загрузка CSV"])
    
    if input_method == "Ручной ввод":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Год выпуска", min_value=1980, max_value=2024, value=2015)
            km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
        
        with col2:
            fuel = st.selectbox("Тип топлива", ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
            seller_type = st.selectbox("Тип продавца", ['Individual', 'Dealer', 'Trustmark Dealer'])
        
        with col3:
            transmission = st.selectbox("Коробка передач", ['Manual', 'Automatic'])
            owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
        
        mileage = st.number_input("Расход топлива (kmpl)", min_value=0.0, value=20.0)
        engine = st.number_input("Объем двигателя (CC)", min_value=0, value=1200)
        max_power = st.number_input("Мощность (bhp)", min_value=0.0, value=80.0)
        seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)
        
        if st.button("Предсказать цену"):
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'seats': [seats]
            })
            
            X_num = input_data[num_cols].fillna(medians)
            X_cat = input_data[cat_cols].astype('string').fillna('missing')
            X_ohe = ohe.transform(X_cat)
            X_full = pd.concat([X_num.reset_index(drop=True),
                               pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(cat_cols))],
                              axis=1)
            
            prediction = model.predict(X_full)[0]
            st.success(f"Прогнозируемая цена: ₹{prediction:,.2f}")
    
    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:", df)
            
            if st.button("Предсказать цены"):
                X_num = df[num_cols].fillna(medians)
                X_cat = df[cat_cols].astype('string').fillna('missing')
                X_ohe = ohe.transform(X_cat)
                X_full = pd.concat([X_num.reset_index(drop=True),
                                   pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(cat_cols))],
                                  axis=1)
                
                predictions = model.predict(X_full)
                df['predicted_price'] = predictions
                st.write("Результаты прогнозирования:", df)
                
                csv = df.to_csv(index=False)
                st.download_button("Скачать результаты", csv, "predictions.csv", "text/csv")

with tab3:
    st.header("Важность признаков")
    
    coef = pd.DataFrame({
        'feature': X_train_full.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in coef['coefficient']]
    ax.barh(coef['feature'], coef['coefficient'], color=colors)
    ax.set_xlabel('Коэффициент')
    ax.set_title('Топ-20 важных признаков модели')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    st.pyplot(fig)
    
    st.subheader("Все коэффициенты")
    all_coef = pd.DataFrame({
        'feature': X_train_full.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    st.dataframe(all_coef)
