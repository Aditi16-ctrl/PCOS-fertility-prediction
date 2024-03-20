import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('pcos_prediction model.sav', 'rb'))

def prediction(input_data):
    input_data = (1.99, 494.08, 1.99)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is fertile'
    else:
        return 'The person is infertile'

def main():
    st.title('Prediction pcos model')

    I_beta_HCG_mIU_mL = st.text_input('I-beta-HCG(mIU/mL)')
    II_beta_HCG_mIU_mL = st.text_input('II-beta-HCG(mIU/mL)')
    AMH_ng_mL = st.text_input('AMH(ng/mL)')

    diagnosis = ''

    if st.button('Result'):
        diagnosis = prediction([I_beta_HCG_mIU_mL, II_beta_HCG_mIU_mL, AMH_ng_mL])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
