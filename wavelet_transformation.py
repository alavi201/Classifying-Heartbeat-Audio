import pywt
import numpy as np

def wavelet_transformation(audio_data):
    (ca, cd) = pywt.dwt(audio_data,'haar')

    cat = pywt.threshold(ca, np.std(ca)/2, 'soft')
    cdt = pywt.threshold(cd, np.std(cd)/2, 'soft')

    audio_data_transformed = pywt.idwt(cat, cdt, 'haar')

    return audio_data_transformed
