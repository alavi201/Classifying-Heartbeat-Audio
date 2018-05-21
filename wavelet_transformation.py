import pywt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

audio_file = 'artifact__201105060108.wav'
ts, sample_rate = librosa.load('heartbeat-sounds/set_a/'+audio_file)
#ts = [2, 56, 3, 22, 3, 4, 56, 7, 8, 9, 44, 23, 1, 4, 6, 2]

(ca, cd) = pywt.dwt(ts,'haar')

cat = pywt.threshold(ca, np.std(ca)/2, 'soft')
cdt = pywt.threshold(cd, np.std(cd)/2, 'soft')

ts_rec = pywt.idwt(cat, cdt, 'haar')

print(ts_rec)
print(ts_rec.shape)

plt.close('all')

plt.subplot(211)
# Original coefficients
plt.plot(ca, '--*b')
plt.plot(cd, '--*r')
# Thresholded coefficients
plt.plot(cat, '--*c')
plt.plot(cdt, '--*m')
plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
plt.grid(1)

plt.subplot(212)
plt.plot(ts)
plt.hold(1)
plt.plot(ts_rec, 'r')
plt.legend(['original signal', 'reconstructed signal'])
plt.grid(1)
plt.show()