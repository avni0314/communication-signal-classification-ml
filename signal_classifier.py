import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("Program started")

# -------------------------
# SIGNAL GENERATION
# -------------------------

def generate_sine():
    t = np.linspace(0, 1, 500)
    return np.sin(2 * np.pi * 5 * t)

def generate_square():
    t = np.linspace(0, 1, 500)
    return signal.square(2 * np.pi * 5 * t)

def generate_bpsk():
    bits = np.random.choice([-1, 1], 500)
    carrier = np.cos(2 * np.pi * 5 * np.linspace(0, 1, 500))
    return bits * carrier

def generate_noise():
    return np.random.normal(0, 1, 500)

def add_noise(sig, noise_level):
    noise = np.random.normal(0, noise_level, len(sig))
    return sig + noise


# -------------------------
# FEATURE EXTRACTION
# -------------------------

def extract_features(sig):

    # Time domain features
    mean = np.mean(sig)
    std = np.std(sig)
    max_val = np.max(sig)
    min_val = np.min(sig)

    # Frequency domain using FFT
    fft_vals = np.abs(np.fft.fft(sig))
    fft_mean = np.mean(fft_vals)
    fft_std = np.std(fft_vals)
    fft_max = np.max(fft_vals)

    return [
        mean,
        std,
        max_val,
        min_val,
        fft_mean,
        fft_std,
        fft_max
    ]


# -------------------------
# CREATE DATASET
# -------------------------

X = []
y = []

for _ in range(200):

    s = generate_sine()
    s = add_noise(s, 0.5)
    X.append(extract_features(s))
    y.append(0)

    s = generate_square()
    s = add_noise(s, 0.5)
    X.append(extract_features(s))
    y.append(1)

    s = generate_bpsk()
    s = add_noise(s, 0.5)
    X.append(extract_features(s))
    y.append(2)

    s = generate_noise()
    X.append(extract_features(s))
    y.append(3)

X = np.array(X)
y = np.array(y)

labels = ["Sine", "Square", "BPSK", "Noise"]


# -------------------------
# TRAIN MODEL
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


# -------------------------
# TEST MODEL
# -------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Confusion Matrix")



# -------------------------
# TEST NEW SIGNAL
# -------------------------

test_signal = add_noise(generate_bpsk(), 0.5)

features = extract_features(test_signal)

prediction = model.predict([features])

print("Predicted signal:", labels[prediction[0]])


# -------------------------
# PLOT SIGNAL AND FFT
# -------------------------

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(test_signal)
plt.title("Time Domain Signal")

plt.subplot(1,2,2)
plt.plot(np.abs(np.fft.fft(test_signal)))
plt.title("Frequency Domain (FFT)")



# -------------------------
# ACCURACY vs NOISE LEVEL
# -------------------------

noise_levels = [0, 0.2, 0.5, 1, 2]

accuracies = []

for noise in noise_levels:

    X_noise = []
    y_noise = []

    for _ in range(100):

        s = add_noise(generate_sine(), noise)
        X_noise.append(extract_features(s))
        y_noise.append(0)

        s = add_noise(generate_square(), noise)
        X_noise.append(extract_features(s))
        y_noise.append(1)

        s = add_noise(generate_bpsk(), noise)
        X_noise.append(extract_features(s))
        y_noise.append(2)

        s = generate_noise()
        X_noise.append(extract_features(s))
        y_noise.append(3)

    preds = model.predict(X_noise)

    acc = accuracy_score(y_noise, preds)

    accuracies.append(acc)


# Plot accuracy vs noise

plt.figure()
plt.plot(noise_levels, accuracies, marker='o')
plt.title("Accuracy vs Noise Level")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.grid()


# -------------------------
# ACCURACY vs SNR (Wireless Channel Simulation)
# -------------------------

def add_awgn(signal, snr_db):

    signal_power = np.mean(signal**2)

    snr_linear = 10**(snr_db/10)

    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise


snr_levels = [-5, 0, 5, 10, 15, 20]

snr_accuracies = []

for snr in snr_levels:

    X_snr = []
    y_snr = []

    for _ in range(100):

        s = add_awgn(generate_sine(), snr)
        X_snr.append(extract_features(s))
        y_snr.append(0)

        s = add_awgn(generate_square(), snr)
        X_snr.append(extract_features(s))
        y_snr.append(1)

        s = add_awgn(generate_bpsk(), snr)
        X_snr.append(extract_features(s))
        y_snr.append(2)

        s = generate_noise()
        X_snr.append(extract_features(s))
        y_snr.append(3)

    preds = model.predict(X_snr)

    acc = accuracy_score(y_snr, preds)

    snr_accuracies.append(acc)


# Plot Accuracy vs SNR
plt.figure()
plt.plot(snr_levels, snr_accuracies, marker='o')

plt.title("Classification Accuracy vs SNR (Wireless Channel Simulation)")

plt.xlabel("SNR (dB)")

plt.ylabel("Accuracy")

plt.grid()

plt.show()