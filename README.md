# Machine Learning Based Communication Signal Classification under Noisy Wireless Channel

This project implements a machine learning system to classify communication signals under realistic noisy wireless channel conditions. The system uses both time-domain and frequency-domain features extracted using Fast Fourier Transform (FFT) and evaluates classification robustness across different Signal-to-Noise Ratio (SNR) levels.

## Features

- Generation of communication signals:
  - Sine wave
  - Square wave
  - BPSK (Binary Phase Shift Keying)
  - Noise

- Wireless channel simulation:
  - Additive White Gaussian Noise (AWGN)
  - SNR-based noise modeling

- Feature extraction:
  - Time-domain features (mean, standard deviation, max, min)
  - Frequency-domain features using FFT

- Machine learning classification:
  - Random Forest classifier
  - Multi-class signal classification

- Performance evaluation:
  - Classification accuracy
  - Confusion matrix
  - Accuracy vs Noise Level analysis
  - Accuracy vs SNR analysis

## Results

- Achieved classification accuracy of up to 99.3%
- Successfully classified communication signals under noisy wireless channel conditions

## Technologies Used

- Python
- NumPy
- SciPy
- scikit-learn
- matplotlib

## Author

Avni Bansal  
Electronics and Communication Engineering Student

