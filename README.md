# Audio-Bleeding-Removal
Neural Network for Removal of bleeding in carnatic music (Saraga dataset) for the application of Music Source Separation.

## Dynamic Frame Input Deep Convolutional Autoencoder (Spectrogram)
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |


## Bleed Removal Optimized Algorithm (Time Domain - Limited Setup)
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | SDR |
|------|-----|
|Overall| __50.2__ |


## Tailed U-Net Interference Learning Network (Time Domain)
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
