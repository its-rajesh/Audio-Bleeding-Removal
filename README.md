# Bleeding Removal in Music Signals
Neural Networks for the removal of bleeding, interference, and cross-talk in live recorded music (saraga dataset) for the application of Music Source Separation (MSS). The standard MSS dataset: MUSDB18HQ is artificially bled internally to simulate the real-world bleeding effect during the training of these models.

| Models | Vocal | Bass | Drums | Others | Overall SDR |
|------|-----|-----|-----|-----|-----|
|[KAMIR]()| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |
|[DI-CAE]()| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |
|[Optimisation]()| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |
|[t-Unet]()| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |
|[S-WaveUnet]()| 6.50 | 9.84 | 10.85 | 10.32 | __9.38__ |


## Model 1: Dynamic Frame Input Deep Convolutional Autoencoder (Spectrogram)
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |


## Model 2: Bleed Removal Optimized Algorithm (Time Domain - Limited Setup)
| Metrics | SDR |
|------|-----|
|Overall| __50.2__ |


## Model 3: Truncated U-Net Interference Learning Network (Time Domain)
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |


## Model 4: Wave U-Net (Time Domain)
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 6.50 | 9.84 | 10.85 | 10.32 | __9.38__ |

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
