# Bleeding Removal in Music Signals
Neural Networks for the removal of bleeding, interference, and cross-talk in live recorded music (saraga dataset) for the application of Music Source Separation (MSS). The standard MSS dataset: MUSDB18HQ is artificially bled internally to simulate the real-world bleeding effect during the training of these models.

| Models | Vocal | Bass | Drums | Others | Overall SDR |
|------|-----|-----|-----|-----|-----|
|[Base line]()| 1.78 | 4.44 | 6.78 | 5.96 | __4.74__ |
|[KAMIR]()| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |
|[DI-CAE]()| 1.89 | 5.81 | 6.18 | 4.48 | __4.59__ |
|[Optimisation]()| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |
|[t-Unet (IL)]()| 39.25 | 42.90 | 44.22 | 42.11 | __42.12__ |
|[S-iWaveUnet]()| 6.50 | 9.84 | 10.85 | 10.32 | __9.38__ |

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
