# Bleeding Removal in Music Signals
Neural Networks for the removal of bleeding, interference, and cross-talk in live recorded music dataset for the application of Music Source Separation (MSS). The standard MSS dataset: MUSDB18HQ is artificially bled internally to simulate the real-world bleeding effect during the training of these models. Various proposed model are listed below and compared with KAMIR (Kernel Additive Modelling for Interference Reduction) algorithm.

| Models | Vocal | Bass | Drums | Others | Overall SDR | Live |
|------|-----|-----|-----|-----|-----|-----|
|[Base line]()| 1.78 | 4.44 | 6.78 | 5.96 | 4.74 | - |
|[KAMIR](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/main/Papers/pratzlich_Kernel_Additive_Modeling_for_interference_reduction_in_multi-channel_music_recordings_icassp2015.pdf)| 6.41 | 6.75 | 6.83 | 5.61 | 6.40 | Y |
|[DI-CAE]()| 1.89 | 5.81 | 6.18 | 4.48 | 4.59 | Y |
|[Optimisation]()| 39.25 | 42.90 | 44.22 | 42.11 | 42.12 | N |
|[t-Unet (IL)]()| 12.05 | 15.05 | 16.255 | 15.69 | __14.76__ | N |
|[S-iWaveUnet]()| 6.50 | 9.84 | 10.85 | 10.32 | 9.38 | Y |

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
