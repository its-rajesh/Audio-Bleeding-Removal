# Bleeding Removal in Music Signals

While recording instrument sounds in live concerts, dedicated microphones are placed to capture their corresponding source. Practically, these microphones pick up the other sources as well, as they are not acoustically shielded, leading to bleeding effects. These are also called leakage, interference, or crosstalk. This problem is closely related to source separation but is simpler as we have multiple sources of information available. Till now, traditional signal processing based methods have been developed, and to our knowledge none of the neural network based approaches were proposed (till 2022). This could be because of lack of dataset to train the models.

I have utilised the standard MSS dataset: MUSDB18HQ. The MUSDB18HQ dataset is artificially bleeded in multiple ways to provide bleeding effects. 
1. Linear mixtures
2. Adding time delays and room impulse responses and by creating convolute mixtures
3. Creating artificial room and stimulating the real-world effects

I have proposed various models listed below and compared with the baseline KAMIR (Kernel Additive Modelling for Interference Reduction) algorithm. The median Source to Distortion Ratio (SDR) are shown.


| Models | Vocal | Bass | Drums | Others | Overall SDR | 
|------|-----|-----|-----|-----|-----|
|[Reference]()| 1.86 | 4.44 | 6.78 | 5.96 | 5.82 | 
|[KAMIR](https://ieeexplore.ieee.org/abstract/document/7178036)| 13.84 | 6.75 | 6.83 | 5.61 | 7.00 |
|[DI-CAE]()| 1.89 | 5.81 | 6.18 | 4.48 | 6.92 | 
|[Optimisation]()*| 39.25 | 42.90 | 44.22 | 42.11 | 42.12 |
|[t-UNet]()| 8.05 | 9.05 | 8.255 | 6.69 | 8.83 |
|[f-UNet]()| 6.50 | 9.84 | 10.85 | 10.32 | 9.38 | 
|[df-UNet]()| 6.50 | 9.84 | 10.85 | 10.32 | __11.54__ |


*works only for limited setups (linear mixtures). All the proposed models were submitted for publication and separate links & code will be updated soon.

Publications:

[1] Rajesh R and Padmanabhan Rajan. "Neural Networks for Interference Reduction in Multi-track Recordings." In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2023. (link will be provided soon)


[2] "Bleeding Removal in Music Signals Via Optimization"

[3] 



## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
