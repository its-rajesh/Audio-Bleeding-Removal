# Bleeding Removal in Music Signals

When capturing instrument sounds in live concerts, dedicated microphones are strategically placed to record each source. However, these microphones often pick up unintended sounds from other sources due to the lack of acoustic shielding, resulting in bleeding effects. These effects are also known as leakage, interference, or crosstalk. While this problem is closely related to source separation, it is somewhat simpler because we have multiple sources of information available. Until 2022, traditional signal processing-based methods were the primary approach to address this issue. Surprisingly, no neural network-based solutions were proposed, possibly due to the scarcity of suitable training datasets.

To tackle this challenge, I leveraged the MUSDB18HQ dataset, a standard benchmark dataset for music source separation. This dataset was artificially enhanced to introduce various bleeding effects using the following techniques:

- **Linear Mixtures**: Basic mixing of audio sources.
- **Time Delays and Room Impulse Responses**: Simulating real-world conditions with convoluted mixtures.
- **Artificial Room Effects**: Creating artificial room environments to replicate authentic recording conditions.

### Performance of Proposed Models

I developed several bleeding removal models and compared their performance against the baseline KAMIR (Kernel Additive Modeling for Interference Reduction) algorithm. The median source-to-distortion ratio (SDR) for each model is presented below:


| Models | Vocal | Bass | Drums | Others | Overall SDR | 
|------|-----|-----|-----|-----|-----|
|[Reference]()| 1.86 | 4.44 | 6.78 | 5.96 | 5.82 | 
|[KAMIR](https://ieeexplore.ieee.org/abstract/document/7178036)| 13.84 | 6.75 | 6.83 | 5.61 | 7.00 |
|[DI-CAE]()| 1.89 | 5.81 | 6.18 | 4.48 | 6.92 | 
|[Optimisation]()*| 39.25 | 42.90 | 44.22 | 42.11 | 42.12 |
|[t-UNet]()| 8.05 | 9.05 | 8.255 | 6.69 | 8.83 |
|[f-UNet]()| 6.50 | 9.84 | 10.85 | 10.32 | 9.38 | 
|[df-UNet]()| 6.50 | 9.84 | 10.85 | 10.32 | __11.54__ |


Note: The 'Optimization*' model works only for limited setups (linear mixtures). All proposed models have been submitted for publication, and separate links and codes will be provided soon.


### Listening Test

To be updated...

### Publications:

[1] Rajesh R and Padmanabhan Rajan. "Neural Networks for Interference Reduction in Multi-track Recordings." In 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2023. (A link will be provided soon.) 

[2] "Bleeding Removal in Music Signals Via Optimization" (Details will be updated)

[3] (Details will be updated)



## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
