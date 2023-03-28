# Audio-Bleeding-Removal
Neural Network for Removal of bleeding in carnatic music (Saraga dataset) for the application of Music Source Separation.

## Dynamic Frame Input Autoencoder (Spectrogram)
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR| 3.226 | 11.745 | 8.57 | 7.319 | __7.715__ |


## Least Square Solution of Bleed Removal (Time Domain)
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | SDR |
|------|-----|
|Overall| ✨ |

## Attention Based
The architecture is tested with artifically bleeded MUSDB18HQ and the bleed is removed.
### Results
| Metrics | Vocal | Bass | Drums | Others | Overall |
|------|-----|-----|-----|-----|-----|
|SDR|  |  |  |  |  |


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/its-rajesh/Audio-Bleeding-Removal/blob/cde41b94a1be385efc46888a04b30a7b82c33375/LICENSE) file for details.
