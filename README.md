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

## License

This project is licensed under the GNU General Public License - see the [LICENSE]() file for details.
