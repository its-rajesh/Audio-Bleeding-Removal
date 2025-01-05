import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import h5py
import librosa as lb
from tqdm import tqdm

def build_generator(input_shape):
    input_tensor = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, (3, 3), dilation_rate=1, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(16, (3, 3), dilation_rate=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    en1 = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(32, (3, 3), dilation_rate=2, padding="same")(en1)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(32, (3, 3), dilation_rate=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    en2 = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(64, (3, 3), dilation_rate=4, padding="same")(en2)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(64, (3, 3), dilation_rate=4, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    en3 = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), dilation_rate=4, padding="same")(en3)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3, 3), dilation_rate=4, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x)
    en4 = layers.Dropout(0.3)(x)
        

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), padding="same")(en4)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    bottle = layers.LeakyReLU()(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding="same")(bottle)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, en4])
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(1, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, en3])
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=(1, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, en2])
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(16, (3, 3), strides=(1, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, en1])
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.LeakyReLU()(x)
    
    #Matching input layer
    x = layers.Conv2DTranspose(44, (3, 3), strides=(1, 2), padding="same")(x) 
    x = layers.Conv2D(44, (3, 3), padding="same")(x)
    mask = layers.LeakyReLU()(x)
        
    # Final Convolution
    output = layers.Multiply()([mask, input_tensor])

    return Model(inputs=input_tensor, outputs=output, name="Generator")


def build_discriminator(input_shape):
    X_bleed = layers.Input(shape=input_shape)
    Y_clean = layers.Input(shape=input_shape)
    concatenated = layers.Concatenate()([X_bleed, Y_clean])

    x = concatenated
    for _ in range(4):
        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D((1, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs=[X_bleed, Y_clean], outputs=output, name="Discriminator")


def reconstruct_time_domain(magnitude, phase, frame_length=1024, frame_step=512):
    """
    Reconstruct time-domain signals using magnitude and phase.
    Args:
        magnitude: Magnitude STFT, shape (batch, channels, freq, time).
        phase: Phase STFT, shape (batch, channels, freq, time).
        frame_length: Length of each frame for iSTFT.
        frame_step: Step size between frames for iSTFT.
    Returns:
        Time-domain signal, shape (batch, channels, time_samples).
    """
    stft_complex = tf.cast(magnitude, tf.complex64) * tf.exp(1j * tf.cast(phase, tf.complex64))
    batch_size, channels, freq, time = tf.shape(stft_complex)
    time_domain = []
    for b in range(batch_size):
        channel_signals = []
        for c in range(channels):
            signal = tf.signal.inverse_stft(stft_complex[b, c, :, :], frame_length=frame_length, frame_step=frame_step)
            channel_signals.append(signal)
        time_domain.append(tf.stack(channel_signals, axis=0))
    return tf.stack(time_domain, axis=0)

def time_domain_mse(y_pred, y_true, pred_phase, true_phase):
    """
    Compute MSE loss in time domain.
    Args:
        y_pred: Predicted magnitude STFT, shape (batch, channels, freq, time).
        y_true: Ground truth magnitude STFT, shape (batch, channels, freq, time).
        pred_phase: Phase corresponding to y_pred, shape (batch, channels, freq, time).
        true_phase: Phase corresponding to y_true, shape (batch, channels, freq, time).
    Returns:
        MSE loss in time domain.
    """
    y_pred_td = reconstruct_time_domain(y_pred, pred_phase)
    y_true_td = reconstruct_time_domain(y_true, true_phase)
    return tf.reduce_mean(tf.square(y_pred_td - y_true_td))

def sisdr_time_domain_pit(y_pred, y_true, pred_phase, true_phase, epsilon=1e-9):
    """
    Compute SISDR with Permutation Invariant Training in the time domain.
    Args:
        y_pred: Predicted magnitude STFT, shape (batch, channels, freq, time).
        y_true: Ground truth magnitude STFT, shape (batch, channels, freq, time).
        pred_phase: Phase corresponding to y_pred, shape (batch, channels, freq, time).
        true_phase: Phase corresponding to y_true, shape (batch, channels, freq, time).
    Returns:
        Average SISDR across all batches.
    """
    y_pred_td = reconstruct_time_domain(y_pred, pred_phase)
    y_true_td = reconstruct_time_domain(y_true, true_phase)

    batch_size, channels, time = tf.shape(y_true_td)
    sisdr_list = []
    for b in range(batch_size):
        pairwise_sisdr = []
        for i in range(channels):
            s_target = tf.reduce_sum(y_pred_td[b, i, :] * y_true_td[b, :, :], axis=-1) / \
                       (tf.reduce_sum(y_true_td[b, :, :] ** 2, axis=-1) + epsilon)
            s_target = s_target[..., None] * y_true_td[b, :, :]
            e_noise = y_pred_td[b, i, :] - s_target
            sisdr = 10 * tf.math.log(tf.reduce_sum(s_target ** 2, axis=-1) /
                                     (tf.reduce_sum(e_noise ** 2, axis=-1) + epsilon)) / tf.math.log(10.0)
            pairwise_sisdr.append(sisdr)
        sisdr_list.append(-tf.reduce_mean(pairwise_sisdr)) #negative for training
    return tf.reduce_mean(sisdr_list)

def discriminator_loss(real_output, fake_output):
    """
    Binary cross-entropy loss for the discriminator.
    Args:
        real_output: Predicted probabilities for real samples (discriminator output on real data).
        fake_output: Predicted probabilities for fake samples (discriminator output on generator output).
    Returns:
        Total discriminator loss.
    """
    real_labels = tf.ones_like(real_output)  # Real samples labeled as 1
    fake_labels = tf.zeros_like(fake_output)  # Fake samples labeled as 0

    real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_output, from_logits=False)
    fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output, from_logits=False)

    return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

def generator_loss(fake_output):
    """
    Binary cross-entropy loss for the generator.
    Args:
        fake_output: Discriminator predictions for generator outputs.
    Returns:
        Generator loss.
    """
    real_labels = tf.ones_like(fake_output)  # Generator wants the discriminator to think generated samples are real
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_labels, fake_output, from_logits=False))

generator = build_generator((3, 1024, 44))
discriminator = build_discriminator((3, 1024, 44))

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(X_bleed, Y_clean, theta):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        G_Xbleed = generator(X_bleed, training=True)
        real_output = discriminator([X_bleed, Y_clean], training=True)
        fake_output = discriminator([X_bleed, G_Xbleed], training=True)

        gen_loss = generator_loss(fake_output, G_Xbleed, Y_clean, theta)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



#### DATASET LOADING

Xtrain_stft = np.load('/home/anchal/Desktop/rajesh/research2/cGANIR/Xtrain_stft.npy')
Ytrain_stft = np.load('/home/anchal/Desktop/rajesh/research2/cGANIR/Ytrain_stft.npy')
print(Xtrain_stft.shape, Ytrain_stft.shape)

def compute_mag_and_phase(stft):
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase

Xtrain, Xphases = compute_mag_and_phase(Xtrain_stft)
Ytrain, Yphases = compute_mag_and_phase(Ytrain_stft)
print('Magnitude and phase:', Xtrain.shape, Xphases.shape)

# Preparing Dataset
Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)  # Ensure proper tensor format
Ytrain = tf.convert_to_tensor(Ytrain, dtype=tf.float32)
print('Conversion to tensors done')

# Training Loop
EPOCHS = 5
theta = [0.25, 0.25, 0.25, 0.25]  # Example weights for different loss terms
batch_size = 4

print('Preparing batches')
train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain)).batch(batch_size)  # Batch size of 2 for small dataset

print('Initiate Training..')
for epoch in range(EPOCHS):
    for step, (X_bleed, Y_clean) in enumerate(train_dataset):
        # Corresponding phases
        X_phase = Xphases[step * batch_size:(step + 1) * batch_size]
        Y_phase = Yphases[step * batch_size:(step + 1) * batch_size]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            G_Xbleed = generator(X_bleed, training=True)
            real_output = discriminator([X_bleed, Y_clean], training=True)
            fake_output = discriminator([X_bleed, G_Xbleed], training=True)

            # Loss calculations
            gen_loss = (
                generator_loss(fake_output) +
                theta[0] * tf.reduce_mean(tf.abs(G_Xbleed - Y_clean)) +
                theta[1] * time_domain_mse(G_Xbleed, Y_clean, X_phase, Y_phase) +
                theta[2] * tf.reduce_mean(tf.abs(Y_clean[:, 0, :, :] - Y_clean[:, 1, :, :])) +  # Example cross-talk
                theta[3] * sisdr_time_domain_pit(G_Xbleed, Y_clean, X_phase, Y_phase)
            )
            disc_loss = discriminator_loss(real_output, fake_output)

        # Compute Gradients
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply Gradients
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    print(f"Epoch {epoch + 1}, Step {step + 1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")
    
 
# Define paths for saving the models
generator_save_path = "/home/anchal/Desktop/rajesh/research2/cGANIR/generator"
discriminator_save_path = "/home/anchal/Desktop/rajesh/research2/cGANIR/discriminator"

# Save the generator
generator.save(generator_save_path)

# Save the discriminator
discriminator.save(discriminator_save_path)

print(f"Models saved to '{generator_save_path}' and '{discriminator_save_path}'")

