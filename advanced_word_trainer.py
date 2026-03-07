"""
ADVANCED WORD TRAINER
======================
Bidirectional LSTM + Multi-head Attention + Residual Connections
Optimized for few-shot learning (10-20 sequences per word)
"""
import numpy as np
import json
import os
from glob import glob

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pickle
except ImportError:
    print("Install: pip install tensorflow scikit-learn")
    exit()


class AdvancedWordTrainer:
    def __init__(self):
        print("\n" + "="*80)
        print("ADVANCED WORD TRAINER")
        print("State-of-the-Art Architecture for Few-Shot Learning")
        print("="*80)
        print("\nUsing float32 for CPU compatibility")

        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self):
        X, y = [], []

        data_dir = "sign_data/advanced_words"
        if not os.path.exists(data_dir):
            print(f"\nNo data found in {data_dir}")
            print("Run: python advanced_word_collector.py")
            return None, None, None

        print("\nLoading augmented data...")

        for word_folder in os.listdir(data_dir):
            folder_path = f"{data_dir}/{word_folder}"
            if not os.path.isdir(folder_path):
                continue

            json_files = glob(f"{folder_path}/*.json")
            print(f"   {word_folder}: {len(json_files)} samples (augmented)")

            for file in json_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                X.append(data['sequence'])
                y.append(data['label'])

        if len(X) == 0:
            return None, None, None

        max_len = max(len(seq) for seq in X)
        print(f"\n   Max sequence length: {max_len} frames")

        feature_dim = len(X[0][0])
        print(f"   Feature dimension: {feature_dim}")

        X_padded = []
        for seq in X:
            if len(seq) < max_len:
                pad = [[0] * feature_dim for _ in range(max_len - len(seq))]
                seq = list(seq) + pad
            X_padded.append(seq[:max_len])

        X_array = np.array(X_padded, dtype=np.float32)
        y_array = np.array(y)

        print(f"\nLoaded: {len(X_array)} samples")
        print(f"   Shape: {X_array.shape}")

        return X_array, y_array, sorted(list(set(y)))

    def attention_layer(self, inputs, units=256):
        """Multi-head attention — units=256 matches BiLSTM(128) output shape"""
        query = layers.Dense(units)(inputs)
        key   = layers.Dense(units)(inputs)
        value = layers.Dense(units)(inputs)

        scores = tf.matmul(query, key, transpose_b=True)
        scale  = tf.cast(tf.math.sqrt(float(units)), scores.dtype)
        scores = scores / scale
        weights = tf.nn.softmax(scores, axis=-1)

        attended = tf.matmul(weights, value)
        return attended

    def build_advanced_model(self, num_classes, timesteps, feature_dim):
        print("\nBuilding Advanced Model...")
        print("   Architecture: BiLSTM + Multi-head Attention + Residual")

        inputs = layers.Input(shape=(timesteps, feature_dim))

        # BiLSTM layer 1 — outputs 512 features (256*2)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # BiLSTM layer 2 — outputs 256 features (128*2)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Attention — units=256 matches BiLSTM layer 2 output for residual Add()
        attended = self.attention_layer(x, units=256)

        # Residual connection (both tensors are shape [batch, timesteps, 256])
        x = layers.Add()([x, attended])
        x = layers.BatchNormalization()(x)

        # Global pooling
        x_max = layers.GlobalMaxPooling1D()(x)
        x_avg = layers.GlobalAveragePooling1D()(x)
        x = layers.Concatenate()([x_max, x_avg])

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            alpha=0.0001
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        # CategoricalCrossentropy supports label_smoothing in all Keras versions
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        print(f"\n   Model built:")
        print(f"      BiLSTM: 2 layers (256, 128 units -> 256 output)")
        print(f"      Attention: units=256 (matches residual)")
        print(f"      Pooling: Max + Average")
        print(f"      Total params: {model.count_params():,}")
        print(f"      Loss: CategoricalCrossentropy (label_smoothing=0.1)")

        return model

    def train(self):
        print("\n" + "="*80)
        print("TRAINING ADVANCED WORD MODEL")
        print("="*80)

        X, y, labels = self.load_data()
        if X is None:
            return

        num_classes = len(labels)
        print(f"\nClasses: {num_classes}")
        print(f"   {labels}")

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.15,
            random_state=42,
            stratify=y_encoded
        )

        # One-hot encode — required for CategoricalCrossentropy
        y_train_oh = keras.utils.to_categorical(y_train, num_classes)
        y_test_oh  = keras.utils.to_categorical(y_test,  num_classes)

        print(f"\nDataset Split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")
        print(f"   Avg per class: {len(X_train) / num_classes:.1f} training samples")

        model = self.build_advanced_model(num_classes, X.shape[1], X.shape[2])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20,
                restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=7, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{self.models_dir}/best_model.h5',
                monitor='val_accuracy', save_best_only=True, verbose=0
            )
        ]

        print("\nTRAINING...")
        print("   Using advanced techniques for few-shot learning")
        print("   This may take 3-10 minutes\n")

        history = model.fit(
            X_train, y_train_oh,
            validation_split=0.2,
            epochs=150,
            batch_size=8,
            verbose=1,
            callbacks=callbacks
        )

        model.load_weights(f'{self.models_dir}/best_model.h5')

        print("\nEVALUATING...")
        loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)

        print(f"\n{'='*80}")
        print(f"TEST ACCURACY: {acc*100:.2f}%")
        print(f"{'='*80}")

        if acc > 0.90:
            print("   EXCELLENT! Advanced techniques working perfectly!")
        elif acc > 0.85:
            print("   Very Good! Few-shot learning successful!")
        elif acc > 0.80:
            print("   Good! Consider collecting 5 more sequences per word")
        else:
            print("   Collect 10-15 more sequences per word for better results")

        model_path  = f"{self.models_dir}/advanced_words_model.h5"
        labels_path = f"{self.models_dir}/advanced_words_labels.pkl"

        model.save(model_path)
        with open(labels_path, 'wb') as f:
            pickle.dump(encoder, f)

        print(f"\nSAVED:")
        print(f"   Model:  {model_path}")
        print(f"   Labels: {labels_path}")

        best_epoch   = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = max(history.history['val_accuracy'])

        print(f"\nTRAINING SUMMARY:")
        print(f"   Best epoch:       {best_epoch}")
        print(f"   Best val acc:     {best_val_acc*100:.2f}%")
        print(f"   Test accuracy:    {acc*100:.2f}%")
        print(f"   Total epochs run: {len(history.history['loss'])}")

        print(f"\nPER-CLASS PERFORMANCE:")
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        for i, label in enumerate(encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).mean()
                print(f"   {label:15s}: {class_acc*100:5.1f}%")

        print("\nAdvanced training complete!")
        return acc


def main():
    trainer = AdvancedWordTrainer()

    print("\n" + "="*80)
    print("ADVANCED WORD TRAINER")
    print("="*80)
    print("\nPress ENTER to start training...")
    input()

    trainer.train()

    print("\n" + "="*80)
    print("NEXT STEP: python advanced_word_interpreter.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()