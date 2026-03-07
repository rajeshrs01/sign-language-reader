"""
hello 

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
except ImportError:
        print("❌ Install: pip install tensorflow scikit-learn")
        exit()


class UltraRobustTrainer:
        def __init__(self):
            self.data_dir = "sign_data"
            self.models_dir = "models"
            os.makedirs(self.models_dir, exist_ok=True)
        
        def load_letters(self):
            """Load letter data"""
            X, y = [], []
            
            letters_dir = f"{self.data_dir}/letters"
            if not os.path.exists(letters_dir):
                return None, None, None
            
            for letter_folder in os.listdir(letters_dir):
                folder_path = f"{letters_dir}/{letter_folder}"
                if not os.path.isdir(folder_path):
                    continue
                
                json_files = glob(f"{folder_path}/*.json")
                print(f"Loading {letter_folder}: {len(json_files)} samples")
                
                for file in json_files:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    X.append(data['landmarks'])
                    y.append(data['label'])
            
            if len(X) == 0:
                return None, None, None
            
            return np.array(X), np.array(y), sorted(list(set(y)))
        
        def load_words(self):
            """Load word data - ULTRA ROBUST version"""
            raw_sequences = []
            labels = []
            
            words_dir = f"{self.data_dir}/words"
            if not os.path.exists(words_dir):
                return None, None, None
            
            print("\n🔍 ULTRA-ROBUST DATA LOADING")
            print("="*60)
            
            # Step 1: Load all raw data
            for word_folder in os.listdir(words_dir):
                folder_path = f"{words_dir}/{word_folder}"
                if not os.path.isdir(folder_path):
                    continue
                
                json_files = glob(f"{folder_path}/*.json")
                print(f"📁 {word_folder}: {len(json_files)} files")
                
                for file in json_files:
                    try:
                        with open(file, 'r') as f:
                            data = json.load(f)
                        raw_sequences.append(data['sequence'])
                        labels.append(data['label'])
                    except Exception as e:
                        print(f"   ⚠️  Skipping {file}: {e}")
            
            if len(raw_sequences) == 0:
                return None, None, None
            
            print(f"\n✅ Loaded {len(raw_sequences)} sequences")
            
            # Step 2: Analyze dimensions
            print("\n🔍 ANALYZING DATA DIMENSIONS...")
            all_frame_sizes = set()
            all_sequence_lengths = set()
            
            for seq_idx, seq in enumerate(raw_sequences):
                all_sequence_lengths.add(len(seq))
                for frame in seq:
                    if isinstance(frame, (list, tuple)):
                        all_frame_sizes.add(len(frame))
                    else:
                        print(f"   ⚠️  Sequence {seq_idx}: Frame is not a list/tuple: {type(frame)}")
            
            print(f"   Sequence lengths found: {sorted(all_sequence_lengths)}")
            print(f"   Frame sizes found: {sorted(all_frame_sizes)}")
            
            # Step 3: Determine target dimensions
            max_frame_size = max(all_frame_sizes)
            target_frame_size = 126 if max_frame_size > 63 else 63
            max_seq_length = max(all_sequence_lengths)
            
            print(f"\n✅ TARGET DIMENSIONS:")
            print(f"   Frame size: {target_frame_size} ({'both hands' if target_frame_size == 126 else 'single hand'})")
            print(f"   Sequence length: {max_seq_length} frames")
            
            # Step 4: Normalize EVERYTHING explicitly
            print(f"\n🔧 NORMALIZING DATA...")
            normalized_sequences = []
            problems = 0
            
            for seq_idx, raw_seq in enumerate(raw_sequences):
                normalized_seq = []
                
                # Process each frame
                for frame_idx, raw_frame in enumerate(raw_seq):
                    # Convert to list if needed
                    if not isinstance(raw_frame, list):
                        raw_frame = list(raw_frame)
                    
                    # Ensure it's the right size
                    if len(raw_frame) == target_frame_size:
                        normalized_frame = raw_frame
                    elif len(raw_frame) < target_frame_size:
                        # Pad with zeros
                        normalized_frame = raw_frame + [0.0] * (target_frame_size - len(raw_frame))
                    else:
                        # Truncate (shouldn't happen)
                        normalized_frame = raw_frame[:target_frame_size]
                        problems += 1
                    
                    # Convert to floats and validate
                    try:
                        normalized_frame = [float(x) for x in normalized_frame]
                    except:
                        # If conversion fails, use zeros
                        normalized_frame = [0.0] * target_frame_size
                        problems += 1
                    
                    # Double-check size
                    assert len(normalized_frame) == target_frame_size, f"Frame {frame_idx} in seq {seq_idx} is wrong size: {len(normalized_frame)}"
                    
                    normalized_seq.append(normalized_frame)
                
                # Pad sequence to max length
                while len(normalized_seq) < max_seq_length:
                    normalized_seq.append([0.0] * target_frame_size)
                
                # Truncate if too long
                if len(normalized_seq) > max_seq_length:
                    normalized_seq = normalized_seq[:max_seq_length]
                
                # Final validation
                assert len(normalized_seq) == max_seq_length, f"Sequence {seq_idx} has wrong length: {len(normalized_seq)}"
                for frame in normalized_seq:
                    assert len(frame) == target_frame_size, f"Sequence {seq_idx} has wrong frame size"
                
                normalized_sequences.append(normalized_seq)
            
            if problems > 0:
                print(f"   ⚠️  Fixed {problems} problematic frames")
            
            print(f"   ✅ All sequences normalized to ({max_seq_length}, {target_frame_size})")
            
            # Step 5: Convert to numpy with explicit dtype
            print(f"\n🔧 CONVERTING TO NUMPY ARRAY...")
            
            try:
                # Build array explicitly, row by row
                X_array = np.zeros((len(normalized_sequences), max_seq_length, target_frame_size), dtype=np.float32)
                
                for seq_idx, seq in enumerate(normalized_sequences):
                    for frame_idx, frame in enumerate(seq):
                        X_array[seq_idx, frame_idx, :] = frame
                
                print(f"   ✅ Success! Shape: {X_array.shape}")
                print(f"      ({X_array.shape[0]} sequences, {X_array.shape[1]} frames, {X_array.shape[2]} features)")
                
                # Verify no NaN or Inf
                if np.any(np.isnan(X_array)):
                    print(f"   ⚠️  WARNING: Found NaN values, replacing with zeros")
                    X_array = np.nan_to_num(X_array)
                
                if np.any(np.isinf(X_array)):
                    print(f"   ⚠️  WARNING: Found Inf values, replacing with zeros")
                    X_array = np.nan_to_num(X_array)
                
                return X_array, np.array(labels), sorted(list(set(labels)))
                
            except Exception as e:
                print(f"\n❌ CONVERSION FAILED: {e}")
                print("\n🔍 DETAILED DIAGNOSIS:")
                
                # Check each sequence
                for i in range(min(5, len(normalized_sequences))):
                    seq = normalized_sequences[i]
                    print(f"\n   Sequence {i}:")
                    print(f"      Length: {len(seq)}")
                    frame_sizes = [len(frame) for frame in seq]
                    unique_sizes = set(frame_sizes)
                    print(f"      Frame sizes: {unique_sizes}")
                    
                    if len(unique_sizes) > 1:
                        print(f"      ❌ INCONSISTENT! Sizes: {frame_sizes[:10]}...")
                        for j, frame in enumerate(seq[:5]):
                            print(f"         Frame {j}: {len(frame)} values, type: {type(frame)}")
                            if len(frame) > 0:
                                print(f"            First value: {frame[0]}, type: {type(frame[0])}")
                    else:
                        print(f"      ✓ All frames have {list(unique_sizes)[0]} values")
                
                raise
        
        def build_letter_model(self, num_classes):
            """Fast, accurate model for letters"""
            model = keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(63,)),
                layers.Dropout(0.4),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        def build_word_model(self, num_classes, timesteps, feature_dim):
            """Fast, accurate model for words"""
            print(f"\n✅ BUILDING LSTM MODEL:")
            print(f"   Input: ({timesteps} timesteps, {feature_dim} features)")
            print(f"   Output: {num_classes} word classes")
            
            model = keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=(timesteps, feature_dim)),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dense(64, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        def train_letters(self):
            """Train letter model"""
            print("\n" + "="*60)
            print("TRAINING LETTERS")
            print("="*60)
            
            X, y, labels = self.load_letters()
            
            if X is None:
                print("\n❌ No letter data found!")
                print("   Run: python continuous_collector_fixed.py")
                return
            
            print(f"\n✅ Loaded: {len(X)} samples, {len(labels)} classes")
            print(f"   Classes: {labels}")
            
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            print(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test")
            
            model = self.build_letter_model(len(labels))
            
            print("\n🎓 TRAINING...")
            model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ TEST ACCURACY: {acc*100:.2f}%")
            
            model.save(f"{self.models_dir}/letters_model.h5")
            
            import pickle
            with open(f"{self.models_dir}/letters_labels.pkl", 'wb') as f:
                pickle.dump(encoder, f)
            
            print(f"✅ Saved: {self.models_dir}/letters_model.h5")
            return acc
        
        def train_words(self):
            """Train word model"""
            print("\n" + "="*60)
            print("TRAINING WORDS")
            print("="*60)
            
            X, y, labels = self.load_words()
            
            if X is None:
                print("\n❌ No word data found!")
                print("   Run: python continuous_collector_fixed.py")
                return
            
            print(f"\n✅ DATA LOADED: {len(X)} sequences, {len(labels)} classes")
            print(f"   Classes: {labels}")
            print(f"   Shape: {X.shape}")
            
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            print(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test")
            
            model = self.build_word_model(len(labels), X.shape[1], X.shape[2])
            
            print("\n🎓 TRAINING...")
            model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=16,
                verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ TEST ACCURACY: {acc*100:.2f}%")
            
            model.save(f"{self.models_dir}/words_model.h5")
            
            import pickle
            with open(f"{self.models_dir}/words_labels.pkl", 'wb') as f:
                pickle.dump(encoder, f)
            
            print(f"✅ Saved: {self.models_dir}/words_model.h5")
            return acc


def main():
        trainer = UltraRobustTrainer()
        
        print("\n" + "="*60)
        print("ULTRA-ROBUST TRAINER")
        print("Handles ANY data inconsistencies automatically")
        print("="*60)
        print("\n1. Train LETTERS")
        print("2. Train WORDS")
        print("3. Train BOTH")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            trainer.train_letters()
        elif choice == '2':
            trainer.train_words()
        elif choice == '3':
            trainer.train_letters()
            trainer.train_words()
        else:
            print("\n❌ Invalid choice")
            return
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Models saved in: {trainer.models_dir}/")
        print("\nNext step: python enhanced_asl_interpreter.py")


if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            print("\n📋 Full traceback:")
            traceback.print_exc()