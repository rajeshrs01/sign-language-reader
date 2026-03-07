"""
ADVANCED WORD COLLECTOR
========================
✅ Data augmentation (multiply samples 5x)
✅ Temporal smoothing (reduce noise)
✅ Normalization (consistent features)
✅ Quality filtering (reject bad samples)
Result: High accuracy with only 10-20 sequences!
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

try:
    import mediapipe as mp
except ImportError:
    print("❌ Install: pip install mediapipe scipy")
    exit()


class AdvancedWordCollector:
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 ADVANCED WORD COLLECTOR")
        print("   Data Augmentation • Quality Filtering • Temporal Smoothing")
        print("   High accuracy with only 10-20 sequences!")
        print("="*80)
        
        self.setup_mediapipe()
        self.data_dir = "sign_data/advanced_words"
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("\n✅ Advanced collector ready!")
    
    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,  # Higher for quality
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def normalize_sequence(self, sequence):
        """Normalize to be position/scale invariant"""
        sequence = np.array(sequence)
        
        # Reshape to (frames, landmarks, coords)
        frames, features = sequence.shape
        sequence = sequence.reshape(frames, -1, 3)  # (frames, 42 landmarks, 3)
        
        # Normalize each frame
        normalized = []
        for frame in sequence:
            if frame.sum() == 0:  # Skip empty frames
                normalized.append(frame.flatten())
                continue
            
            # Center at origin (subtract mean position)
            center = frame.mean(axis=0)
            centered = frame - center
            
            # Scale to unit variance
            scale = np.std(centered)
            if scale > 0:
                scaled = centered / scale
            else:
                scaled = centered
            
            normalized.append(scaled.flatten())
        
        return np.array(normalized)
    
    def smooth_sequence(self, sequence, sigma=1.5):
        """Apply temporal smoothing to reduce noise"""
        sequence = np.array(sequence)
        smoothed = gaussian_filter1d(sequence, sigma=sigma, axis=0)
        return smoothed
    
    def augment_sequence(self, sequence, augmentations=4):
        """
        Augment data to create multiple variations
        Techniques: rotation, scaling, speed variation, noise
        """
        augmented = [sequence]  # Original
        sequence = np.array(sequence)
        
        # 1. Rotation augmentation (slight angle changes)
        for angle in [-5, 5]:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rotated = sequence.copy()
            
            for i in range(len(rotated)):
                frame = rotated[i].reshape(-1, 3)
                # Rotate around z-axis (2D rotation in x-y plane)
                for j in range(len(frame)):
                    x, y, z = frame[j]
                    frame[j, 0] = x * cos_a - y * sin_a
                    frame[j, 1] = x * sin_a + y * cos_a
                rotated[i] = frame.flatten()
            
            augmented.append(rotated)
        
        # 2. Speed variation (temporal stretching)
        # Slower version
        indices_slow = np.linspace(0, len(sequence)-1, int(len(sequence)*1.2))
        slow = np.array([sequence[int(i)] for i in indices_slow])
        augmented.append(slow)
        
        # Faster version
        indices_fast = np.linspace(0, len(sequence)-1, int(len(sequence)*0.8))
        fast = np.array([sequence[int(i)] for i in indices_fast])
        augmented.append(fast)
        
        # 3. Small noise addition (simulates hand tremor)
        noise = sequence + np.random.normal(0, 0.01, sequence.shape)
        augmented.append(noise)
        
        return augmented[:augmentations+1]  # Original + augmentations
    
    def check_quality(self, sequence):
        """Check if sequence has good quality"""
        sequence = np.array(sequence)
        
        # Check 1: Not too many missing frames
        zero_frames = np.sum(sequence.sum(axis=1) == 0)
        if zero_frames > len(sequence) * 0.3:  # More than 30% missing
            return False, "Too many missing frames"
        
        # Check 2: Sufficient movement
        movement = np.std(sequence)
        if movement < 0.01:  # Too static
            return False, "Insufficient movement"
        
        # Check 3: Reasonable duration
        if len(sequence) < 20:  # Less than ~1 second
            return False, "Too short"
        
        return True, "Good quality"
    
    def collect_word(self, word, target_samples=10):
        """
        Collect word with advanced processing
        Only need 10-20 raw sequences → augmented to 50-100!
        """
        word_display = word
        word_file = word.lower().replace(' ', '_')
        
        save_dir = f"{self.data_dir}/{word_file}"
        os.makedirs(save_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera error!")
            return 0
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        collected = 0
        total_saved = 0
        
        print(f"\n{'='*80}")
        print(f"🎬 COLLECTING: {word_display}")
        print(f"{'='*80}")
        print(f"Target: {target_samples} sequences")
        print(f"With augmentation: ~{target_samples * 5} final samples")
        print(f"Duration: 2 seconds per sequence")
        print("Press SPACE to record | Q to finish early\n")
        
        while collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            # Draw
            if results and results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # UI
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 80), (40, 40, 40), -1)
            cv2.putText(frame, f"Word: {word_display}", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Collected: {collected}/{target_samples}  |  Saved: {total_saved}", (15, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "SPACE=Record | Q=Finish", (15, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Advanced Word Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print(f"\n  🔴 Recording {collected + 1}/{target_samples}...")
                
                # Record sequence
                sequence = []
                for i in range(60):  # 2 seconds at 30fps
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb)
                    
                    # Extract both hands (126 features)
                    landmarks = []
                    if results and results.multi_hand_landmarks:
                        for hand_idx in range(2):
                            if hand_idx < len(results.multi_hand_landmarks):
                                hand = results.multi_hand_landmarks[hand_idx]
                                for lm in hand.landmark:
                                    landmarks.extend([lm.x, lm.y, lm.z])
                            else:
                                landmarks.extend([0] * 63)
                    else:
                        landmarks = [0] * 126
                    
                    sequence.append(landmarks)
                    
                    # Draw feedback
                    if results and results.multi_hand_landmarks:
                        for hand in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                    
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
                    cv2.putText(frame, f"🔴 RECORDING  {i+1}/60", (15, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    
                    cv2.imshow('Advanced Word Collector', frame)
                    cv2.waitKey(33)
                
                # Quality check
                is_good, msg = self.check_quality(sequence)
                if not is_good:
                    print(f"  ⚠️  Rejected: {msg} - Try again")
                    continue
                
                # Process: smooth → normalize
                sequence = self.smooth_sequence(sequence)
                sequence = self.normalize_sequence(sequence)
                
                # Augment (5x multiplication)
                augmented_sequences = self.augment_sequence(sequence, augmentations=4)
                
                # Save all augmented versions
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for aug_idx, aug_seq in enumerate(augmented_sequences):
                    data = {
                        "label": word_display,
                        "label_file": word_file,
                        "sequence": aug_seq.tolist(),
                        "timestamp": timestamp,
                        "augmentation": aug_idx,
                        "augmentation_type": [
                            "original", "rotate_left", "rotate_right", 
                            "slower", "faster", "noise"
                        ][aug_idx] if aug_idx < 6 else "noise"
                    }
                    
                    filename = f"{word_file}_{timestamp}_aug{aug_idx}.json"
                    with open(f"{save_dir}/{filename}", 'w') as f:
                        json.dump(data, f)
                    
                    total_saved += 1
                
                collected += 1
                print(f"  ✅ {collected}/{target_samples}  →  {len(augmented_sequences)} augmented versions saved")
                print(f"     Total samples: {total_saved}\n")
            
            elif key == ord('q'):
                print("\n⚠️  Finished early")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*80}")
        print(f"✅ COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"Raw sequences collected: {collected}")
        print(f"Total samples (with augmentation): {total_saved}")
        print(f"Saved to: {save_dir}/")
        print(f"Multiplication factor: {total_saved/max(1, collected):.1f}x\n")
        
        return collected
    
    def run(self):
        """Run collection"""
        print("\n" + "="*80)
        print("ADVANCED WORD COLLECTION")
        print("="*80)
        print("\n💡 Tips for best results:")
        print("   • Good lighting on hands")
        print("   • Both hands visible")
        print("   • Sign at normal speed (~2 seconds)")
        print("   • Show appropriate facial expressions")
        print("\n🎯 With data augmentation:")
        print("   10 sequences → ~50 training samples")
        print("   20 sequences → ~100 training samples")
        print("   Result: High accuracy with minimal effort!\n")
        
        while True:
            word = input("\nEnter word to collect (or 'done'): ").strip()
            
            if word.lower() == 'done' or word == '':
                break
            
            if not all(c.isalpha() or c.isspace() for c in word):
                print("❌ Letters and spaces only")
                continue
            
            samples = input(f"How many sequences for '{word}'? (10-20 recommended): ").strip()
            samples = int(samples) if samples else 15
            
            print(f"\n🎬 Starting in 3 seconds...")
            import time
            time.sleep(3)
            
            self.collect_word(word, samples)
        
        print("\n" + "="*80)
        print("📝 NEXT STEP:")
        print("   python advanced_word_trainer.py")
        print("="*80 + "\n")


def main():
    try:
        collector = AdvancedWordCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
