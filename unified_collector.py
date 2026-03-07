"""
UNIFIED ADVANCED COLLECTOR
============================
Letters: Auto-capture + image saving + 8x augmentation (JSON + JPG)
Words:   Sequence recording + 6x augmentation
Both:    Quality filtering, normalization, stability detection
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque
from scipy.ndimage import gaussian_filter1d

try:
    import mediapipe as mp
except ImportError:
    print("Install: pip install mediapipe scipy")
    exit()


class UnifiedAdvancedCollector:
    def __init__(self):
        print("\n" + "="*80)
        print("UNIFIED ADVANCED COLLECTOR")
        print("Letters: JSON + Images  |  Words: Sequences  |  8x Augmentation")
        print("="*80)

        self.setup_mediapipe()
        self.data_dir = "sign_data"
        os.makedirs(f"{self.data_dir}/letters", exist_ok=True)
        os.makedirs(f"{self.data_dir}/words",   exist_ok=True)
        print("\nCollector ready!")

    # ─────────────────────────────────────────────────────────────
    # MEDIAPIPE
    # ─────────────────────────────────────────────────────────────

    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def draw_hands(self, frame, results):
        if results and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

    # ─────────────────────────────────────────────────────────────
    # SHARED PROCESSING
    # ─────────────────────────────────────────────────────────────

    def normalize_landmarks(self, landmarks):
        """Position/scale invariant normalization."""
        arr = np.array(landmarks, dtype=np.float32).reshape(-1, 3)
        if arr.sum() == 0:
            return arr.flatten().tolist()
        center   = arr.mean(axis=0)
        centered = arr - center
        scale    = np.std(centered)
        if scale > 0:
            centered /= scale
        return centered.flatten().tolist()

    def smooth_sequence(self, sequence, sigma=1.5):
        arr = np.array(sequence, dtype=np.float32)
        return gaussian_filter1d(arr, sigma=sigma, axis=0).tolist()

    # ─────────────────────────────────────────────────────────────
    # IMAGE AUGMENTATION  (new — for letter images)
    # ─────────────────────────────────────────────────────────────

    def augment_image(self, img):
        """
        Generate 7 image variants from one raw capture:
          0 - original (cropped hand region)
          1 - flip horizontal
          2 - brightness +40
          3 - brightness -40
          4 - contrast ×1.3
          5 - contrast ×0.75
          6 - slight rotation +10°
          7 - slight rotation -10°
        Returns list of (suffix, image) tuples.
        """
        variants = [('orig', img.copy())]

        # Horizontal flip
        variants.append(('flip', cv2.flip(img, 1)))

        # Brightness
        bright = np.clip(img.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        dark   = np.clip(img.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        variants.append(('bright', bright))
        variants.append(('dark',   dark))

        # Contrast
        hi_con = np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        lo_con = np.clip(img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
        variants.append(('con_hi', hi_con))
        variants.append(('con_lo', lo_con))

        # Rotation ±10°
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        for angle, tag in [(10, 'rot10'), (-10, 'rotm10')]:
            M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            variants.append((tag, rot))

        return variants  # 8 total

    def crop_hand_region(self, frame, results, pad=40):
        """
        Crop a tight bounding box around the detected hand(s).
        Falls back to full frame if crop fails.
        Returns a 224x224 image ready to save.
        """
        h, w = frame.shape[:2]
        all_x, all_y = [], []

        if results and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))

        if all_x and all_y:
            x1 = max(0,     min(all_x) - pad)
            y1 = max(0,     min(all_y) - pad)
            x2 = min(w - 1, max(all_x) + pad)
            y2 = min(h - 1, max(all_y) + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (224, 224))

        return cv2.resize(frame, (224, 224))

    # ─────────────────────────────────────────────────────────────
    # LANDMARK AUGMENTATION  (for JSON data)
    # ─────────────────────────────────────────────────────────────

    def augment_landmarks(self, landmarks):
        """
        5 landmark variants: original + rotate ±8° + scale ×0.92/1.08
        """
        arr = np.array(landmarks, dtype=np.float32).reshape(-1, 3)
        variants = [arr.copy()]

        for angle_deg in [-8, 8]:
            rad   = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot = arr.copy()
            x_new = rot[:, 0] * cos_a - rot[:, 1] * sin_a
            y_new = rot[:, 0] * sin_a + rot[:, 1] * cos_a
            rot[:, 0] = x_new
            rot[:, 1] = y_new
            variants.append(rot)

        for scale in [0.92, 1.08]:
            variants.append(arr * scale)

        return [v.flatten().tolist() for v in variants]

    def check_landmark_quality(self, landmarks):
        arr = np.array(landmarks)
        if arr.sum() == 0:
            return False, "No hand detected"
        if np.std(arr) < 0.01:
            return False, "Hand too flat / static"
        return True, "OK"

    # ─────────────────────────────────────────────────────────────
    # SEQUENCE AUGMENTATION  (for words)
    # ─────────────────────────────────────────────────────────────

    def augment_sequence(self, sequence):
        seq = np.array(sequence, dtype=np.float32)
        variants = [seq.copy()]

        for angle_deg in [-5, 5]:
            rad   = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot  = seq.copy()
            rot3 = rot.reshape(len(rot), -1, 3)
            x_new = rot3[:, :, 0] * cos_a - rot3[:, :, 1] * sin_a
            y_new = rot3[:, :, 0] * sin_a + rot3[:, :, 1] * cos_a
            rot3[:, :, 0] = x_new
            rot3[:, :, 1] = y_new
            variants.append(rot3.reshape(len(rot), -1))

        n = len(seq)
        for factor in [1.2, 0.8]:
            idx = np.linspace(0, n - 1, int(n * factor))
            variants.append(np.array([seq[int(i)] for i in idx]))

        variants.append(seq + np.random.normal(0, 0.008, seq.shape))
        return [v.tolist() for v in variants]

    def check_sequence_quality(self, sequence):
        arr = np.array(sequence, dtype=np.float32)
        zero_frames = np.sum(arr.sum(axis=1) == 0)
        if zero_frames > len(sequence) * 0.3:
            return False, "Too many missing frames (>30%)"
        if np.std(arr) < 0.01:
            return False, "Too static"
        if len(sequence) < 20:
            return False, "Recording too short"
        return True, "OK"

    # ─────────────────────────────────────────────────────────────
    # LETTER COLLECTION  (with image saving)
    # ─────────────────────────────────────────────────────────────

    def collect_letter(self, letter, target_raw=20):
        """
        Per capture saves:
          - 5 JSON files  (landmark variants for model training)
          - 8 JPG images  (hand crop variants for CNN / visual augmentation)
        Total per raw capture: 13 files → target_raw=20 → 260 files
        """
        letter_upper = letter.upper()
        save_dir     = f"{self.data_dir}/letters/{letter_upper}"
        img_dir      = f"{save_dir}/images"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(img_dir,  exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera error!")
            return 0

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        collected     = 0
        json_saved    = 0
        img_saved     = 0
        last_capture  = 0
        INTERVAL      = 1.2          # seconds between auto-captures
        stability_buf = deque(maxlen=15)

        print(f"\n{'='*80}")
        print(f"COLLECTING LETTER: {letter_upper}")
        print(f"{'='*80}")
        print(f"Each capture saves:  5 JSON landmark files  +  8 cropped hand images")
        print(f"Target: {target_raw} raw  →  ~{target_raw*5} JSON  +  ~{target_raw*8} images")
        print("Hold the sign steady — auto-captures when stable")
        print("Q = quit early\n")

        while collected < target_raw:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            now     = time.time()
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            self.draw_hands(frame, results)

            has_hand = bool(results and results.multi_hand_landmarks)
            stability_buf.append(has_hand)
            stable = len(stability_buf) == stability_buf.maxlen and all(stability_buf)

            h, w, _ = frame.shape
            progress = int((collected / target_raw) * (w - 30))

            # ── Top bar ──
            cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 30), -1)
            cv2.putText(frame, f"LETTER: {letter_upper}", (15, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 100), 2)
            cv2.putText(frame,
                        f"Raw: {collected}/{target_raw}   JSON: {json_saved}   Images: {img_saved}",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # ── Stability pill ──
            pill_color = (0, 180, 60) if stable else (0, 60, 200)
            pill_text  = "STABLE — capturing soon" if stable else "Hold still..."
            cv2.rectangle(frame, (10, h - 70), (360, h - 42), pill_color, -1)
            cv2.rectangle(frame, (10, h - 70), (360, h - 42), (255, 255, 255), 1)
            cv2.putText(frame, pill_text, (18, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ── Progress bar ──
            cv2.rectangle(frame, (15, h - 30), (w - 15, h - 12), (40, 40, 40), -1)
            cv2.rectangle(frame, (15, h - 30), (15 + progress, h - 12), (0, 200, 80), -1)
            cv2.rectangle(frame, (15, h - 30), (w - 15, h - 12), (90, 90, 90), 1)
            cv2.putText(frame, f"{int(collected/target_raw*100)}%",
                        (w // 2 - 15, h - 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # ── Flash on capture ──
            time_since = now - last_capture
            if 0 < time_since < 0.15:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 100), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

            cv2.imshow('Letter Collector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # ── AUTO-CAPTURE ──
            if stable and has_hand and (now - last_capture) >= INTERVAL:
                first_hand = results.multi_hand_landmarks[0]
                raw_lm = []
                for lm in first_hand.landmark:
                    raw_lm.extend([lm.x, lm.y, lm.z])

                ok, reason = self.check_landmark_quality(raw_lm)
                if not ok:
                    print(f"  Rejected: {reason}")
                    last_capture = now
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # ── Save JSON landmark variants ──
                normalized    = self.normalize_landmarks(raw_lm)
                lm_variants   = self.augment_landmarks(normalized)
                aug_lm_types  = ['orig', 'rot_-8', 'rot_+8', 'scale_0.92', 'scale_1.08']

                for aug_idx, variant in enumerate(lm_variants):
                    data = {
                        "label":      letter_upper,
                        "landmarks":  variant,
                        "timestamp":  ts,
                        "aug_type":   aug_lm_types[aug_idx],
                        "has_image":  True,   # paired image also saved
                    }
                    fname = f"{letter_upper}_{ts}_lm{aug_idx}.json"
                    with open(f"{save_dir}/{fname}", 'w') as f:
                        json.dump(data, f)
                    json_saved += 1

                # ── Save image variants ──
                # Crop tight around hand region (cleaner than full frame)
                hand_crop   = self.crop_hand_region(frame.copy(), results, pad=50)
                img_variants = self.augment_image(hand_crop)

                for suffix, img_variant in img_variants:
                    img_fname = f"{letter_upper}_{ts}_{suffix}.jpg"
                    cv2.imwrite(
                        f"{img_dir}/{img_fname}",
                        img_variant,
                        [cv2.IMWRITE_JPEG_QUALITY, 92]
                    )
                    img_saved += 1

                collected   += 1
                last_capture = now
                print(f"  [{collected}/{target_raw}]  "
                      f"+{len(lm_variants)} JSON  +{len(img_variants)} images  "
                      f"| Total: {json_saved} JSON, {img_saved} images")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"DONE: {letter_upper}")
        print(f"  Raw captured : {collected}")
        print(f"  JSON files   : {json_saved}  (for landmark model)")
        print(f"  Image files  : {img_saved}  (224x224 hand crops, augmented)")
        print(f"  Saved to     : {save_dir}/")
        print(f"  Images in    : {img_dir}/")
        print(f"{'='*60}")
        return collected

    # ─────────────────────────────────────────────────────────────
    # WORD COLLECTION
    # ─────────────────────────────────────────────────────────────

    def collect_word(self, word, target_raw=10):
        word_display = word.strip()
        word_file    = word_display.lower().replace(' ', '_')
        save_dir     = f"{self.data_dir}/words/{word_file}"
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera error!")
            return 0

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        collected   = 0
        total_saved = 0
        FRAMES      = 60

        print(f"\n{'='*80}")
        print(f"COLLECTING WORD: {word_display}")
        print(f"{'='*80}")
        print(f"Target: {target_raw} recordings → ~{target_raw*6} training samples")
        print("SPACE = start recording  |  Q = quit early\n")

        while collected < target_raw:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            self.draw_hands(frame, results)

            progress = int((collected / target_raw) * (w - 30))

            cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 30), -1)
            cv2.putText(frame, f"WORD: {word_display}", (15, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 2)
            cv2.putText(frame, f"Recorded: {collected}/{target_raw}   Saved: {total_saved}",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.rectangle(frame, (10, h - 70), (380, h - 42), (0, 100, 160), -1)
            cv2.rectangle(frame, (10, h - 70), (380, h - 42), (255, 255, 255), 1)
            cv2.putText(frame, "SPACE = record 2 seconds", (18, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.rectangle(frame, (15, h - 30), (w - 15, h - 12), (40, 40, 40), -1)
            cv2.rectangle(frame, (15, h - 30), (15 + progress, h - 12), (0, 160, 200), -1)
            cv2.rectangle(frame, (15, h - 30), (w - 15, h - 12), (90, 90, 90), 1)

            cv2.imshow('Word Collector', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord(' '):
                print(f"  Recording {collected + 1}/{target_raw}...")
                sequence = []

                for i in range(FRAMES):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame   = cv2.flip(frame, 1)
                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb)

                    lm = []
                    if results and results.multi_hand_landmarks:
                        for hand_idx in range(2):
                            if hand_idx < len(results.multi_hand_landmarks):
                                hand = results.multi_hand_landmarks[hand_idx]
                                for p in hand.landmark:
                                    lm.extend([p.x, p.y, p.z])
                            else:
                                lm.extend([0.0] * 63)
                    else:
                        lm = [0.0] * 126

                    sequence.append(lm)
                    self.draw_hands(frame, results)

                    h2, w2 = frame.shape[:2]
                    pct = (i + 1) / FRAMES
                    cv2.rectangle(frame, (0, 0), (w2, 60), (100, 0, 0), -1)
                    cv2.putText(frame, f"RECORDING  {i+1}/{FRAMES}", (15, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    cv2.rectangle(frame, (15, h2 - 20), (w2 - 15, h2 - 8), (60, 0, 0), -1)
                    cv2.rectangle(frame, (15, h2 - 20),
                                  (15 + int((w2 - 30) * pct), h2 - 8), (0, 80, 255), -1)
                    cv2.imshow('Word Collector', frame)
                    cv2.waitKey(33)

                ok, reason = self.check_sequence_quality(sequence)
                if not ok:
                    print(f"  Rejected: {reason} — try again")
                    continue

                smoothed   = self.smooth_sequence(sequence)
                normalized = [self.normalize_landmarks(f) for f in smoothed]
                variants   = self.augment_sequence(normalized)
                aug_types  = ['orig', 'rot_-5', 'rot_+5', 'slow_1.2x', 'fast_0.8x', 'noise']

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                for aug_idx, variant in enumerate(variants):
                    data = {
                        "label":     word_display,
                        "sequence":  variant,
                        "timestamp": ts,
                        "aug_type":  aug_types[aug_idx],
                    }
                    fname = f"{word_file}_{ts}_aug{aug_idx}.json"
                    with open(f"{save_dir}/{fname}", 'w') as f:
                        json.dump(data, f)
                    total_saved += 1

                collected += 1
                print(f"  [{collected}/{target_raw}]  +{len(variants)} variants saved  (total: {total_saved})")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nDone: {collected} raw  →  {total_saved} training samples")
        print(f"Saved to: {save_dir}/")
        return collected

    # ─────────────────────────────────────────────────────────────
    # MENUS
    # ─────────────────────────────────────────────────────────────

    def run_letter_menu(self):
        print("\n" + "="*80)
        print("LETTER COLLECTION")
        print("="*80)
        print("\n1. Single letter")
        print("2. Batch (multiple letters at once)")
        print("3. Back")

        choice = input("\nChoice: ").strip()

        if choice == '1':
            while True:
                letter = input("\nLetter to collect (or 'done'): ").strip()
                if letter.lower() in ('done', ''):
                    break
                if len(letter) != 1 or not letter.isalpha():
                    print("Single letter only!")
                    continue
                raw = input(f"Raw captures for '{letter.upper()}'? "
                            f"(default 20 → 100 JSON + 160 images): ").strip()
                raw = int(raw) if raw.isdigit() else 20
                print("\nStarting in 3 seconds — show your hand sign...")
                time.sleep(3)
                self.collect_letter(letter, raw)

        elif choice == '2':
            letters = input("\nLetters to collect (e.g. ABCDE): ").strip().upper()
            if not all(c.isalpha() for c in letters):
                print("Letters only!")
                return
            raw = input("Raw captures per letter? (default 20): ").strip()
            raw = int(raw) if raw.isdigit() else 20
            for i, letter in enumerate(letters):
                print(f"\n[{i+1}/{len(letters)}] Starting '{letter}' in 3 seconds...")
                time.sleep(3)
                self.collect_letter(letter, raw)
            print(f"\nBatch complete! Collected {len(letters)} letters.")

    def run_word_menu(self):
        print("\n" + "="*80)
        print("WORD COLLECTION")
        print("="*80)

        while True:
            word = input("\nWord to collect (or 'done'): ").strip()
            if word.lower() in ('done', ''):
                break
            if not all(c.isalpha() or c.isspace() for c in word):
                print("Letters and spaces only!")
                continue
            raw = input(f"Raw recordings for '{word}'? "
                        f"(default 10 → ~60 training samples): ").strip()
            raw = int(raw) if raw.isdigit() else 10
            print("\nStarting in 3 seconds...")
            time.sleep(3)
            self.collect_word(word, raw)

    def run(self):
        while True:
            print("\n" + "="*80)
            print("ASL ADVANCED DATA COLLECTOR")
            print("="*80)
            print("\n1. LETTERS  — auto-capture | 5x JSON aug | 8x image aug | 224x224 crops")
            print("2. WORDS    — SPACE record | 6x sequence aug | smooth + normalize")
            print("3. Exit")

            choice = input("\nChoice: ").strip()

            if choice == '1':
                self.run_letter_menu()
            elif choice == '2':
                self.run_word_menu()
            elif choice == '3':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice")

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("  Letters: python ultra_robust_trainer.py  -> option 1")
        print("  Words:   python advanced_word_trainer.py")
        print("="*80 + "\n")


def main():
    try:
        collector = UnifiedAdvancedCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()