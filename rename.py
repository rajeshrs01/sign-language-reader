"""
ASL SYSTEM  v5.3  — FIXED
======================================================

FIXES vs v5.2:
  FIX A — Letters not showing / slow recognition:
    VOTE_WINDOW reduced 30→18 (≈1.2s at every frame).
    VOTE_MAJORITY lowered 0.55→0.45.
    Model now runs EVERY frame (removed %2 skip).
    Together: a clear held sign registers in ~1.5s reliably.

  FIX B — Blanks swamping the vote buffer:
    Blank slots ('_') no longer count against the majority.
    Winner is now evaluated against non-blank count when
    non-blanks dominate, falling back to full-buffer check.
    This means a clear sign wins even if a few frames are
    dropped/uncertain.

  FIX C — Letter panel shows nothing after adding:
    _locked_label is now preserved as display fallback so
    the letter panel continues showing the added letter
    during the word-commit countdown (orange state).

  FIX D — CONF_MIN lowered 0.45→0.40 for better recall.
    CONF_CONFUSABLE kept at 0.65 (safety for A/S/E etc.)

LETTER MODE  (how it works):
  1. Sign a letter and hold it
  2. After 1.5s stable  →  letter added automatically
  3. Change sign  →  resets timer, ready for next letter
  4. After adding, hold 3 more seconds on same sign  →  word committed
     OR change signs repeatedly to spell then remove hand 1.5s to accept

WORD MODE:
  Sign a word → predictions shown
  Remove hand 1.5s → accept top prediction

GESTURE TIMER (hand absent):
  1.5s → Accept
  4.0s → Toggle Letters ↔ Words
  7.0s → Speak English

KEYBOARD:
  A          =  Manually add current letter (letter mode)
  SPACE      =  Manually commit word (letter mode)
  S          =  Speak English
  M          =  Toggle mode
  BACKSPACE  =  Delete last letter / last word
  Z          =  Undo
  ENTER      =  Clear all
  1-4        =  Accept autocomplete suggestion
  D          =  Debug
  Q          =  Quit
"""

import cv2
import numpy as np
import pickle
import time
import re
import threading
from collections import deque, Counter
import os
from scipy.ndimage import gaussian_filter1d

try:
    import mediapipe as mp
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:
    print(f"[FATAL] Missing: {e}")
    print("pip install mediapipe tensorflow opencv-python scipy")
    exit()

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

import urllib.request
import json as _json

# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LETTER_ADD_TIME   = 1.5
WORD_COMMIT_TIME  = 3.0

# FIX A+B: lower threshold, smaller window, no frame-skip
CONF_MIN          = 0.40          # was 0.45
CONF_CONFUSABLE   = 0.65

# FIX A: smaller window = faster response
VOTE_WINDOW       = 18            # was 30  (≈1.2s at 15fps, or 0.6s at every frame)

# FIX B: winner evaluated against non-blank count (see LetterRecogniser)
VOTE_MAJORITY     = 0.45          # was 0.55

UNLOCK_FRAMES     = 8             # was 10 — slightly faster unlock
MODE_GRACE        = 2.5
ABSENT_DEBOUNCE   = 3

T_ACCEPT  = 1.5
T_TOGGLE  = 4.0
T_SPEAK   = 7.0

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"

SIMILAR_GROUPS = [
    {'A','S','E'}, {'R','U'}, {'M','N'},
    {'B','D'}, {'G','H'}, {'P','K'},
]

def is_confusable(lbl):
    return bool(lbl and any(lbl.upper() in g for g in SIMILAR_GROUPS))

def conf_thr(lbl):
    return CONF_CONFUSABLE if is_confusable(lbl) else CONF_MIN


# ══════════════════════════════════════════════════════════════════════════════
#  GRAMMAR  (local rules + optional Ollama AI)
# ══════════════════════════════════════════════════════════════════════════════
_GRAMMAR_PROMPT = """You are an ASL-to-English converter in a real-time sign language app.
Input: raw ASL gloss. Output: ONE fluent English sentence. Nothing else. No quotes.
Rules: add copulas, fix word order, expand compound signs, add articles, detect questions.
Examples:
HELLO -> Hello! How are you?
I HOME -> I am at home.
I HUNGRY -> I am hungry.
WATER -> Could I have some water, please?
STOP -> Please stop!
I_LOVE_YOU -> I love you!
HOW_ARE_YOU -> How are you doing?
HELLO I HOME HUNGRY -> Hello! I am at home and I am hungry.
ASL: """


class LocalGrammar:
    PRONOUN  = {'me':'I','mine':'mine','him':'him','her':'her'}
    PAST_M   = {'yesterday','before','ago','finish','finished','already','past','last'}
    FUTURE_M = {'tomorrow','later','soon','will','future','next'}
    VERB_PAST= {'go':'went','eat':'ate','drink':'drank','see':'saw','give':'gave',
                'come':'came','do':'did','have':'had','buy':'bought','run':'ran',
                'write':'wrote','read':'read','know':'knew','make':'made',
                'take':'took','leave':'left','sleep':'slept','find':'found',
                'feel':'felt','hear':'heard','get':'got','sit':'sat','stand':'stood'}
    LOCS     = {'home','school','work','hospital','store','market','church',
                'park','office','gym','library','restaurant','here','there',
                'outside','inside','upstairs','downstairs'}
    NO_ART   = {'home','here','there','outside','inside','upstairs','downstairs'}
    ADJS     = {'hungry','tired','happy','sad','sick','fine','good','bad','busy',
                'ready','sorry','angry','excited','bored','hot','cold','thirsty',
                'scared','confused','lost','late','early','wrong','right','sure',
                'okay','ok','well','ill','hurt','deaf','blind','sleepy','full'}
    VERBS    = {'go','eat','drink','see','give','come','do','have','buy','bring',
                'think','tell','meet','sit','run','write','read','know','say',
                'make','take','leave','sleep','want','need','like','love','hate',
                'help','work','play','walk','talk','speak','ask','answer','call',
                'visit','learn','teach','use','find','show','watch','look',
                'listen','feel','understand','remember','forget','try','finish',
                'start','stop','wait','open','close','drive','ride','fly',
                'swim','cook','clean','get','hear','lose','pay','send','win'}
    NOUNS    = {'book','car','dog','cat','ball','house','apple','banana','cup',
                'bag','pen','phone','computer','table','chair','door','window',
                'room','bed','tree','flower','bird','fish','box','bottle','key',
                'shirt','shoe','hat','coat','child','baby','boy','girl','man',
                'woman','person','friend','teacher','doctor','job','problem',
                'question','answer','idea','plan','story','movie','game','test',
                'letter','email','food','money','time','day','week','month','year'}
    MASS     = {'water','milk','juice','rice','bread','coffee','tea','soup',
                'sugar','salt','flour','butter','cheese','meat','help','advice',
                'information','weather','homework','luggage','furniture','news'}
    SUBJS    = {'i','you','he','she','we','they','it'}
    VOWELS   = set('aeiou')

    SOLO = {
        'hello':'Hello! How are you?', 'home':'I am at home.',
        'how_are_you':'How are you doing?', 'hungry':'I am feeling hungry.',
        'i_love_you':'I love you!', 'my_name_is':'My name is...',
        'no':'No, thank you.', 'stop':'Please stop!',
        'thank_you':'Thank you very much!', 'water':'Could I have some water, please?',
        'yes':'Yes, please!', 'tired':'I am tired.', 'thirsty':'I am thirsty.',
        'happy':'I am happy.', 'sad':'I am sad.', 'sick':'I am not feeling well.',
        'fine':'I am fine.', 'okay':'I am okay.', 'ok':'I am okay.',
    }
    COMPOUNDS = {
        'my_name_is':'My name is...', 'how_are_you':'How are you?',
        'i_love_you':'I love you!',   'hello':'Hello!', 'thank_you':'Thank you!',
    }
    PHRASES = [
        (['hello','i','home','hungry'], 'Hello! I am at home and I am hungry.'),
        (['how_are_you','yes'],         'How are you? Yes, I am doing well!'),
        (['hello','i','home'],          'Hello! I am at home.'),
        (['i','home','hungry'],         'I am at home and I am hungry.'),
        (['i','hungry','water'],        'I am hungry and I would like some water.'),
        (['water','hungry'],            'I am hungry and I would like some water.'),
        (['i_love_you','thank_you'],    'I love you! Thank you!'),
        (['no','stop'],                 'No, please stop!'),
        (['no','thank_you'],            'No, thank you!'),
        (['yes','thank_you'],           'Yes, thank you!'),
    ]

    def _art(self, w):
        if w.lower() in self.MASS: return 'some'
        return 'an' if w and w[0].lower() in self.VOWELS else 'a'

    def _past(self, v):
        v = v.lower()
        if v in self.VERB_PAST: return self.VERB_PAST[v]
        if v.endswith('e'): return v + 'd'
        if (len(v)>=3 and v[-1] not in 'aeiou'
                and v[-2] in 'aeiou' and v[-3] not in 'aeiou'):
            return v + v[-1] + 'ed'
        return v + 'ed'

    def _tense(self, ws):
        s = set(ws)
        if s & self.PAST_M:   return 'past'
        if s & self.FUTURE_M: return 'future'
        return 'present'

    def _be(self, s):
        if s == 'i':                  return 'am'
        if s in ('you','we','they'):  return 'are'
        return 'is'

    def convert(self, text):
        if not text or text.strip() in ('','Start signing...'): return ''
        words = text.lower().split()
        if not words: return ''
        if len(words) == 1:
            w = words[0]
            if w in self.SOLO:   return self.SOLO[w]
            if w in self.ADJS:   return f"I am {w}."
            if w in self.VERBS:  return f"I {w}."
            if w in self.LOCS:   return f"I am at {w}."
            return w.capitalize() + '.'
        for pat, result in self.PHRASES:
            n = len(pat)
            if words[:n] == pat:
                tail = words[n:]
                return result + (' ' + self.convert(' '.join(tail)) if tail else '')
        clauses, chunk = [], []
        for w in words:
            if w in self.COMPOUNDS:
                if chunk: clauses.append(self._chunk(chunk)); chunk = []
                clauses.append(self.COMPOUNDS[w])
            else:
                chunk.append(w)
        if chunk: clauses.append(self._chunk(chunk))
        out = ' '.join(c for c in clauses if c)
        return re.sub(r'\s+([.,!?])', r'\1', out)

    def _chunk(self, words):
        if not words: return ''
        words = [self.PRONOUN.get(w, w) for w in words]
        dd = [words[0]]
        for w in words[1:]:
            if w != dd[-1]: dd.append(w)
        words = dd
        if len(words) == 1:
            w = words[0]
            if w in self.SOLO:   return self.SOLO[w]
            if w in self.ADJS:   return f"I am {w}."
            if w in self.LOCS:   return f"I am at {w}."
            if w in self.VERBS:  return f"I {w}."
            if w == 'i':         return ''
            return w.capitalize() + '.'
        if (words[0] not in self.SUBJS
                and words[0] not in ('a','an','the','no','yes')
                and (words[0] in self.ADJS or words[0] in self.LOCS
                     or words[0] in self.VERBS)):
            words = ['i'] + words
        tense = self._tense(words)
        clean = [w for w in words if w not in (self.PAST_M | self.FUTURE_M)]
        words = clean or words
        res = []; after_cop = False; last_verb = False
        i = 0
        while i < len(words):
            w  = words[i]; nw = words[i+1] if i+1<len(words) else None
            pw = res[-1].lower() if res else None; is_s = w in self.SUBJS
            if is_s and nw and nw in self.ADJS:
                res += [w, self._be(w)]; after_cop = True; last_verb = False
                i += 1; continue
            if w in self.ADJS:
                if after_cop: res.append(w); after_cop = False
                elif pw and pw in self.ADJS: res += ['and', w]
                else: res.append(w)
                last_verb = False; i += 1; continue
            if (pw and pw in self.ADJS and (w in self.MASS or w in self.NOUNS)):
                res += ['and I would like', self._art(w), w]
                after_cop = False; last_verb = False; i += 1; continue
            after_cop = False
            if is_s and nw and nw in self.LOCS:
                res += [w, self._be(w), 'at']; last_verb = False
                i += 1; continue
            if w in self.VERBS:
                if tense == 'past': res.append(self._past(w))
                elif tense == 'future':
                    if not (res and res[-1]=='will'): res.append('will')
                    res.append(w)
                else:
                    if pw in ('want','need','like','love','hate','start','try','help'):
                        res.append('to')
                    res.append(w)
                last_verb = True; i += 1; continue
            if w in self.LOCS and last_verb:
                res.append('to')
                if w not in self.NO_ART: res.append('the')
                res.append(w); last_verb = False; i += 1; continue
            last_verb = False
            if w in self.NOUNS or w in self.MASS:
                if pw not in ('a','an','the','my','your','his','her','our',
                              'their','this','that','some'):
                    res.append(self._art(w))
                res.append(w); i += 1; continue
            res.append(w); i += 1
        if res: res[0] = res[0].capitalize()
        out = ' '.join(res)
        out = re.sub(r'\bi\b', 'I', out)
        out = re.sub(r'\s+', ' ', out).strip()
        if out and out[-1] not in '.!?': out += '.'
        return re.sub(r'\s+([.,!?])', r'\1', out)


class GrammarEngine:
    def __init__(self):
        self.local    = LocalGrammar()
        self._cache   = {}
        self._pending = set()
        self._lock    = threading.Lock()
        self.ai_ok    = False
        self._check_ai()

    def _check_ai(self):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request("http://localhost:11434/api/tags"),
                    timeout=2) as r:
                data   = _json.loads(r.read())
                models = [m['name'] for m in data.get('models',[])]
                if any(OLLAMA_MODEL.split(':')[0] in m for m in models):
                    self.ai_ok = True
                    print(f"  [AI] Ollama ready — {OLLAMA_MODEL}")
                else:
                    print(f"  [AI] Model missing. Run: ollama pull {OLLAMA_MODEL}")
        except Exception:
            print("  [AI] Ollama not running — local grammar only")

    def convert(self, text):
        if not text or text == 'Start signing...': return ''
        key = text.strip().lower()
        with self._lock:
            if key in self._cache: return self._cache[key]
        if self.ai_ok and key not in self._pending:
            with self._lock: self._pending.add(key)
            threading.Thread(target=self._fetch, args=(text,key),
                             daemon=True).start()
        return self.local.convert(text)

    def _fetch(self, text, key):
        try:
            payload = _json.dumps({
                "model":OLLAMA_MODEL,"stream":False,
                "prompt":_GRAMMAR_PROMPT + text.upper(),
                "options":{"temperature":0.1,"num_predict":60,
                           "stop":["\n","ASL:","Input:"]}
            }).encode()
            req = urllib.request.Request(
                OLLAMA_URL, data=payload,
                headers={"Content-Type":"application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=12) as r:
                resp = _json.loads(r.read()).get("response","").strip()
            for p in ["English:","Output:","->",":"]:
                if resp.startswith(p): resp = resp[len(p):].strip()
            resp = resp.strip('"\'')
            if resp and resp[-1] not in '.!?': resp += '.'
            if resp:
                with self._lock: self._cache[key] = resp
                print(f"  [AI] {text!r} -> {resp!r}")
        except Exception as e:
            print(f"  [AI ERR] {e}")
        finally:
            with self._lock: self._pending.discard(key)

    def is_ai(self, text):
        with self._lock: return text.strip().lower() in self._cache


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOCOMPLETE
# ══════════════════════════════════════════════════════════════════════════════
class Autocomplete:
    # ── 20 demo training words ────────────────────────────────────────────────
    VOCAB = [
        'hello','goodbye','thank_you','sorry',
        'help','i','you','want','need',
        'go','come','food','water','bathroom',
        'doctor','pain','happy','sick',
        'emergency','understand',
    ]

    # Words shown when sentence is empty
    STARTERS = ['hello','i','help','emergency','sorry','you']

    # Next-word suggestions based on what was just signed
    BIGRAMS = {
        'hello':      ['i','you','happy','thank_you','goodbye'],
        'goodbye':    ['thank_you','happy','i','you'],
        'thank_you':  ['you','hello','goodbye','i','help'],
        'sorry':      ['i','you','understand','help'],

        'i':          ['want','need','happy','sick','pain','go','come',
                       'understand','sorry','thank_you'],
        'you':        ['want','need','understand','go','come','happy','sick'],

        'want':       ['food','water','go','come','help','doctor'],
        'need':       ['help','doctor','water','food','bathroom','go','come'],

        'go':         ['doctor','bathroom','food','water','help'],
        'come':       ['help','doctor','i','you'],

        'food':       ['want','need','thank_you','water'],
        'water':      ['want','need','thank_you','food'],
        'bathroom':   ['need','help','i','sorry'],

        'doctor':     ['need','help','i','come','go'],
        'pain':       ['i','need','doctor','help','sorry'],

        'happy':      ['i','you','thank_you','hello','goodbye'],
        'sick':       ['i','you','need','help','doctor','pain'],

        'emergency':  ['help','come','doctor','need','i'],
        'understand': ['i','you','sorry','need','help'],

        'help':       ['need','i','you','emergency','come','doctor'],
    }

    def suggest(self, sentence, spelling, n=4):
        norm = [w.lower() for w in sentence]
        if spelling:
            prefix = ''.join(spelling).lower()
            return [w for w in self.VOCAB
                    if w.startswith(prefix) and w not in norm][:n]
        if not norm: return self.STARTERS[:n]
        last  = norm[-1]
        cands = [c for c in self.BIGRAMS.get(last, []) if c not in norm]
        cands += [s for s in self.STARTERS if s not in norm and s not in cands]
        return cands[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS
# ══════════════════════════════════════════════════════════════════════════════
class Alerts:
    EV = {
        'letter': (1200, 55,  (185,200, 30), '+ LETTER'),
        'word':   (1000,120,  ( 30,145,255), '⎵ WORD'),
        'accept': ( 880, 80,  ( 55,210, 90), '✓ ACCEPT'),
        'toggle': ( 660,120,  ( 20,210,210), '⇄ MODE'),
        'speak':  ( 520,200,  (195, 90,195), '▶ SPEAK'),
        'undo':   ( 440,100,  ( 30,145,255), '↩ UNDO'),
        'clear':  ( 330,150,  (110,118,130), '✕ CLEAR'),
        'error':  ( 220,200,  ( 55, 55,210), '! ERROR'),
    }

    def __init__(self):
        self._active = None; self._lock = threading.Lock()
        try:
            import winsound; self._ws=winsound; self._bt='win'
        except ImportError:
            self._bt='bell'

    def fire(self, ev):
        if ev not in self.EV: return
        freq,ms,col,lbl = self.EV[ev]
        def _b():
            try:
                if self._bt=='win': self._ws.Beep(freq,ms)
                else: print('\a',end='',flush=True)
            except: pass
        threading.Thread(target=_b,daemon=True).start()
        with self._lock: self._active=(time.time()+0.55,col,lbl)

    def draw(self, frame):
        with self._lock: a=self._active
        if not a: return
        expire,col,lbl=a; rem=expire-time.time()
        if rem<=0:
            with self._lock: self._active=None; return
        h,w=frame.shape[:2]; alpha=rem/0.55; bw=int(220*alpha)
        if bw>20:
            ov=frame.copy()
            cv2.rectangle(ov,(w-bw-4,60),(w-4,112),col,-1)
            cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
            cv2.putText(frame,lbl,(max(w-bw+4,w-210),94),
                        cv2.FONT_HERSHEY_SIMPLEX,0.72,(6,7,10),2,cv2.LINE_AA)
        ph=max(3,int(8*alpha))
        cv2.rectangle(frame,(0,h-ph-26),(w,h-26),col,-1)


# ══════════════════════════════════════════════════════════════════════════════
#  LETTER RECOGNISER  v5.3
#
#  KEY CHANGES vs v5.2:
#
#  FIX A — VOTE_WINDOW 30→18, model runs every frame (no %2 skip).
#    With 30 slots and every-other-frame skipping, you needed ~60 frames
#    (4s) to fill the buffer. Now 18 slots filled every frame = ~1.2s to
#    fill, so a held sign registers at the 1.5s target.
#
#  FIX B — Majority now measured against non-blank count, not total.
#    Previously blank '_' slots diluted the count. Now winner needs
#    VOTE_MAJORITY of non-blank frames, making it robust to brief dropouts.
#    A secondary check ensures at least 40% of ALL slots are the winner
#    (prevents a tiny non-blank sample from triggering).
#
#  FIX C — display_label falls back to _locked_label when locked.
#    The letter panel now shows the added letter during word-commit
#    countdown instead of going blank.
# ══════════════════════════════════════════════════════════════════════════════
class LetterRecogniser:

    def __init__(self):
        self._buf          = deque(maxlen=VOTE_WINDOW)
        self._stable_label = None
        self._stable_conf  = 0.0
        self._stable_since = None

        self._locked       = False
        self._locked_label = None
        self._diff_streak  = 0

        self._word_armed   = False
        self._word_since   = None

        self._added_label  = None
        self._after_add    = False
        self.display_label = None
        self.display_conf  = 0.0
        self.events        = set()

    # ── Public: call every frame ──────────────────────────────────────────────

    def update(self, raw_label, raw_conf):
        self.events = set()
        now = time.time()

        # 1. Push raw prediction into buffer
        if raw_label and raw_conf >= conf_thr(raw_label):
            self._buf.append((raw_label, raw_conf))
        else:
            self._buf.append(('_', 0.0))

        # 2. FIX B: Compute winner against non-blank frames
        #    Winner needs VOTE_MAJORITY of non-blank slots AND
        #    at least 40% of ALL slots (prevents micro-sample triggers)
        counts     = Counter(l for l,_ in self._buf if l != '_')
        non_blanks = sum(counts.values())
        winner     = counts.most_common(1)[0][0] if counts else None
        w_count    = counts[winner] if winner else 0

        # Require majority of non-blank frames AND minimum absolute presence
        min_absolute = int(VOTE_WINDOW * 0.40)  # at least 40% of all slots
        if (winner and non_blanks > 0
                and w_count / non_blanks >= VOTE_MAJORITY
                and w_count >= min_absolute):
            w_confs   = [c for l,c in self._buf if l == winner]
            cur_label = winner
            cur_conf  = sum(w_confs) / len(w_confs)
        else:
            cur_label = None
            cur_conf  = 0.0

        # 3. FIX C: display falls back to locked label so panel stays visible
        if cur_label:
            self.display_label = cur_label
            self.display_conf  = cur_conf
        elif self._locked and self._locked_label:
            # Keep showing the letter that was just added
            self.display_label = self._locked_label
            self.display_conf  = self._stable_conf
        else:
            self.display_label = None
            self.display_conf  = 0.0

        # ── Locked-state unlock logic ─────────────────────────────────────────
        if self._locked:
            self._after_add = True
            if cur_label is not None and cur_label != self._locked_label:
                self._diff_streak += 1
                if self._diff_streak >= UNLOCK_FRAMES:
                    self._locked       = False
                    self._after_add    = False
                    self._locked_label = None
                    self._diff_streak  = 0
                    self._stable_label = cur_label
                    self._stable_conf  = cur_conf
                    self._stable_since = now
                    self._word_armed   = False
                    self._word_since   = None
            else:
                self._diff_streak = 0

            # Word-commit countdown (still runs while locked)
            if self._word_armed and self._word_since:
                if now - self._word_since >= WORD_COMMIT_TIME:
                    self.events.add('commit_word')
                    self._word_armed = False
                    self._word_since = None
                    self._do_full_reset()

            return self.events

        # ── Normal (unlocked) state ───────────────────────────────────────────

        if cur_label is None:
            return self.events

        if cur_label != self._stable_label:
            self._stable_label = cur_label
            self._stable_conf  = cur_conf
            self._stable_since = now
            self._word_armed   = False
            self._word_since   = None
        else:
            self._stable_conf = cur_conf

        stable_elapsed = now - self._stable_since if self._stable_since else 0.0

        if stable_elapsed >= LETTER_ADD_TIME:
            self._added_label  = cur_label
            self._after_add    = True
            self._locked       = True
            self._locked_label = cur_label
            self._diff_streak  = 0
            self._word_armed   = True
            self._word_since   = now
            self.events.add('add_letter')

        return self.events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _do_full_reset(self):
        self._stable_label = None
        self._stable_conf  = 0.0
        self._stable_since = None
        self._locked       = False
        self._after_add    = False
        self._locked_label = None
        self._diff_streak  = 0
        self._word_armed   = False
        self._word_since   = None
        self._added_label  = None

    def reset(self):
        self._buf.clear()
        self._do_full_reset()
        self.display_label = None
        self.display_conf  = 0.0
        self.events        = set()

    def letter_progress(self):
        if self._locked:
            return 1.0
        if self._stable_since is None:
            return 0.0
        return min((time.time()-self._stable_since)/LETTER_ADD_TIME, 1.0)

    def word_progress(self):
        if not self._word_armed or self._word_since is None:
            return 0.0
        return min((time.time()-self._word_since)/WORD_COMMIT_TIME, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ABSENT TIMER
# ══════════════════════════════════════════════════════════════════════════════
class AbsentTimer:
    def __init__(self):
        self._since=None; self._frames=0
        self._af=self._tf=self._sf=False
        self.events=set()

    def update(self, has_hands):
        self.events=set()
        if has_hands:
            self._since=None; self._frames=0
            self._af=self._tf=self._sf=False
            return self.events
        self._frames+=1
        if self._frames < ABSENT_DEBOUNCE: return self.events
        if self._since is None: self._since=time.time()
        e=time.time()-self._since
        if   e>=T_SPEAK  and not self._sf: self.events.add('speak');  self._sf=True
        elif e>=T_TOGGLE and not self._tf: self.events.add('toggle'); self._tf=True
        elif e>=T_ACCEPT and not self._af: self.events.add('accept'); self._af=True
        return self.events

    def elapsed(self):
        return time.time()-self._since if self._since else 0.0

    def reset(self):
        self._since=None; self._frames=0
        self._af=self._tf=self._sf=False


# ══════════════════════════════════════════════════════════════════════════════
#  SPARKLINE
# ══════════════════════════════════════════════════════════════════════════════
class Sparkline:
    def __init__(self, n=40): self._v=deque([0.0]*n,maxlen=n)
    def push(self, v): self._v.append(float(v))
    def get(self): return list(self._v)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
class ASLSystem:

    def __init__(self):
        self._banner()
        self.grammar = GrammarEngine()
        self.lrec    = LetterRecogniser()
        self.absent  = AbsentTimer()
        self.ac      = Autocomplete()
        self.alerts  = Alerts()
        self._load_models()
        self._setup_mp()
        self._setup_tts()
        self._init_state()
        print("\n[READY]\n")

    # ── Init ──────────────────────────────────────────────────────────────────

    def _banner(self):
        print("""
╔══════════════════════════════════════════════════════╗
║        ASL SYSTEM  v5.3   —   How to use            ║
╠══════════════════════════════════════════════════════╣
║  LETTER MODE:                                        ║
║    Sign a letter and hold still                      ║
║    1.5s stable → letter auto-added                   ║
║    Change sign → timer resets for next letter        ║
║    After adding: hold 3s more → word committed       ║
║    Remove hand 1.5s → accept word into sentence      ║
║                                                      ║
║    A key = add current letter immediately            ║
║    SPACE = commit word immediately                   ║
║                                                      ║
║  WORD MODE:                                          ║
║    Sign a word → predictions shown                   ║
║    Remove hand 1.5s → accept top prediction          ║
║                                                      ║
║  GESTURE (hand absent):                              ║
║    1.5s=Accept   4s=Toggle mode   7s=Speak           ║
║                                                      ║
║  KEYS: S=speak M=toggle BKSP=delete Z=undo           ║
║        ENTER=clear 1-4=suggest D=debug Q=quit        ║
╚══════════════════════════════════════════════════════╝
""")

    def _init_state(self):
        self.mode        = 'words'
        self.sentence    = []
        self._undo       = []
        self.spelling    = []
        self.english     = ''
        self._speaking   = False
        self._fb_text    = ''
        self._fb_col     = (0,220,100)
        self._fb_until   = 0.0
        self._debug      = False
        self._frame      = 0
        self._sugg       = []
        self._spark      = Sparkline()
        self._mode_grace = 0.0
        self.top3        = []
        self.last_top3   = []
        self.wbuf        = deque(maxlen=150)

        self.C = dict(
            panel  =(28,30,40),   green  =(55,210,90),
            blue   =(220,150,30), orange =(30,145,255),
            yellow =(20,210,210), white  =(225,230,235),
            gray   =(110,118,130),red    =(55,55,210),
            purple =(195,90,195), teal   =(185,200,30),
            black  =(6,7,10),     ready  =(40,235,130),
            dim    =(45,48,60),
        )

    # ── Models ────────────────────────────────────────────────────────────────

    def _load_models(self):
        print("── Loading models ──────────────────────────────────")
        self.lm=None; self.le=None; self.lfd=63
        for mp_,ep_ in [
            ("models/letters_model.h5","models/letters_labels.pkl"),
            ("models/letter_model.h5", "models/letter_labels.pkl"),
        ]:
            if not (os.path.exists(mp_) and os.path.exists(ep_)): continue
            try:
                self.lm  = keras.models.load_model(mp_)
                self.lfd = int(self.lm.input_shape[-1])
                with open(ep_,'rb') as f: self.le=pickle.load(f)
                print(f"  [OK] Letters — {len(list(self.le.classes_))} classes, "
                      f"{self.lfd} feats")
                break
            except Exception as e: print(f"  [ERR] {mp_}: {e}")
        if self.lm is None: print("  [MISS] No letter model")

        self.wm=None; self.we=None; self.wmt=None; self.wil=72; self.wfd=126
        for mp_,ep_,tag in [
            ("models/advanced_words_model.h5","models/advanced_words_labels.pkl","adv"),
            ("models/words_model.h5",          "models/words_labels.pkl",         "basic"),
        ]:
            if not (os.path.exists(mp_) and os.path.exists(ep_)): continue
            try:
                raw=keras.models.load_model(mp_)
                with open(ep_,'rb') as f: self.we=pickle.load(f)
                self.wil=int(raw.input_shape[1]); self.wfd=int(raw.input_shape[2])
                self.wm=tf.function(
                    lambda x: raw(x,training=False),
                    input_signature=[tf.TensorSpec(
                        shape=[1,self.wil,self.wfd],dtype=tf.float32)])
                self.wmt=tag
                print(f"  [OK] Words ({tag}) — {len(list(self.we.classes_))} classes")
                break
            except Exception as e: print(f"  [ERR] {mp_}: {e}")
        if self.wm is None: print("  [MISS] No word model")
        print("────────────────────────────────────────────────────")

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _setup_mp(self):
        self._mph         = mp.solutions.hands
        self._mp_hands    = self._mph.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
            model_complexity=0)
        self._mp_draw        = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _setup_tts(self):
        self.tts=None
        if TTS_OK:
            try:
                self.tts=pyttsx3.init(); self.tts.setProperty('rate',148)
                for v in self.tts.getProperty('voices'):
                    if 'english' in v.name.lower() or 'en_' in v.id.lower():
                        self.tts.setProperty('voice',v.id); break
                print("  [OK] TTS")
            except: print("  [WARN] TTS failed")

    def _speak(self, text):
        if not self.tts or not text or self._speaking: return
        def _r():
            self._speaking=True
            try: self.tts.say(text); self.tts.runAndWait()
            except: pass
            self._speaking=False
        threading.Thread(target=_r,daemon=True).start()

    # ── Feature extraction ────────────────────────────────────────────────────

    def _orient(self, lm):
        l=lm.landmark
        def pt(i): return np.array([l[i].x,l[i].y,l[i].z],np.float32)
        w=pt(0); n=np.cross(pt(9)-w,pt(17)-pt(5)).astype(np.float32)
        nn=np.linalg.norm(n)
        if nn>0: n/=nn
        curls=[float(np.linalg.norm(pt(t)-w)/(np.linalg.norm(pt(b)-w)+1e-6))
               for t,b in zip([4,8,12,16,20],[2,5,9,13,17])]
        return n.tolist()+curls

    def _lfeats(self, lm):
        raw=[]
        for l in lm.landmark: raw.extend([l.x,l.y,l.z])
        a=np.array(raw,np.float32).reshape(-1,3); c=a-a.mean(0); s=np.std(c)
        if s>0: c/=s
        n=c.flatten().tolist()
        if self.lfd==71: return n+self._orient(lm)
        if len(n)<self.lfd: n+=[0.0]*(self.lfd-len(n))
        return n[:self.lfd]

    def _wfeats(self, multi_lm):
        per=self.wfd//2; pts=[]
        for i in range(2):
            if i<len(multi_lm):
                for l in multi_lm[i].landmark: pts.extend([l.x,l.y,l.z])
            else: pts.extend([0.0]*per)
        return pts[:self.wfd]

    # ── Prediction ────────────────────────────────────────────────────────────

    def _pred_letter(self, feats):
        if self.lm is None or len(feats)!=self.lfd: return None,0.0
        try:
            x=tf.constant(np.array(feats,np.float32).reshape(1,-1))
            p=self.lm(x,training=False).numpy()[0]
            i=int(np.argmax(p))
            l=(self.le.inverse_transform([i])[0]
               if hasattr(self.le,'inverse_transform') else self.le[i])
            return l,float(p[i])
        except Exception as e:
            if self._debug: print(f"[ERR letter] {e}")
            return None,0.0

    def _pred_word(self, seq):
        if self.wm is None or len(seq)<20: return []
        try:
            arr=np.array(seq,np.float32); t=self.wil; fd=self.wfd
            def prep(s):
                if self.wmt=='adv':
                    s=gaussian_filter1d(s,sigma=1.5,axis=0)
                if len(s)<t:
                    s=np.vstack([s,np.zeros((t-len(s),fd),np.float32)])
                else: s=s[-t:]
                if s.shape[1]<fd:
                    s=np.hstack([s,np.zeros((len(s),fd-s.shape[1]),np.float32)])
                return s[:,:fd]
            wins=[arr]
            if len(arr)>t+10: wins.append(arr[:-10])
            if len(arr)>t+20: wins.append(arr[:-20])
            acc=None
            for w in wins:
                p=self.wm(tf.constant(prep(w).reshape(1,t,fd),
                                       dtype=tf.float32)).numpy()[0]
                acc=p if acc is None else acc+p
            acc/=len(wins)
            top=np.argsort(acc)[-3:][::-1]
            return [(str(self.we.inverse_transform([i])[0]
                         if hasattr(self.we,'inverse_transform')
                         else self.we[i]),
                     float(acc[i])) for i in top]
        except Exception as e:
            if self._debug: print(f"[ERR word] {e}")
            return []

    # ── Actions ───────────────────────────────────────────────────────────────

    def _asl(self):
        parts=list(self.sentence)
        if self.spelling: parts.append(''.join(self.spelling))
        return ' '.join(parts) if parts else 'Start signing...'

    def _eng(self):
        asl=self._asl()
        return '' if asl=='Start signing...' else self.grammar.convert(asl)

    def _add_word(self, word):
        word=word.upper()
        if self.sentence and self.sentence[-1]==word:
            self._fb(f"Duplicate skipped: {word}",self.C['gray']); return
        self._undo.append(list(self.sentence))
        self.sentence.append(word)
        self.english=self.grammar.convert(self._asl())
        self._fb(f"✓ {word}",self.C['green'])
        self.alerts.fire('accept')
        print(f"  [+] {word}  |  {self._asl()}")

    def _commit_word(self):
        if self.spelling:
            word=''.join(self.spelling).upper()
            self._add_word(word)
            self.spelling=[]
            self.lrec.reset()
            self.alerts.fire('word')
            print(f"  [WORD] '{word}' committed")

    def _do_accept(self):
        if self.mode=='letters':
            if self.spelling:
                self._commit_word()
        else:
            picks=self.top3 or self.last_top3
            if picks:
                self._add_word(str(picks[0][0]))
                self.top3=self.last_top3=[]; self.wbuf.clear()

    def _do_toggle(self):
        self.mode='words' if self.mode=='letters' else 'letters'
        self.top3=self.last_top3=[]; self.wbuf.clear()
        self.spelling=[]; self.lrec.reset(); self.absent.reset()
        self._mode_grace=time.time()+MODE_GRACE
        self.alerts.fire('toggle')
        self._fb(f"→ {'LETTER' if self.mode=='letters' else 'WORD'} MODE"
                 +(f"  ({MODE_GRACE:.0f}s grace)" if self.mode=='letters' else ''),
                 self.C['yellow'])
        print(f"\n  [MODE] {self.mode.upper()}\n")

    def _do_speak(self):
        eng=self._eng()
        if not eng:
            self._fb("Nothing to speak",self.C['gray'])
            self.alerts.fire('error'); return
        self.english=eng
        self._fb("Speaking...",self.C['purple'])
        self.alerts.fire('speak')
        print(f"  [SPEAK] {eng}")
        self._speak(eng)

    def _do_undo(self):
        if self._undo:
            self.sentence=self._undo.pop(); self.english=self._eng()
            self._fb("Undo",self.C['orange']); self.alerts.fire('undo')
        else:
            self._fb("Nothing to undo",self.C['gray']); self.alerts.fire('error')

    def _fb(self, text, col):
        self._fb_text=text; self._fb_col=col; self._fb_until=time.time()+2.5

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAW HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _txt(self, f, s, pos, sc, col, th=1):
        cv2.putText(f,s,pos,cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)

    def _bar(self, f, x1,y1,x2,y2, pct, col, bg=(6,7,10)):
        cv2.rectangle(f,(x1,y1),(x2,y2),bg,-1)
        if pct>0: cv2.rectangle(f,(x1,y1),(x1+int((x2-x1)*pct),y2),col,-1)
        cv2.rectangle(f,(x1,y1),(x2,y2),(45,48,60),1)

    def _sparkline(self, f, vals, x,y,w,h,col):
        if len(vals)<2: return
        pts=[(x+int(i*w/(len(vals)-1)),y+h-int(v*h))
             for i,v in enumerate(vals)]
        for i in range(len(pts)-1):
            cv2.line(f,pts[i],pts[i+1],col,1,cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAW UI
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_ui(self, frame, num_hands):
        C=self.C; h,w=frame.shape[:2]; now=time.time()
        T=self._txt

        # ── Top bar ───────────────────────────────────────────────
        cv2.rectangle(frame,(0,0),(w,56),C['panel'],-1)
        mc=C['green'] if self.mode=='letters' else C['blue']
        T(frame,"[ LETTER MODE ]" if self.mode=='letters'
          else "[ WORD / PHRASE MODE ]",(14,38),1.0,mc,2)
        hc=C['ready'] if num_hands>0 else C['red']
        hl=f"{num_hands} hand{'s' if num_hands!=1 else ''}"
        bw=len(hl)*11+20
        cv2.rectangle(frame,(w-bw-4,8),(w-4,48),hc,-1)
        T(frame,hl,(w-bw+4,36),0.72,C['black'],2)
        if self._speaking:
            T(frame,"◀ SPEAKING",(w-bw-170,36),0.65,C['purple'],2)

        # ── Timer / status bar (BY=56 to BY+44) ───────────────────
        BY=56; BY2=100
        cv2.rectangle(frame,(0,BY),(w,BY2),(12,14,18),-1)
        grace=max(0.0,self._mode_grace-now)
        abs_e=self.absent.elapsed()

        if grace>0 and self.mode=='letters':
            pct=grace/MODE_GRACE
            cv2.rectangle(frame,(0,BY+6),(int(w*pct),BY2-6),C['yellow'],-1)
            msg=f"Get ready — {grace:.1f}s"
            tw=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.60,1)[0][0]
            T(frame,msg,(w//2-tw//2,BY+29),0.60,C['black'],2)

        elif num_hands==0 and abs_e>0:
            s1=int(w*T_ACCEPT/T_SPEAK); s2=int(w*T_TOGGLE/T_SPEAK)
            cv2.rectangle(frame,(0,BY),(s1,BY2),(18,42,18),-1)
            cv2.rectangle(frame,(s1,BY),(s2,BY2),(42,42,10),-1)
            cv2.rectangle(frame,(s2,BY),(w,BY2),(42,16,42),-1)
            cv2.line(frame,(s1,BY),(s1,BY2),(80,88,95),1)
            cv2.line(frame,(s2,BY),(s2,BY2),(80,88,95),1)
            T(frame,"ACCEPT",(8,BY+17),0.44,C['green'])
            T(frame,f"{T_ACCEPT}s",(8,BY+34),0.36,C['gray'])
            T(frame,"TOGGLE",(s1+6,BY+17),0.44,C['yellow'])
            T(frame,f"{T_TOGGLE}s",(s1+6,BY+34),0.36,C['gray'])
            T(frame,"SPEAK",(s2+6,BY+17),0.44,C['purple'])
            T(frame,f"{T_SPEAK}s",(s2+6,BY+34),0.36,C['gray'])
            fx=int(w*min(abs_e/T_SPEAK,1.0))
            bc=(C['green'] if abs_e<T_ACCEPT else
                C['yellow'] if abs_e<T_TOGGLE else C['purple'])
            cv2.rectangle(frame,(0,BY+6),(fx,BY2-6),bc,-1)
            if abs_e<T_ACCEPT:     msg=f"ACCEPT in {T_ACCEPT-abs_e:.1f}s"
            elif abs_e<T_TOGGLE:   msg=f"TOGGLE in {T_TOGGLE-abs_e:.1f}s"
            else:                  msg=f"SPEAK in {T_SPEAK-abs_e:.1f}s"
            tw=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)[0][0]
            T(frame,msg,(w//2-tw//2,BY+28),0.52,C['white'],1)

        elif self.mode=='letters' and num_hands>0:
            lp=self.lrec.letter_progress()
            wp=self.lrec.word_progress()
            if wp>0:
                cv2.rectangle(frame,(0,BY+6),(int(w*wp),BY2-6),C['orange'],-1)
                rem=max(0.0,WORD_COMMIT_TIME*(1.0-wp))
                msg=f"Keep holding — WORD SPACE in {rem:.1f}s"
                tw=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.55,1)[0][0]
                T(frame,msg,(w//2-tw//2,BY+28),0.55,C['black'],2)
            elif lp>0:
                col=C['ready'] if lp>=1.0 else C['teal']
                cv2.rectangle(frame,(0,BY+6),(int(w*lp),BY2-6),col,-1)
                if lp>=1.0:
                    msg="Adding letter..."
                else:
                    rem=max(0.0,LETTER_ADD_TIME*(1.0-lp))
                    msg=f"Hold still — ADD LETTER in {rem:.1f}s"
                tw=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.54,1)[0][0]
                T(frame,msg,(w//2-tw//2,BY+28),0.54,C['white'],1)
            else:
                T(frame,
                  "Sign a letter and hold still  |  A=add now  SPACE=commit word",
                  (8,BY+28),0.42,C['gray'])
        else:
            T(frame,
              "Remove hand: 1.5s=accept  4s=toggle  7s=speak  |  S=speak  Z=undo",
              (8,BY+28),0.40,C['gray'])

        # ── Letter panel ──────────────────────────────────────────
        if self.mode=='letters':
            px,py,pw,ph=12,108,265,225
            cv2.rectangle(frame,(px,py),(px+pw,py+ph),C['panel'],-1)

            if grace>0:
                cv2.rectangle(frame,(px,py),(px+pw,py+ph),(22,28,40),-1)
                cv2.rectangle(frame,(px,py),(px+pw,py+ph),C['yellow'],2)
                T(frame,"READY IN",(px+68,py+85),0.70,C['yellow'],2)
                T(frame,f"{grace:.1f}s",(px+55,py+150),2.4,C['yellow'],6)
                T(frame,"position hand",(px+32,py+185),0.50,C['gray'])

            elif self.lrec.display_label:
                lbl=self.lrec.display_label; conf=self.lrec.display_conf
                self._spark.push(conf)
                lp=self.lrec.letter_progress(); wp=self.lrec.word_progress()
                added=self.lrec._after_add

                rdy = conf>=conf_thr(lbl)
                if added:        col=C['orange']
                elif lp>=1.0:    col=C['ready']
                elif rdy:        col=C['green']
                else:            col=C['gray'] if not rdy else C['red']
                bord=3 if (lp>=1.0 or added) else 1
                cv2.rectangle(frame,(px,py),(px+pw,py+ph),col,bord)

                if added:        chip,chip_c="ADDED ✓",   C['orange']
                elif lp>=1.0:    chip,chip_c="ADDING...", C['ready']
                elif lp>0.5:     chip,chip_c="ALMOST",    C['teal']
                elif rdy:        chip,chip_c="DETECTED",  C['green']
                else:            chip,chip_c="LOW CONF",  C['gray']
                T(frame,chip,(px+pw-95,py+22),0.48,chip_c,2)

                T(frame,str(lbl),(px+72,py+130),3.6,col,7)
                T(frame,f"{conf:.0%}",(px+92,py+168),0.90,C['white'],2)
                if is_confusable(lbl):
                    T(frame,"hold very still",(px+46,py+188),0.42,C['red'])

                self._bar(frame,px+18,py+196,px+247,py+208,conf,col)
                self._sparkline(frame,self._spark.get(),
                                px+18,py+212,pw-36,24,col)

                bar_y=py+ph+6
                cv2.rectangle(frame,(px+18,bar_y),(px+pw-18,bar_y+10),(30,32,40),-1)
                if wp>0:
                    cv2.rectangle(frame,(px+18,bar_y),
                                  (px+18+int((pw-36)*wp),bar_y+10),C['orange'],-1)
                    T(frame,f"word {wp:.0%}",(px+18,bar_y+22),0.38,C['orange'])
                elif lp>0:
                    cv2.rectangle(frame,(px+18,bar_y),
                                  (px+18+int((pw-36)*lp),bar_y+10),C['teal'],-1)
                    T(frame,f"letter {lp:.0%}",(px+18,bar_y+22),0.38,C['teal'])

            else:
                cv2.rectangle(frame,(px,py),(px+pw,py+ph),(50,52,62),1)
                T(frame,"No model" if self.lm is None else "Show your hand...",
                  (px+22,py+115),0.65,C['gray'])

            if self.spelling:
                sx=px+pw+14
                T(frame,"spelling:",(sx,py+28),0.50,C['gray'])
                ws=''.join(self.spelling)
                if len(ws)>10: ws='...'+ws[-9:]
                T(frame,ws,(sx,py+72),1.6,C['orange'],3)
                T(frame,f"{len(self.spelling)} letters",
                  (sx,py+100),0.40,C['gray'])

        # ── Word panel ────────────────────────────────────────────
        elif self.mode=='words':
            d3=self.top3 if self.top3 else self.last_top3; px=12
            if d3:
                cols3=[C['green'],C['blue'],C['orange']]
                ranks=["TOP","2nd","3rd"]
                for i,(lbl,conf) in enumerate(d3[:3]):
                    bx=px+i*218; bby,bw2,bh=108,208,158
                    cv2.rectangle(frame,(bx,bby),(bx+bw2,bby+bh),C['panel'],-1)
                    cv2.rectangle(frame,(bx,bby),(bx+bw2,bby+bh),
                                  cols3[i],3 if i==0 else 1)
                    T(frame,ranks[i],(bx+10,bby+22),0.44,cols3[i])
                    fs=1.20 if len(str(lbl))<=8 else 0.80
                    T(frame,str(lbl),(bx+12,bby+102),fs,cols3[i],3)
                    T(frame,f"{conf:.0%}",(bx+70,bby+134),0.80,C['white'],2)
                    self._bar(frame,bx+12,bby+142,bx+196,bby+154,conf,cols3[i])
                T(frame,"Remove hand 1.5s = accept top",
                  (px,280),0.52,C['yellow'])
            else:
                bp=min(len(self.wbuf)/max(1,self.wil),1.0)
                cv2.rectangle(frame,(px,108),(px+520,290),C['panel'],-1)
                cv2.rectangle(frame,(px,108),(px+520,290),(58,60,75),1)
                T(frame,"WORD / PHRASE MODE",(px+20,148),0.90,C['blue'],2)
                T(frame,"Sign any word with 1 or 2 hands",(px+20,180),0.56,C['white'])
                T(frame,"Remove hand 1.5s → accept",(px+20,206),0.56,C['green'])
                T(frame,"Remove hand 4.0s → letter mode",(px+20,232),0.56,C['yellow'])
                T(frame,"Remove hand 7.0s → speak",(px+20,258),0.56,C['purple'])
                self._bar(frame,px+20,284,px+500,292,bp,C['blue'])

        # ── ASL + English rows ────────────────────────────────────
        gy=h-192
        cv2.rectangle(frame,(0,gy),(w,gy+52),(14,18,12),-1)
        cv2.line(frame,(0,gy),(w,gy),(45,125,55),1)
        T(frame,"ASL:",(12,gy+16),0.52,C['blue'],1)
        asl=self._asl()
        if len(asl)>70: asl='...'+asl[-68:]
        T(frame,asl,(68,gy+16),0.68,C['white'],2)
        T(frame,f"{len(self.sentence)}w",(w-55,gy+16),0.48,C['gray'])

        ey=gy+52
        cv2.rectangle(frame,(0,ey),(w,ey+90),(8,16,8),-1)
        cv2.line(frame,(0,ey),(w,ey),(38,175,55),2)
        badge='[AI]' if self.grammar.is_ai(self._asl()) else '[local]'
        bc=C['teal'] if badge=='[AI]' else C['gray']
        T(frame,"ENGLISH:",(12,ey+24),0.55,C['green'],1)
        T(frame,badge,(100,ey+24),0.48,bc,1)
        eng=self.english or self._eng() or "Grammar output appears here..."
        if len(eng)>58:
            T(frame,eng[:58],(12,ey+50),0.68,C['green'],2)
            T(frame,eng[58:116],(12,ey+78),0.68,C['green'],2)
        else:
            T(frame,eng,(12,ey+50),0.68,C['green'],2)
        bx=w-156; by3=ey+8
        sc2=C['purple'] if not self._speaking else C['dim']
        cv2.rectangle(frame,(bx,by3),(bx+146,by3+44),sc2,-1)
        cv2.rectangle(frame,(bx,by3),(bx+146,by3+44),C['white'],1)
        T(frame,"SPEAKING..." if self._speaking else "[ S ]  SPEAK",
          (bx+8,by3+30),0.62,C['white'],2)

        # ── Suggestion strip ──────────────────────────────────────
        if self._sugg:
            sy=h-232; sh=30
            cv2.rectangle(frame,(0,sy),(w,sy+sh),(20,22,30),-1)
            cv2.line(frame,(0,sy),(w,sy),(50,55,70),1)
            T(frame,"SUGGEST:",(8,sy+20),0.42,C['gray'])
            xc=92
            for idx,sg in enumerate(self._sugg[:4]):
                lbl2=f"[{idx+1}] {sg.replace('_',' ')}"
                tw2=cv2.getTextSize(lbl2,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)[0][0]
                cx2=xc+tw2+16
                if cx2>w-10: break
                cc=C['teal'] if idx==0 else (38,48,60)
                cv2.rectangle(frame,(xc,sy+4),(cx2,sy+sh-4),cc,-1)
                cv2.rectangle(frame,(xc,sy+4),(cx2,sy+sh-4),C['gray'],1)
                T(frame,lbl2,(xc+8,sy+20),0.44,
                  C['black'] if idx==0 else C['white'],1)
                xc=cx2+8

        # ── Feedback overlay ──────────────────────────────────────
        if now<self._fb_until and self._fb_text:
            ov=frame.copy()
            fw=min(len(self._fb_text)*13+44,w-40)
            fx=w//2-fw//2; fy=h//2-36
            cv2.rectangle(ov,(fx,fy),(fx+fw,fy+56),(12,14,20),-1)
            cv2.rectangle(ov,(fx,fy),(fx+fw,fy+56),self._fb_col,2)
            cv2.addWeighted(ov,0.87,frame,0.13,0,frame)
            T(frame,self._fb_text,(fx+14,fy+36),0.78,self._fb_col,2)

        # ── Alert overlay ─────────────────────────────────────────
        self.alerts.draw(frame)

        # ── Controls bar ──────────────────────────────────────────
        cv2.rectangle(frame,(0,h-26),(w,h),C['black'],-1)
        T(frame,
          "A=add letter  SPACE=commit word  S=speak  M=toggle  BKSP=del  "
          "Z=undo  1-4=suggest  ENTER=clear  Q=quit",
          (10,h-8),0.37,C['gray'])

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): print("[ERR] No camera"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        cap.set(cv2.CAP_PROP_FPS,30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        WN='ASL System v5.3'
        cv2.namedWindow(WN,cv2.WINDOW_NORMAL)

        while True:
            ret,frame=cap.read()
            if not ret: continue
            frame=cv2.flip(frame,1); self._frame+=1
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=self._mp_hands.process(rgb)
            hands=bool(res and res.multi_hand_landmarks)
            nh=len(res.multi_hand_landmarks) if hands else 0

            if hands:
                for hnd in res.multi_hand_landmarks:
                    self._mp_draw.draw_landmarks(
                        frame,hnd,self._mph.HAND_CONNECTIONS,
                        self._mp_draw_styles.get_default_hand_landmarks_style(),
                        self._mp_draw_styles.get_default_hand_connections_style())

            abs_ev=self.absent.update(hands)
            grace=time.time()<self._mode_grace

            if not grace:
                if 'speak'  in abs_ev: self._do_speak()
                if 'toggle' in abs_ev: self._do_toggle()
                if 'accept' in abs_ev: self._do_accept()

            # ── Letter mode ────────────────────────────────────────
            if self.mode=='letters':
                if hands and not grace:
                    lm0   = res.multi_hand_landmarks[0]
                    feats = self._lfeats(lm0)
                    # FIX A: run model EVERY frame — no more %2 skip
                    rl, rc = self._pred_letter(feats)
                    lev=self.lrec.update(rl,rc)
                    self._spark.push(rc if rc else 0.0)

                    if 'add_letter' in lev and self.lrec._added_label:
                        ltr=self.lrec._added_label
                        self.spelling.append(ltr)
                        self._fb(f"+ '{ltr}'  →  {''.join(self.spelling)}",
                                 self.C['teal'])
                        self.alerts.fire('letter')
                        print(f"  [L] '{ltr}' → {''.join(self.spelling)}")

                    if 'commit_word' in lev and self.spelling:
                        self._commit_word()

                elif not hands:
                    self.lrec.reset()
                    self._spark.push(0.0)

            # ── Word mode ──────────────────────────────────────────
            elif self.mode=='words' and hands:
                self.wbuf.append(self._wfeats(res.multi_hand_landmarks))
                if len(self.wbuf)>=min(self.wil,30) and self._frame%8==0:
                    p=self._pred_word(list(self.wbuf))
                    if p and p[0][1]>=0.40: self.top3=self.last_top3=p
                    else: self.top3=[]

            self._sugg=self.ac.suggest(self.sentence,self.spelling)

            self._draw_ui(frame,nh)
            cv2.imshow(WN,frame)
            key=cv2.waitKey(1)&0xFF

            if   key==ord('q'): break
            elif key==ord('d'):
                self._debug=not self._debug
                print(f"  Debug: {self._debug}")
            elif key==ord('s'): self._do_speak()
            elif key==ord('m'): self._do_toggle()
            elif key==ord('z') or key==26: self._do_undo()

            elif key==ord('a') and self.mode=='letters':
                lbl=self.lrec.display_label
                if lbl:
                    self.spelling.append(lbl)
                    self._fb(f"+ '{lbl}'  →  {''.join(self.spelling)}",
                             self.C['teal'])
                    self.alerts.fire('letter')
                    print(f"  [MANUAL L] '{lbl}' → {''.join(self.spelling)}")
                    self.lrec.reset()

            elif key==ord(' ') and self.mode=='letters':
                if self.spelling:
                    self._commit_word()
                else:
                    self._fb("No letters to commit",self.C['gray'])

            elif key in (ord('1'),ord('2'),ord('3'),ord('4')):
                i=key-ord('1')
                if i<len(self._sugg): self._add_word(self._sugg[i])

            elif key==8:  # BACKSPACE
                if self.mode=='letters' and self.spelling:
                    rem=self.spelling.pop()
                    self._fb(f"Deleted '{rem}'  → {''.join(self.spelling)}",
                             self.C['red'])
                    self.alerts.fire('undo')
                elif self.sentence:
                    r=self.sentence.pop(); self.english=self._eng()
                    self._fb(f"Deleted word: {r}",self.C['red'])
                    self.alerts.fire('undo'); print(f"  [DEL] {r}")

            elif key==13:  # ENTER
                self.sentence.clear(); self.spelling.clear()
                self._undo.clear(); self.top3=self.last_top3=[]
                self.english=''; self.wbuf.clear()
                self.lrec.reset(); self.absent.reset()
                self._fb("Cleared",self.C['gray'])
                self.alerts.fire('clear'); print("  [CLEAR]")

        cap.release(); cv2.destroyAllWindows()
        if self.tts:
            try: self.tts.stop()
            except: pass
        print("\n[DONE]")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists("models"):
        print("\n[ERR] No models/ folder.\n"
              "  Letters: python ultra_robust_trainer.py\n"
              "  Words:   python advanced_word_trainer.py\n")
        return
    print("Grammar warm-up:")
    g=GrammarEngine()
    for ex in ["hello i home","i hungry","me go store","i_love_you thank_you"]:
        print(f"  '{ex}' -> '{g.local.convert(ex)}'")
    print()
    try:
        ASLSystem().run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback; print(f"\n[FATAL] {e}"); traceback.print_exc()


if __name__=="__main__":
    main()