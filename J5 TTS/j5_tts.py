"""
audiolens — j5 text to speech evaluation

benchmarks 2 tts models on 50 test sentences:
  - 20 harvard sentences  (phonetically balanced, standard benchmark)
  - 30 audiolens sentences (real-world doc types: bills, prescriptions, menus, etc.)

models tested:
  1. edge tts  — microsoft neural tts, online (no hf download needed)
  2. kokoro    — open-source neural tts, offline (loaded from ./models/kokoro-82m)

metrics:
  - avg synthesis time per sentence (ms)
  - total synthesis time for all sentences (ms)
  - offline capability (yes/no)
  - audio files saved to results/j5_audio/ for manual mos scoring

usage:
    python j5_tts.py
"""

# -- config --

MODELS_DIR   = './models'
RESULTS_DIR  = './results'
AUDIO_DIR    = f'{RESULTS_DIR}/j5_audio'
RESULTS_FILE = f'{RESULTS_DIR}/j5_tts_results.json'
LOG_FILE     = f'{RESULTS_DIR}/j5_run_log.txt'

# kokoro model path — downloaded by populate_models.py
MODEL_PATHS = {
    'kokoro': f'{MODELS_DIR}/kokoro-82m',
}

# tts voice settings
EDGE_TTS_VOICE = 'en-GB-SoniaNeural'
KOKORO_LANG    = 'b'          # b = british english
KOKORO_VOICE   = 'bf_emma'


# -- imports --

import os
import sys
import json
import time
import asyncio
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,   exist_ok=True)


# -- tee logger --

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee(LOG_FILE)


# -- preflight: check kokoro model folder exists --

print('=' * 60)
print('  audiolens — j5 text to speech evaluation')
print('=' * 60)

kokoro_path = MODEL_PATHS['kokoro']
if not os.path.isdir(kokoro_path) or len(os.listdir(kokoro_path)) == 0:
    print(f'\n[error] kokoro model folder missing or empty: {kokoro_path}')
    print('\nrun this first:')
    print('  python populate_models.py')
    sys.exit(1)

print(f'kokoro path : {os.path.abspath(kokoro_path)}')
print(f'audio dir   : {os.path.abspath(AUDIO_DIR)}')
print('preflight passed.')
print('=' * 60)


# -- test sentences --

# 20 harvard sentences — ieee phonetically balanced standard
HARVARD_SENTENCES = [
    {'id':  1, 'type': 'harvard', 'text': 'The birch canoe slid on the smooth planks.'},
    {'id':  2, 'type': 'harvard', 'text': 'Glue the sheet to the dark blue background.'},
    {'id':  3, 'type': 'harvard', 'text': 'It is easy to tell the depth of a well.'},
    {'id':  4, 'type': 'harvard', 'text': 'These days a chicken leg is a rare dish.'},
    {'id':  5, 'type': 'harvard', 'text': 'Rice is often served in round bowls.'},
    {'id':  6, 'type': 'harvard', 'text': 'The juice of lemons makes fine punch.'},
    {'id':  7, 'type': 'harvard', 'text': 'The box was thrown beside the parked truck.'},
    {'id':  8, 'type': 'harvard', 'text': 'Four hours of steady work faced us.'},
    {'id':  9, 'type': 'harvard', 'text': 'Large size in stockings is hard to sell.'},
    {'id': 10, 'type': 'harvard', 'text': 'The boy was there when the sun rose.'},
    {'id': 11, 'type': 'harvard', 'text': 'A rod is used to catch pink salmon.'},
    {'id': 12, 'type': 'harvard', 'text': 'Kick the ball straight and follow through.'},
    {'id': 13, 'type': 'harvard', 'text': 'Help the woman get back to her feet.'},
    {'id': 14, 'type': 'harvard', 'text': 'A pot of tea helps to pass the evening.'},
    {'id': 15, 'type': 'harvard', 'text': 'Smoky fumes make the town look bleak.'},
    {'id': 16, 'type': 'harvard', 'text': 'Oak is strong and also gives shade.'},
    {'id': 17, 'type': 'harvard', 'text': 'The source of the big river is the clear spring.'},
    {'id': 18, 'type': 'harvard', 'text': 'The stray cat gave birth to kittens.'},
    {'id': 19, 'type': 'harvard', 'text': 'A blue crane is a tall wading bird.'},
    {'id': 20, 'type': 'harvard', 'text': 'The wrist was badly strained and hung limp.'},
]

# 30 audiolens sentences — one or more per real-world document type
AUDIOLENS_SENTENCES = [
    # restaurant bills
    {'id': 21, 'type': 'restaurant_bill', 'text': 'Your total for this evening comes to thirty-two pounds and fifty pence, including service charge.'},
    {'id': 22, 'type': 'restaurant_bill', 'text': 'Table four: two pasta dishes and one glass of house wine. Total eighteen pounds.'},
    {'id': 23, 'type': 'restaurant_bill', 'text': 'Thank you for dining with us. Your bill including VAT is forty-five pounds and twenty pence.'},

    # atm receipts
    {'id': 24, 'type': 'atm_receipt', 'text': 'Cash withdrawal of one hundred pounds completed. Available balance is three hundred and forty-two pounds.'},
    {'id': 25, 'type': 'atm_receipt', 'text': 'Transaction declined. Insufficient funds. Your current balance is twelve pounds and thirty pence.'},
    {'id': 26, 'type': 'atm_receipt', 'text': 'Your ATM withdrawal of fifty pounds was processed on the twenty-third of March at nine forty-five AM.'},

    # medicine tablet labels
    {'id': 27, 'type': 'medicine_label', 'text': 'Paracetamol five hundred milligrams. Take one to two tablets every four to six hours. Do not exceed eight tablets in twenty-four hours.'},
    {'id': 28, 'type': 'medicine_label', 'text': 'Amoxicillin two hundred and fifty milligrams capsules. Take one capsule three times daily for seven days with or without food.'},
    {'id': 29, 'type': 'medicine_label', 'text': 'Ibuprofen four hundred milligrams. Take one tablet with food every six to eight hours. Maximum three tablets per day.'},

    # syrup labels
    {'id': 30, 'type': 'syrup_label', 'text': 'Benylin cough syrup. Adults and children over twelve: take two five-millilitre spoonfuls every four hours. Shake well before use.'},
    {'id': 31, 'type': 'syrup_label', 'text': 'Paracetamol oral suspension, two hundred and fifty milligrams per five millilitres, for children aged six to twelve years.'},
    {'id': 32, 'type': 'syrup_label', 'text': 'Piriton allergy syrup. Children aged six to twelve: one five-millilitre spoonful three times daily. Do not exceed three doses in twenty-four hours.'},

    # prescription slips
    {'id': 33, 'type': 'prescription', 'text': 'Patient: Mrs. Sarah Ahmed. Prescribed Metformin five hundred milligrams twice daily with meals. Review in three months.'},
    {'id': 34, 'type': 'prescription', 'text': 'Doctor James Wilson prescribes Omeprazole twenty milligrams once daily before breakfast for four weeks.'},
    {'id': 35, 'type': 'prescription', 'text': 'Repeat prescription for Salbutamol inhaler. Two puffs as required. Do not exceed eight puffs in twenty-four hours.'},

    # utility bills
    {'id': 36, 'type': 'utility_bill', 'text': 'Your electricity bill for March is ninety-four pounds and sixty pence. Payment is due by the fifteenth of April.'},
    {'id': 37, 'type': 'utility_bill', 'text': 'British Gas account number seven seven three four five. Current balance outstanding is one hundred and twelve pounds.'},
    {'id': 38, 'type': 'utility_bill', 'text': 'Water usage for the quarter ending March was forty-two cubic metres. Amount due is sixty-seven pounds.'},

    # restaurant menus
    {'id': 39, 'type': 'menu', 'text': "Today's special: pan-seared salmon with lemon butter sauce, served with seasonal vegetables and new potatoes. Fifteen pounds ninety-nine."},
    {'id': 40, 'type': 'menu', 'text': 'Margherita pizza with fresh mozzarella and san marzano tomatoes. Available in nine or twelve inch. From eleven pounds fifty.'},
    {'id': 41, 'type': 'menu', 'text': 'Dietary note: this dish contains gluten, dairy and nuts. Suitable for vegetarians. Please inform staff of any allergies.'},

    # official letters
    {'id': 42, 'type': 'letter', 'text': 'Dear Mr. Patel, we are writing to confirm your appointment on Monday the seventh of April at two thirty in the afternoon.'},
    {'id': 43, 'type': 'letter', 'text': 'Your application for the position of senior software engineer has been received and is currently under review.'},
    {'id': 44, 'type': 'letter', 'text': 'This letter confirms that your council tax band has been reassigned, effective from the first of April.'},

    # product specifications
    {'id': 45, 'type': 'specification', 'text': 'Samsung Galaxy display: six point one inch dynamic AMOLED. Battery: four thousand milliamp hours. Storage: one hundred and twenty-eight gigabytes.'},
    {'id': 46, 'type': 'specification', 'text': 'Bosch washing machine capacity eight kilograms. Energy rating A. Maximum spin speed one thousand four hundred RPM.'},
    {'id': 47, 'type': 'specification', 'text': 'Active ingredient: ibuprofen lysine equivalent to ibuprofen two hundred milligrams. Each capsule contains two hundred milligrams.'},

    # notices and news
    {'id': 48, 'type': 'notice', 'text': 'Road closure notice: High Street will be closed to traffic from Monday to Wednesday due to emergency water main repairs.'},
    {'id': 49, 'type': 'notice', 'text': 'Community centre open day this Saturday from ten AM to four PM. Free entry for all residents. Refreshments will be provided.'},
    {'id': 50, 'type': 'notice', 'text': 'Strong winds forecast for coastal areas tonight. Gusts of up to seventy miles per hour are expected. Please take care.'},
]

ALL_SENTENCES = HARVARD_SENTENCES + AUDIOLENS_SENTENCES

print(f'\ntest set: {len(HARVARD_SENTENCES)} harvard + {len(AUDIOLENS_SENTENCES)} audiolens = {len(ALL_SENTENCES)} total sentences')


# -- helpers --

def make_audio_dir(model_name):
    path = os.path.join(AUDIO_DIR, model_name)
    os.makedirs(path, exist_ok=True)
    return path


def compute_tts_metrics(times_ms, model_name, offline, size_mb):
    return {
        'model':         model_name,
        'avg_ms':        round(np.mean(times_ms), 2),
        'min_ms':        round(np.min(times_ms),  2),
        'max_ms':        round(np.max(times_ms),  2),
        'total_ms':      round(np.sum(times_ms),  2),
        'offline':       offline,
        'size_mb':       size_mb,
        'sentences_run': len(times_ms),
    }


def print_tts_metrics(m):
    print(f'\n--- {m["model"]} ---')
    print(f'  avg synthesis time : {m["avg_ms"]} ms')
    print(f'  min / max          : {m["min_ms"]} ms / {m["max_ms"]} ms')
    print(f'  total for 50 sent. : {m["total_ms"]} ms  ({m["total_ms"]/1000:.1f}s)')
    print(f'  offline capable    : {"yes" if m["offline"] else "no"}')
    print(f'  model size         : {m["size_mb"]} MB')
    print(f'  note               : audio saved to {AUDIO_DIR}/{m["model"].lower().replace(" ", "_")}/')
    print(f'                       listen and assign mos score (1-5) for report')


# -- model 1: edge tts --
# microsoft neural tts, online only. no hf download needed.
# uses asyncio since the edge_tts library is async.

print('\n' + '=' * 60)
print('  MODEL 1 — Edge TTS  (online, microsoft neural)')
print('=' * 60)

try:
    import edge_tts

    print(f'voice: {EDGE_TTS_VOICE}')
    print(f'synthesising {len(ALL_SENTENCES)} sentences...')

    audio_dir_edge = make_audio_dir('edge_tts')
    edge_times     = []
    edge_failed    = 0

    async def synthesize_edge(text, outfile):
        """synthesises one sentence and saves to outfile."""
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        await communicate.save(outfile)

    for s in ALL_SENTENCES:
        outfile = os.path.join(audio_dir_edge, f'sentence_{s["id"]:02d}.mp3')
        try:
            t0 = time.time()
            asyncio.run(synthesize_edge(s['text'], outfile))
            t1 = time.time()
            edge_times.append((t1 - t0) * 1000)
        except Exception as e:
            print(f'  [warn] sentence {s["id"]} failed: {e}')
            edge_failed += 1

    if edge_failed:
        print(f'  {edge_failed} sentences failed — skipped from metrics.')

    edge_metrics = compute_tts_metrics(edge_times, 'Edge TTS', offline=False, size_mb=0)
    print_tts_metrics(edge_metrics)

except Exception as e:
    print(f'[error] edge tts failed: {e}')
    print('make sure edge-tts is installed: pip install edge-tts')
    edge_metrics = None


# -- model 2: kokoro --
# open-source neural tts, offline. loaded from ./models/kokoro-82m.
# produces high-quality speech at 24000 hz sample rate.

print('\n' + '=' * 60)
print('  MODEL 2 — Kokoro  (offline, open-source neural)')
print(f'  path: {MODEL_PATHS["kokoro"]}')
print('=' * 60)

try:
    import soundfile as sf
    from kokoro import KPipeline

    # try local path first, fall back to hf hub if local loading fails
    try:
        pipeline = KPipeline(lang_code=KOKORO_LANG, repo_id=MODEL_PATHS['kokoro'])
        print(f'kokoro loaded from local path.')
    except Exception:
        pipeline = KPipeline(lang_code=KOKORO_LANG, repo_id='hexgrad/Kokoro-82M')
        print(f'kokoro loaded from hf hub (local path failed).')

    print(f'voice: {KOKORO_VOICE}')
    print(f'synthesising {len(ALL_SENTENCES)} sentences...')

    audio_dir_kokoro = make_audio_dir('kokoro')
    kokoro_times     = []
    kokoro_failed    = 0

    for s in ALL_SENTENCES:
        outfile = os.path.join(audio_dir_kokoro, f'sentence_{s["id"]:02d}.wav')
        try:
            t0     = time.time()
            chunks = []
            for _, _, audio in pipeline(s['text'], voice=KOKORO_VOICE, speed=1.0):
                chunks.append(audio)
            t1 = time.time()

            if chunks:
                audio_array = np.concatenate(chunks)
                sf.write(outfile, audio_array, 24000)
                kokoro_times.append((t1 - t0) * 1000)
            else:
                print(f'  [warn] sentence {s["id"]} produced no audio.')
                kokoro_failed += 1

        except Exception as e:
            print(f'  [warn] sentence {s["id"]} failed: {e}')
            kokoro_failed += 1

    if kokoro_failed:
        print(f'  {kokoro_failed} sentences failed — skipped from metrics.')

    kokoro_metrics = compute_tts_metrics(kokoro_times, 'Kokoro', offline=True, size_mb=326)
    print_tts_metrics(kokoro_metrics)

except Exception as e:
    print(f'[error] kokoro failed: {e}')
    print('make sure kokoro and soundfile are installed: pip install kokoro soundfile')
    kokoro_metrics = None


# -- final comparison table --

all_metrics = {
    'Edge TTS': edge_metrics,
    'Kokoro':   kokoro_metrics,
}

valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}

print('\n' + '=' * 75)
print('  J5 TEXT TO SPEECH — FINAL COMPARISON')
print('=' * 75)
print(f"{'Model':<15} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Total (s)':>10} {'Offline':>8} {'MB':>6}")
print('-' * 75)
for name, m in valid_metrics.items():
    offline_str = 'yes' if m['offline'] else 'no'
    print(f"{name:<15} {m['avg_ms']:>10} {m['min_ms']:>10} {m['max_ms']:>10} "
          f"{m['total_ms']/1000:>9.1f}s {offline_str:>8} {m['size_mb']:>6}")
print('=' * 75)
print('\nnote: audio quality (mos score) requires manual listening evaluation.')
print('      audio files saved per model in results/j5_audio/ for that purpose.')


# -- comparison bar chart --

if valid_metrics:
    model_names = list(valid_metrics.keys())
    avg_times   = [m['avg_ms']        for m in valid_metrics.values()]
    total_times = [m['total_ms']/1000 for m in valid_metrics.values()]
    offline_val = [1 if m['offline'] else 0 for m in valid_metrics.values()]
    colors      = ['steelblue', 'darkorange'][:len(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, values, title, ylabel in zip(
        axes,
        [avg_times, total_times,        offline_val],
        ['Avg Synthesis Time (ms)', 'Total for 50 Sentences (s)', 'Offline Capable'],
        ['Milliseconds',            'Seconds',                    '1 = Yes, 0 = No'],
    ):
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=10, ha='right')
        for bar, val in zip(bars, values):
            label = f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) * 0.02),
                    label, ha='center', va='bottom', fontsize=9)

    plt.suptitle('AudioLens J5 — TTS Model Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart_path = f'{RESULTS_DIR}/j5_comparison_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\ncomparison chart saved -> {chart_path}')


# -- winner selection --
# composite: 70% speed (lower = better) + 30% offline capability

if valid_metrics:
    max_avg = max(m['avg_ms'] for m in valid_metrics.values())

    composite = {}
    for name, m in valid_metrics.items():
        speed_score     = 1 - (m['avg_ms'] / max_avg)
        offline_score   = 1.0 if m['offline'] else 0.0
        composite[name] = round((0.70 * speed_score) + (0.30 * offline_score), 4)

    winner = max(composite, key=composite.get)

    print('\n' + '=' * 55)
    print('  composite score  (speed 70% + offline 30%)')
    print('=' * 55)
    for name, score in sorted(composite.items(), key=lambda x: -x[1]):
        marker = '  <-- winner' if name == winner else ''
        print(f'  {name:<15}: {score}{marker}')

    print(f'\nj5 winner      : {winner}')
    print(f'avg time (ms)  : {valid_metrics[winner]["avg_ms"]}')
    print(f'offline        : {"yes" if valid_metrics[winner]["offline"] else "no"}')
    print(f'size (mb)      : {valid_metrics[winner]["size_mb"]}')
    print(f'\nreminder: also evaluate audio quality manually using the saved audio files.')
    print(f'          mos scoring guide — 1: bad, 2: poor, 3: fair, 4: good, 5: excellent')
else:
    winner    = None
    composite = {}
    print('\n[warn] no models ran successfully — no winner to select.')


# -- save results to json --

output = {
    'juncture': 'J5 — Text to Speech',
    'config': {
        'test_sentences':  len(ALL_SENTENCES),
        'harvard_count':   len(HARVARD_SENTENCES),
        'audiolens_count': len(AUDIOLENS_SENTENCES),
        'edge_tts_voice':  EDGE_TTS_VOICE,
        'kokoro_voice':    KOKORO_VOICE,
        'models_dir':      os.path.abspath(MODELS_DIR),
        'audio_dir':       os.path.abspath(AUDIO_DIR),
    },
    'models': {
        name: {
            'metrics':         m,
            'composite_score': composite.get(name, None),
        }
        for name, m in valid_metrics.items()
    },
    'winner': winner,
    'mos_note': 'manual mos scoring required — listen to audio files in results/j5_audio/',
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nfull results saved -> {RESULTS_FILE}')
print('\ndone.')