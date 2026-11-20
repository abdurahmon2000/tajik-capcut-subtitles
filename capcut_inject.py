import copy
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "facebook/mms-1b-all"
DEFAULT_LANG = "tgk"
MAX_WORDS_PER_BLOCK = 3
INSTAGRAM_URL = "https://www.instagram.com/hafizoff.dev?mdxt=MWsxcmxiOGNndnBuaQ=="

IS_MAC = sys.platform == "darwin"
UI_FONT = "Helvetica Neue" if IS_MAC else "Segoe UI"

GEMINI_PROMPT = (
    "Транскрибируй речь из этого аудио/видео максимально точно, слово в слово, "
    "без таймкодов, без переводов и без комментариев — верни ТОЛЬКО чистый сплошной текст.\n"
    "Все языки, которые используются в аудио/видео, распознавай и записывай с максимальной точностью "
    "в соответствии с их звучанием в оригинале. Не добавляй заголовков, пояснений или текста в скобках."
)

_cached_processor = None
_cached_model = None
_cached_device = None


def get_mms_model(progress_cb):
    global _cached_processor, _cached_model, _cached_device
    if _cached_model is None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        gc.collect()
        if torch.cuda.is_available():
            _cached_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _cached_device = "mps"
        else:
            _cached_device = "cpu"

        dtype = torch.float16 if _cached_device == "cuda" else torch.float32

        progress_cb(25, f"Загрузка нейросети Meta MMS ({_cached_device.upper()})...")
        _cached_processor = AutoProcessor.from_pretrained(MODEL_ID)
        progress_cb(40, "Загрузка весов модели...")
        _cached_model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        progress_cb(55, "Подключение языкового модуля...")
        _cached_processor.tokenizer.set_target_lang(DEFAULT_LANG)
        _cached_model.load_adapter(DEFAULT_LANG)
        if _cached_device in ("cuda", "mps"):
            _cached_model = _cached_model.to(_cached_device, dtype=dtype)
        _cached_model.eval()
        gc.collect()
    else:
        progress_cb(55, f"Нейросеть готова ({_cached_device.upper()})...")
    return _cached_processor, _cached_model, _cached_device


def align_text_to_audio(audio_path, raw_text, progress_cb):
    progress_cb(10, f"Извлечение аудио из {os.path.basename(audio_path)}...")
    import imageio_ffmpeg
    import soundfile as sf
    import torch
    import torchaudio

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    temp_wav = os.path.join(BASE_DIR, "temp_align.wav")
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", temp_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    progress_cb(20, "Чтение звуковой волны...")
    data_wav, sr = sf.read(temp_wav, dtype="float32")
    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass

    total_samples = len(data_wav)
    processor, model, device = get_mms_model(progress_cb)

    chunk_sec = 20
    chunk_samples = chunk_sec * sr
    emissions_list = []
    num_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)
    dtype = torch.float16 if device == "cuda" else torch.float32

    with torch.inference_mode():
        for idx_c in range(num_chunks):
            start_s = idx_c * chunk_samples
            end_s = min(total_samples, start_s + chunk_samples)
            if end_s - start_s < 400:
                continue

            pct = 55 + int(25 * (idx_c + 1) / num_chunks)
            progress_cb(pct, f"Обработка речи ({idx_c + 1}/{num_chunks})...")

            seg_tensor = torch.from_numpy(data_wav[start_s:end_s]).unsqueeze(0)
            if device in ("cuda", "mps"):
                seg_tensor = seg_tensor.to(device, dtype=dtype)

            logits = model(seg_tensor).logits
            em = torch.log_softmax(logits.float(), dim=-1).cpu()
            emissions_list.append(em)
            del seg_tensor, logits

    del data_wav
    gc.collect()

    emissions = torch.cat(emissions_list, dim=1)
    del emissions_list

    progress_cb(82, "Привязка каждого слова к таймкодам...")
    orig_words = raw_text.split()
    clean_words = [re.sub(r"[^\w\sъ]", "", w.lower()) for w in orig_words]
    vocab = processor.tokenizer.get_vocab()

    word_tokens, valid_orig_words = [], []
    for orig_w, clean_w in zip(orig_words, clean_words):
        tokens = [vocab[c] for c in clean_w if c in vocab]
        if tokens:
            word_tokens.append(tokens)
            valid_orig_words.append(orig_w)

    if not word_tokens:
        raise ValueError("Текст пустой или не содержит распознаваемых символов!")

    flat_tokens = [t for wt in word_tokens for t in wt]
    targets = torch.tensor([flat_tokens], dtype=torch.int32)

    aligned_tokens, scores = torchaudio.functional.forced_align(emissions, targets, blank=0)
    token_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    ratio = (total_samples / sr) / emissions.shape[1]
    word_spans = []
    idx = 0
    for orig_w, wt in zip(valid_orig_words, word_tokens):
        s_sec = token_spans[idx].start * ratio
        e_sec = token_spans[idx + len(wt) - 1].end * ratio
        word_spans.append((orig_w, s_sec, e_sec))
        idx += len(wt)

    del emissions, targets, aligned_tokens, scores
    gc.collect()

    return word_spans


def find_latest_project():
    if IS_MAC:
        base_drafts = os.path.expanduser("~/Movies/CapCut/User Data/Projects/com.lveditor.draft/*")
    else:
        base_drafts = os.path.expandvars(r"%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft\*")

    draft_dirs = [x for x in glob.glob(base_drafts) if os.path.isdir(x)]
    return max(draft_dirs, key=os.path.getmtime) if draft_dirs else None


def find_audio_from_capcut(draft):
    for mat in draft.get("materials", {}).get("audios", []) + draft.get("materials", {}).get("videos", []):
        path = mat.get("path", "")
        if path and os.path.exists(path):
            return path
    return None


def group_words_smartly(word_spans, max_words=MAX_WORDS_PER_BLOCK, pause_threshold=0.45):
    if not word_spans:
        return []

    chunks = []
    current_chunk = []
    punct_break_re = re.compile(r"[.!?,:;…—»\"]+$")

    for i, item in enumerate(word_spans):
        w_text, w_s, w_e = item
        current_chunk.append(item)

        is_last = (i == len(word_spans) - 1)
        has_punct = bool(punct_break_re.search(w_text))
        reached_limit = (len(current_chunk) >= max_words)

        has_pause = False
        if not is_last:
            next_start = word_spans[i + 1][1]
            if (next_start - w_e) >= pause_threshold:
                has_pause = True

        if is_last or has_punct or reached_limit or has_pause:
            chunks.append(current_chunk)
            current_chunk = []

    return chunks


def format_srt_timestamp(seconds):
    ms = int(round(seconds * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_content(word_spans):
    chunks = group_words_smartly(word_spans)
    lines = []
    for block_idx, chunk in enumerate(chunks, start=1):
        phrase = " ".join(w[0] for w in chunk)
        start_t = format_srt_timestamp(chunk[0][1])
        end_t = format_srt_timestamp(max(chunk[-1][2], chunk[0][1] + 0.25))
        lines.append(f"{block_idx}\n{start_t} --> {end_t}\n{phrase}\n")
    return "\n".join(lines)


def build_ae_jsx_content(word_spans):
    chunks = group_words_smartly(word_spans)
    blocks = []
    for chunk in chunks:
        phrase = " ".join(w[0] for w in chunk)
        start_s = round(chunk[0][1], 3)
        end_s = round(max(chunk[-1][2], chunk[0][1] + 0.25), 3)
        blocks.append({"text": phrase, "start": start_s, "end": end_s})

    json_payload = json.dumps(blocks, ensure_ascii=False)
    return f"""(function() {{
    var comp = app.project.activeItem;
    if (!comp || !(comp instanceof CompItem)) {{
        alert("Откройте нужную композицию в After Effects перед запуском скрипта!");
        return;
    }}
    app.beginUndoGroup("Import Subtitles - hafizoff dev");
    var subs = {json_payload};
    for (var i = 0; i < subs.length; i++) {{
        var item = subs[i];
        var layer = comp.layers.addText(item.text);
        layer.inPoint = item.start;
        layer.outPoint = item.end;
        var pos = layer.property("Transform").property("Position");
        pos.setValue([comp.width / 2, comp.height * 0.82]);
        var textProp = layer.property("Source Text");
        var textDoc = textProp.value;
        textDoc.fontSize = Math.round(comp.height * 0.045);
        textDoc.fillColor = [1, 1, 1];
        textDoc.justification = ParagraphJustification.CENTER_JUSTIFY;
        textProp.setValue(textDoc);
    }}
    app.endUndoGroup();
    alert("Готово! Добавлено " + subs.length + " слоёв субтитров.");
}})();
"""


def run_capcut_inject(raw_text, progress_cb, done_cb):
    try:
        progress_cb(5, "Поиск последнего проекта CapCut...")
        project_dir = find_latest_project()
        if not project_dir:
            raise FileNotFoundError("Проекты CapCut не найдены на компьютере!")

        proj_name = os.path.basename(project_dir)
        candidates = [
            os.path.join(project_dir, "draft_info.json"),
            os.path.join(project_dir, "draft_content.json"),
        ] + glob.glob(os.path.join(project_dir, "Timelines", "*", "draft_info.json")) + glob.glob(os.path.join(project_dir, "Timelines", "*", "draft_content.json"))
        existing = [p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 100]
        if not existing:
            raise FileNotFoundError(f"?? ?????? ???? ????????? ? ??????? CapCut ?{proj_name}?!")
        # ?? macOS ??????? ???????? draft_info.json (??? ????? ?????? ???????? ????)
        if sys.platform == "darwin":
            info_files = [p for p in existing if os.path.basename(p) == "draft_info.json"]
            draft_path = max(info_files or existing, key=os.path.getmtime)
        else:
            draft_path = max(existing, key=os.path.getmtime)

        shutil.copy2(draft_path, draft_path + ".bak")

        with open(draft_path, "r", encoding="utf-8") as f:
            draft = json.load(f)

        texts_list = draft.get("materials", {}).get("texts", [])
        native_sub = next(
            (t for t in texts_list if t.get("recognize_task_id") or t.get("type") == "subtitle" or t.get("recognize_type")),
            texts_list[0] if texts_list else None
        )
        if not native_sub:
            raise RuntimeError(
                f"В последнем проекте CapCut («{proj_name}») не найдена заготовка автосубтитров!\n\n"
                "Сделайте 2 шага:\n"
                "1. Откройте проект в CapCut → Текст → Автосубтитры → Создать (на любом языке).\n"
                "2. Выйдите в главное меню CapCut и нажмите кнопку снова."
            )

        audio_source = find_audio_from_capcut(draft)
        if not audio_source:
            raise FileNotFoundError(
                f"В проекте CapCut «{proj_name}» не найдено видео или аудио на дорожке!"
            )

        timeline_offset_us = 0
        for track in draft.get("tracks", []):
            if track.get("type") in ("audio", "video") and track.get("segments"):
                timeline_offset_us = track["segments"][0].get("target_timerange", {}).get("start", 0)
                break

        word_spans = align_text_to_audio(audio_source, raw_text, progress_cb)
        chunks = group_words_smartly(word_spans)

        progress_cb(92, "Запись нативных автосубтитров в проект CapCut...")
        template_text = copy.deepcopy(native_sub)
        task_id = template_text["recognize_task_id"]

        target_mat_ids = {t["id"] for t in texts_list if t.get("recognize_task_id") == task_id}
        text_track = next(
            (tr for tr in draft.get("tracks", []) if tr.get("type") == "text" and any(s.get("material_id") in target_mat_ids for s in tr.get("segments", []))),
            None
        ) or next((tr for tr in draft.get("tracks", []) if tr.get("type") == "text"), None)

        template_segment = copy.deepcopy(text_track["segments"][0])

        new_texts = [t for t in texts_list if t.get("recognize_task_id") != task_id]
        new_segments = []

        for idx_chunk, chunk in enumerate(chunks):
            phrase = " ".join(w[0] for w in chunk)
            start_us = int(chunk[0][1] * 1_000_000) + timeline_offset_us
            end_us = int(chunk[-1][2] * 1_000_000) + timeline_offset_us

            if idx_chunk < len(chunks) - 1:
                next_start_us = int(chunks[idx_chunk + 1][0][1] * 1_000_000) + timeline_offset_us
                if end_us > next_start_us:
                    end_us = next_start_us

            dur_us = max(end_us - start_us, 200_000)

            mat_id = str(uuid.uuid4()).upper()
            t_item = copy.deepcopy(template_text)
            t_item["id"] = mat_id
            t_item["type"] = "text"
            t_item["recognize_task_id"] = task_id
            t_item["recognize_text"] = phrase

            c_obj = json.loads(t_item["content"])
            c_obj["text"] = phrase
            for st in c_obj.get("styles", []):
                st["range"] = [0, len(phrase)]
            t_item["content"] = json.dumps(c_obj, ensure_ascii=False)

            w_starts, w_ends, w_texts = [], [], []
            for idx_w, (w_text, w_s, w_e) in enumerate(chunk):
                ws_ms = max(0, int((w_s * 1_000_000 + timeline_offset_us - start_us) / 1000))
                we_ms = max(ws_ms + 50, int((w_e * 1_000_000 + timeline_offset_us - start_us) / 1000))
                w_starts.append(ws_ms)
                w_ends.append(we_ms)
                w_texts.append(w_text)
                if idx_w < len(chunk) - 1:
                    w_starts.append(we_ms)
                    w_ends.append(we_ms)
                    w_texts.append(" ")

            words_obj = {"start_time": w_starts, "end_time": w_ends, "text": w_texts}
            t_item["words"] = words_obj
            if "current_words" in t_item:
                t_item["current_words"] = copy.deepcopy(words_obj)

            seg = copy.deepcopy(template_segment)
            seg["id"] = str(uuid.uuid4()).upper()
            seg["material_id"] = mat_id
            seg["target_timerange"] = {"start": start_us, "duration": dur_us}

            new_texts.append(t_item)
            new_segments.append(seg)

        draft["materials"]["texts"] = new_texts
        text_track["segments"] = new_segments

        targets_to_save = (
            [draft_path]
            + glob.glob(os.path.join(project_dir, "Timelines", "*", "draft_content.json"))
            + glob.glob(os.path.join(project_dir, "Timelines", "*", "draft_info.json"))
        )
        for target_file in set(targets_to_save):
            if os.path.exists(target_file):
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(draft, f, ensure_ascii=False)

        progress_cb(100, "Готово! 100%")
        done_cb(
            True,
            f"Готово! В проект CapCut «{proj_name}» добавлено {len(new_segments)} блоков субтитров.\n"
            "Откройте проект в CapCut и выберите любой анимированный шаблон!"
        )
    except Exception as e:
        done_cb(False, str(e))


def run_srt_export(audio_path, save_srt_path, raw_text, progress_cb, done_cb):
    try:
        word_spans = align_text_to_audio(audio_path, raw_text, progress_cb)
        progress_cb(92, "Формирование файла .SRT...")
        srt_text = build_srt_content(word_spans)
        with open(save_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_text)
        progress_cb(100, "Готово! 100%")
        done_cb(True, f"Файл субтитров успешно сохранён:\n{save_srt_path}")
    except Exception as e:
        done_cb(False, str(e))


def run_ae_export(audio_path, save_jsx_path, raw_text, progress_cb, done_cb):
    try:
        word_spans = align_text_to_audio(audio_path, raw_text, progress_cb)
        progress_cb(92, "Генерация скрипта для After Effects (.JSX)...")
        jsx_text = build_ae_jsx_content(word_spans)
        with open(save_jsx_path, "w", encoding="utf-8") as f:
            f.write(jsx_text)
        progress_cb(100, "Готово! 100%")
        done_cb(
            True,
            f"Скрипт для After Effects сохранён:\n{save_jsx_path}\n\n"
            "В After Effects откройте композицию и выберите:\nFile → Scripts → Run Script File..."
        )
    except Exception as e:
        done_cb(False, str(e))


# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС (DARK UI) ====================
root = tk.Tk()
root.title("Программа для генерации субтитров — hafizoff dev")
root.geometry("740x625")
root.configure(bg="#17212B")

style = ttk.Style()
style.theme_use("clam")
style.configure(
    "Custom.Horizontal.TProgressbar",
    troughcolor="#242F3D",
    background="#4CD964",
    bordercolor="#17212B",
    lightcolor="#6AB2F2",
    darkcolor="#2B5278",
)

top_frame = tk.Frame(root, bg="#17212B")
top_frame.pack(fill="x", padx=18, pady=(14, 6))

lbl_title = tk.Label(
    top_frame, text="Программа для генерации субтитров",
    bg="#17212B", fg="#FFFFFF", font=(UI_FONT, 13, "bold")
)
lbl_title.pack(side="left")

dev_frame = tk.Frame(top_frame, bg="#17212B")
dev_frame.pack(side="right")

tk.Label(dev_frame, text="Разработчик: ", bg="#17212B", fg="#A8B5C2", font=(UI_FONT, 9)).pack(side="left")

lbl_dev = tk.Label(
    dev_frame, text="hafizoff dev (Instagram)",
    bg="#17212B", fg="#6AB2F2", font=(UI_FONT, 9, "bold underline"), cursor="hand2"
)
lbl_dev.pack(side="left")
lbl_dev.bind("<Button-1>", lambda e: webbrowser.open(INSTAGRAM_URL))

prompt_frame = tk.LabelFrame(
    root, text=" Шаг 1. Промпт для Gemini (чтобы получить чистый текст из видео/аудио) ",
    bg="#17212B", fg="#6AB2F2", font=(UI_FONT, 9, "bold"), padx=10, pady=8
)
prompt_frame.pack(fill="x", padx=18, pady=(6, 10))

prompt_lbl = tk.Label(
    prompt_frame, text=GEMINI_PROMPT, bg="#242F3D", fg="#D5DFE8",
    font=(UI_FONT, 9), justify="left", wraplength=540, padx=8, pady=6
)
prompt_lbl.pack(side="left", fill="x", expand=True)

def copy_prompt():
    root.clipboard_clear()
    root.clipboard_append(GEMINI_PROMPT)
    status_lbl.config(text="Промпт скопирован! Отправьте его в Gemini вместе с вашим видео/аудио.", fg="#4CD964")

btn_copy_prompt = tk.Button(
    prompt_frame, text="📋 Копировать\nпромпт", command=copy_prompt,
    bg="#2B5278", fg="#FFFFFF", activebackground="#3A6D9E", activeforeground="#FFFFFF",
    relief="flat", padx=12, pady=8, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_copy_prompt.pack(side="right", padx=(10, 0))

text_frame = tk.LabelFrame(
    root, text=" Шаг 2. Вставьте распознанный текст от Gemini сюда ",
    bg="#17212B", fg="#6AB2F2", font=(UI_FONT, 9, "bold"), padx=10, pady=8
)
text_frame.pack(fill="both", expand=True, padx=18, pady=(0, 10))

text_box = tk.Text(
    text_frame, height=11, bg="#242F3D", fg="#F5F5F5", insertbackground="#FFFFFF",
    font=(UI_FONT, 11), relief="flat", padx=10, pady=8, wrap="word"
)
text_box.pack(fill="both", expand=True)


def handle_hotkeys(event):
    # Поддержка Ctrl (Windows) и Command (macOS) при любой раскладке клавиатуры
    if (event.state & 0x4) or (event.state & 0x8) or (event.state & 0x10):
        ks = event.keysym.lower()
        kc = event.keycode
        if kc == 65 or ks in ("a", "cyrillic_ef"):
            text_box.tag_add("sel", "1.0", "end-1c")
            text_box.mark_set("insert", "end-1c")
            return "break"
        elif kc == 86 or ks in ("v", "cyrillic_em"):
            text_box.event_generate("<<Paste>>")
            return "break"
        elif kc == 67 or ks in ("c", "cyrillic_es"):
            text_box.event_generate("<<Copy>>")
            return "break"
        elif kc == 88 or ks in ("x", "cyrillic_che"):
            text_box.event_generate("<<Cut>>")
            return "break"


text_box.bind("<KeyPress>", handle_hotkeys)

controls_frame = tk.Frame(root, bg="#17212B")
controls_frame.pack(fill="x", padx=18, pady=(0, 8))

def clear_text():
    text_box.delete("1.0", "end")
    status_lbl.config(text="Поле очищено.", fg="#7F91A4")

def paste_clip():
    try:
        text_box.delete("1.0", "end")
        text_box.insert("1.0", root.clipboard_get())
        status_lbl.config(text="Текст вставлен из буфера обмена.", fg="#6AB2F2")
    except tk.TclError:
        pass

btn_clear = tk.Button(
    controls_frame, text="🗑 Очистить", command=clear_text,
    bg="#242F3D", fg="#FF7A7A", activebackground="#3A2830", activeforeground="#FFFFFF",
    relief="flat", padx=10, pady=6, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_clear.pack(side="left", padx=(0, 6))

btn_paste = tk.Button(
    controls_frame, text="📥 Вставить", command=paste_clip,
    bg="#242F3D", fg="#6AB2F2", activebackground="#2B5278", activeforeground="#FFFFFF",
    relief="flat", padx=10, pady=6, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_paste.pack(side="left")

btn_srt = tk.Button(
    controls_frame, text="💾 .SRT",
    bg="#242F3D", fg="#E0E6ED", activebackground="#344457", activeforeground="#FFFFFF",
    relief="flat", padx=10, pady=6, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_srt.pack(side="right", padx=(6, 0))

btn_ae = tk.Button(
    controls_frame, text="🎬 After Effects (.JSX)",
    bg="#242F3D", fg="#B39DDB", activebackground="#344457", activeforeground="#FFFFFF",
    relief="flat", padx=10, pady=6, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_ae.pack(side="right", padx=(6, 0))

btn_capcut = tk.Button(
    controls_frame, text="▶ Отправить в CapCut",
    bg="#2B5278", fg="#FFFFFF", activebackground="#3A6D9E", activeforeground="#FFFFFF",
    relief="flat", padx=12, pady=6, font=(UI_FONT, 9, "bold"), cursor="hand2"
)
btn_capcut.pack(side="right")

progress_var = tk.IntVar(value=0)
progress_bar = ttk.Progressbar(
    root, style="Custom.Horizontal.TProgressbar", mode="determinate",
    maximum=100, variable=progress_var, length=700
)

status_lbl = tk.Label(
    root, text="Готово к работе.",
    bg="#17212B", fg="#7F91A4", font=(UI_FONT, 9, "bold")
)
status_lbl.pack(pady=(2, 12))


def start_loading(active_btn=None):
    btn_capcut.config(state="disabled")
    btn_srt.config(state="disabled")
    btn_ae.config(state="disabled")
    btn_clear.config(state="disabled")
    btn_paste.config(state="disabled")
    progress_var.set(0)
    progress_bar.pack(fill="x", padx=18, pady=(0, 4), before=status_lbl)
    if active_btn == "capcut":
        btn_capcut.config(text="⏳ Загрузка 0%...")
    elif active_btn == "ae":
        btn_ae.config(text="⏳ Создание 0%...")
    elif active_btn == "srt":
        btn_srt.config(text="⏳ Создание 0%...")


def stop_loading():
    progress_bar.pack_forget()
    progress_var.set(0)
    btn_capcut.config(state="normal", text="▶ Отправить в CapCut")
    btn_srt.config(state="normal", text="💾 .SRT")
    btn_ae.config(state="normal", text="🎬 After Effects (.JSX)")
    btn_clear.config(state="normal")
    btn_paste.config(state="normal")


def update_progress(percent, msg):
    def ui_update():
        progress_var.set(percent)
        status_lbl.config(text=f"[{percent}%] {msg}", fg="#6AB2F2")
        if btn_capcut["state"] == "disabled" and "Загрузка" in btn_capcut["text"]:
            btn_capcut.config(text=f"⏳ Загрузка {percent}%...")
        elif btn_ae["state"] == "disabled" and "Создание" in btn_ae["text"]:
            btn_ae.config(text=f"⏳ Создание {percent}%...")
        elif btn_srt["state"] == "disabled" and "Создание" in btn_srt["text"]:
            btn_srt.config(text=f"⏳ Создание {percent}%...")
    root.after(0, ui_update)


def on_done(ok, msg):
    def finish():
        stop_loading()
        if ok:
            status_lbl.config(text="Готово! [100%]", fg="#4CD964")
            messagebox.showinfo("Успешно", msg)
        else:
            status_lbl.config(text="Ошибка выполнения", fg="#FF5E5E")
            messagebox.showerror("Ошибка", msg)
    root.after(0, finish)


def start_capcut():
    raw = text_box.get("1.0", "end").strip()
    if not raw:
        messagebox.showwarning("Пустой текст", "Сначала вставьте текст от Gemini!")
        return
    start_loading("capcut")
    threading.Thread(
        target=run_capcut_inject, args=(raw, update_progress, on_done), daemon=True
    ).start()


def start_srt():
    raw = text_box.get("1.0", "end").strip()
    if not raw:
        messagebox.showwarning("Пустой текст", "Сначала вставьте текст от Gemini!")
        return
    default_audio = None
    p = find_latest_project()
    if p:
        for fn in ("draft_content.json", "draft_info.json"):
            fp = os.path.join(p, fn)
            if os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    default_audio = find_audio_from_capcut(json.load(f))
                break

    if default_audio and messagebox.askyesno(
        "Источник звука",
        f"Использовать аудио/видео из текущего проекта CapCut?\n({os.path.basename(default_audio)})\n\nНажмите «Нет», чтобы выбрать другой файл вручную."
    ):
        audio_path = default_audio
    else:
        audio_path = filedialog.askopenfilename(
            title="Выберите видео или аудиофайл",
            filetypes=[("Медиафайлы", "*.mp4 *.mov *.mkv *.avi *.aac *.mp3 *.wav *.m4a *.ogg"), ("Все файлы", "*.*")]
        )
        if not audio_path:
            return

    save_path = filedialog.asksaveasfilename(
        title="Сохранить субтитры как .SRT",
        defaultextension=".srt",
        initialfile=os.path.splitext(os.path.basename(audio_path))[0] + ".srt",
        filetypes=[("SubRip Subtitles", "*.srt")]
    )
    if not save_path:
        return

    start_loading("srt")
    threading.Thread(
        target=run_srt_export, args=(audio_path, save_path, raw, update_progress, on_done), daemon=True
    ).start()


def start_ae():
    raw = text_box.get("1.0", "end").strip()
    if not raw:
        messagebox.showwarning("Пустой текст", "Сначала вставьте текст от Gemini!")
        return
    audio_path = filedialog.askopenfilename(
        title="Выберите видео или аудиофайл для After Effects",
        filetypes=[("Медиафайлы", "*.mp4 *.mov *.mkv *.avi *.aac *.mp3 *.wav *.m4a *.ogg"), ("Все файлы", "*.*")]
    )
    if not audio_path:
        return
    save_path = filedialog.asksaveasfilename(
        title="Сохранить скрипт для After Effects (.JSX)",
        defaultextension=".jsx",
        initialfile=os.path.splitext(os.path.basename(audio_path))[0] + "_AE.jsx",
        filetypes=[("After Effects Script", "*.jsx")]
    )
    if not save_path:
        return
    start_loading("ae")
    threading.Thread(
        target=run_ae_export, args=(audio_path, save_path, raw, update_progress, on_done), daemon=True
    ).start()


btn_capcut.config(command=start_capcut)
btn_srt.config(command=start_srt)
btn_ae.config(command=start_ae)

root.mainloop()