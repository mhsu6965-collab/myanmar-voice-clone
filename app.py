import os
import re
import numpy as np
import soundfile as sf
import streamlit as st
from gradio_client import Client, handle_file

st.set_page_config(page_title="Myanmar Voice Clone 24/7", page_icon="🎙️", layout="wide")

st.title("🎙️ Myanmar Voice Clone App (24/7 Free Public GPU)")
st.caption("Card ချိတ်ရန်မလို · Colab ဖွင့်ရန်မလို · Public ZeroGPU Server များနှင့် တိုက်ရိုက်ချိတ်ဆက်ထားသည်")

# လက်ရှိ Hugging Face ပေါ်တွင် ZeroGPU ဖြင့် အခမဲ့ Run နေသော Myanmar VoxCPM2 Public Spaces များ
PUBLIC_SERVERS = [
    "hub10/voxcpm2-myanmar-clone-test",
    "Hub11x/voxcpm2-myanmar-clone-test",
    "hub6x/voxcpm2-myanmar-clone-test",
    "NayWin27/voxcpm2-clone-space",
    "openbmb/VoxCPM-Demo"
]

st.sidebar.header("🌐 Public GPU Server ရွေးချယ်ရန်")
selected_server = st.sidebar.selectbox(
    "Server တစ်ခု လူပြည့်နေပါက အခြားတစ်ခု ပြောင်းသုံးနိုင်သည် -",
    PUBLIC_SERVERS,
    index=0
)
custom_server = st.sidebar.text_input("သို့မဟုတ် အခြား Hugging Face Space အမည်ထည့်ရန် (Optional)", "")
active_space = custom_server.strip() if custom_server.strip() else selected_server

hf_token = st.sidebar.text_input(
    "🔑 HF Token (Optional - Free Quota ပိုရစေရန်)",
    type="password",
    help="Card မလိုသော ရိုးရိုး Hugging Face အကောင့်၏ Read Token ထည့်ပေးပါက ZeroGPU အချိန်ပိုရပါမည်။"
)

def prepare_ref_audio(uploaded_file, max_sec=12.0):
    temp_in = "temp_uploaded_ref.wav"
    with open(temp_in, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data, sr = sf.read(temp_in)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    max_samples = int(max_sec * sr)
    if len(data) > max_samples:
        data = data[:max_samples]
    clean_path = "cleaned_ref_audio.wav"
    sf.write(clean_path, data, sr)
    return clean_path

def split_myanmar_sentences(text, max_chars=180):
    raw_parts = re.split(r'(?<=[။!?])\s*|\n+', text.strip())
    chunks, current = [], ""
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip() if current else part
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks if chunks else [text.strip()]

def extract_audio_path(res):
    if isinstance(res, str) and os.path.exists(res):
        return res
    if isinstance(res, dict) and "path" in res:
        return res["path"]
    if isinstance(res, (list, tuple)):
        for item in res:
            if isinstance(item, str) and os.path.exists(item):
                return item
            if isinstance(item, dict) and "path" in item:
                return item["path"]
    return None

def call_public_voxcpm(client, api_info, chunk_text, ref_path, transcript, style_text, cfg, steps, max_steps, seed):
    endpoints = api_info.get("named_endpoints", {})

    # 1. Myanmar Clone Test Space ပုံစံအတိုင်း လှမ်းခေါ်ခြင်း
    for ep_name, ep_data in endpoints.items():
        params = ep_data.get("parameters", [])
        param_names = [p.get("parameter_name", "").lower() for p in params]

        # Long script endpoint မဟုတ်သော အဓိက Clone endpoint ကို ရှာခြင်း
        if any("audio" in p or "ref" in p or "voice" in p for p in param_names) or len(params) >= 5:
            kwargs = {}
            for p in params:
                pname = p.get("parameter_name")
                plower = pname.lower()
                ptype = str(p.get("python_type", {}).get("type", "")).lower()

                if "audio" in plower or "ref" in plower and ("file" in ptype or "filepath" in ptype or "dict" in ptype):
                    kwargs[pname] = handle_file(ref_path)
                elif "transcript" in plower or "prompt_text" in plower:
                    kwargs[pname] = transcript
                elif "style" in plower or "instruct" in plower:
                    kwargs[pname] = style_text
                elif "text" in plower or "script" in plower:
                    kwargs[pname] = chunk_text
                elif "cfg" in plower:
                    kwargs[pname] = float(cfg)
                elif "max" in plower and "step" in plower:
                    kwargs[pname] = int(max_steps)
                elif "step" in plower or "timestep" in plower:
                    kwargs[pname] = int(steps)
                elif "seed" in plower and "bool" not in ptype:
                    kwargs[pname] = int(seed)
                elif p.get("parameter_has_default"):
                    kwargs[pname] = p.get("parameter_default")

            try:
                res = client.predict(**kwargs, api_name=ep_name)
                audio_out = extract_audio_path(res)
                if audio_out:
                    return audio_out
            except Exception:
                continue

    # 2. Fallback: ပထမဆုံး Endpoint သို့ Positional Arguments ဖြင့် ပို့ခြင်း
    first_ep = list(endpoints.keys())[0] if endpoints else 0
    try:
        res = client.predict(
            chunk_text,
            handle_file(ref_path),
            transcript,
            style_text,
            float(cfg),
            int(steps),
            int(max_steps),
            int(seed),
            api_name=first_ep
        )
        return extract_audio_path(res)
    except Exception as e:
        raise RuntimeError(f"Server မှ တုံ့ပြန်မှု အမှားရှိနေပါသည်: {e}")

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("1. Reference Audio (၅ မှ ၁၂ စက္ကန့်ခန့် နမူနာအသံဖိုင် တင်ရန်)", type=["wav", "mp3", "m4a", "flac"])
    if ref_file:
        st.audio(ref_file)

    prompt_text = st.text_area(
        "2. Exact Reference Transcript (နမူနာအသံဖိုင်ထဲက ပြောထားသော စာသားအတိအကျ - အသံပိုတူစေရန်)",
        height=75,
        placeholder="ဥပမာ - ခင်ဗျားတို့ဘဝမှာ ဘီယာတွေအဝသောက်ပြီး သေရလိမ့်မယ်လို့ တွေးမိကြဖူးသလား"
    )

    style_text = st.text_input(
        "3. Optional Voice Style (အသံပုံစံ - မထည့်လည်းရသည်)",
        placeholder="ဥပမာ - calm storytelling tone"
    )

    target_text = st.text_area(
        "4. Text to Speak (အသစ်ထုတ်ပေးစေလိုသော မြန်မာစာပိုဒ် - အရှည်ကြီးထည့်နိုင်သည်)",
        height=160,
        placeholder="အသံပွားပြီး ပြောစေချင်သော မြန်မာစာသားများကို ဒီနေရာမှာ ရိုက်ထည့်ပါ..."
    )

    c1, c2, c3 = st.columns(3)
    cfg_val = c1.slider("CFG Scale", 1.0, 3.5, 2.0, 0.1)
    steps_val = c2.slider("Inference Steps", 4, 20, 10, 1)
    seed_val = c3.number_input("Seed", value=42, step=1)
    max_audio_steps = 1024

    gen_btn = st.button("🚀 Generate Myanmar Voice Clone", type="primary", use_container_width=True)

with col2:
    st.subheader("🎧 ထွက်ရှိလာသော အသံဖိုင် (48kHz)")
    if gen_btn:
        if not ref_file or not target_text.strip():
            st.error("ကျေးဇူးပြု၍ Reference Audio နှင့် ထုတ်မည့် မြန်မာစာသားကို ထည့်သွင်းပေးပါ။")
        else:
            try:
                with st.spinner(f"⏳ `{active_space}` Public GPU Server သို့ ချိတ်ဆက်နေပါသည်..."):
                    clean_ref = prepare_ref_audio(ref_file, max_sec=12.0)
                    chunks = split_myanmar_sentences(target_text, max_chars=170)

                    client = Client(active_space, hf_token=hf_token.strip() if hf_token.strip() else None)
                    api_info = client.view_api(return_format="dict")

                progress = st.progress(0, text="အသံစတင်ထုတ်ပေးနေပါသည်...")
                all_wavs = []
                out_sr = 48000

                for idx, chunk in enumerate(chunks):
                    progress.progress((idx) / len(chunks), text=f"⏳ စာကြောင်း ({idx+1}/{len(chunks)}) ကို အသံထုတ်နေပါသည်...")
                    part_path = call_public_voxcpm(
                        client=client,
                        api_info=api_info,
                        chunk_text=chunk,
                        ref_path=clean_ref,
                        transcript=prompt_text.strip(),
                        style_text=style_text.strip(),
                        cfg=cfg_val,
                        steps=steps_val,
                        max_steps=max_audio_steps,
                        seed=seed_val
                    )
                    if not part_path:
                        raise RuntimeError("Server မှ အသံဖိုင် ပြန်မပို့ပေးပါ။ ဘေးဘက် Menu မှ အခြား Server တစ်ခု ပြောင်းစမ်းကြည့်ပါ။")

                    wav_data, sr = sf.read(part_path)
                    if wav_data.ndim > 1:
                        wav_data = np.mean(wav_data, axis=1)
                    out_sr = sr
                    all_wavs.append(wav_data.astype(np.float32))
                    if idx < len(chunks) - 1:
                        all_wavs.append(np.zeros(int(out_sr * 0.25), dtype=np.float32))

                progress.progress(1.0, text="✅ အသံဖိုင်များအားလုံး ပေါင်းစည်းပြီးပါပြီ!")
                final_wav = np.concatenate(all_wavs)
                final_out_path = "final_myanmar_clone.wav"
                sf.write(final_out_path, final_wav, out_sr)

                st.success(f"✅ အောင်မြင်ပါသည်! စာကြောင်းရေ ({len(chunks)}) ပိုင်းခွဲ၍ အသံထုတ်ပြီးပါပြီ။")
                st.audio(final_out_path)
                with open(final_out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Cloned Audio (.wav)",
                        f,
                        file_name="myanmar_voice_clone.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(
                    f"⚠️ လက်ရှိ Public Server (`{active_space}`) တွင် လူသုံးများနေပါသည် သို့မဟုတ် Quota ပြည့်နေပါသည်။\n\n"
                    f"**ဖြေရှင်းနည်း:** ဘယ်ဘက်အပေါ်ထောင့် `>` ကိုနှိပ်ပြီး **Public GPU Server ရွေးချယ်ရန်** တွင် အခြား Server တစ်ခု ပြောင်းရွေးပါ။\n\n"
                    f"အသေးစိတ်: `{e}`"
                )
