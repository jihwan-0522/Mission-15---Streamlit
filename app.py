import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("이미지 분류 웹 앱")
st.caption("이미지 업로드 → AI 추론 → Top-5 시각화")

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

clf = load_model()

uploaded = st.file_uploader("이미지를 업로드 해줘", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("👆 위에서 이미지를 업로드하면 예측 결과가 나와.")
else:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    with st.spinner("모델 추론 중..."):
        preds = clf(image, top_k=5)
    # Top-1 (가장 확률 높은 결과)
    top1 = preds[0]
    top1_label = top1["label"]
    top1_score = float(top1["score"])

    st.subheader("✅ 1등 예측")
    st.metric(label=top1_label, value=f"{top1_score*100:.1f}%")

    st.divider()



    st.subheader("예측 결과 (Top-5)")
    for i, p in enumerate(preds, 1):
        st.write(f"{i}. **{p['label']}** — {p['score']:.3f}")

    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]

    fig = plt.figure()
    plt.bar(labels, scores)
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Top-5 Confidence")
    plt.tight_layout()
    st.pyplot(fig)
