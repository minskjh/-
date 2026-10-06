import os
from io import BytesIO
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageOps
import google.generativeai as genai

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="인스타 포스팅 생성기", page_icon="📸", layout="centered")
st.title("📸 인스타 피드 제작기")

# --- 기본 설정값 ---
DEFAULT_FONT_PATH = "font.ttf"
FALLBACK_FONTS = [
    "font.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",  # Windows 맑은 고딕 Bold
    "C:/Windows/Fonts/malgun.ttf",    # Windows 맑은 고딕
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",  # Linux (Ubuntu/Debian)
]


def get_api_key() -> str:
    """환경변수, Streamlit Secrets, 또는 사이드바 입력에서 API 키를 가져옵니다."""
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"]:
        return os.environ["GEMINI_API_KEY"]
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.session_state.get("user_api_key", "")


def load_font(size: int, custom_font_bytes: bytes | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """지정된 크기의 폰트를 로드하며, 파일이 없을 경우 시스템 한글 폰트로 자동 대체합니다."""
    if custom_font_bytes:
        try:
            return ImageFont.truetype(BytesIO(custom_font_bytes), size)
        except Exception:
            pass

    for path in FALLBACK_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    return ImageFont.load_default()


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    initial_size: int,
    max_width: int,
    min_size: int = 24,
    custom_font_bytes: bytes | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """텍스트가 이미지 가로 폭(max_width)을 넘지 않도록 폰트 크기를 자동 조절합니다."""
    size = initial_size
    while size >= min_size:
        font = load_font(size, custom_font_bytes)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            return font
        size -= 4
    return load_font(min_size, custom_font_bytes)


# 1. 네이버 블로그 크롤러 (예외 처리 및 다양한 URL 포맷 대응)
@st.cache_data(show_spinner=False, ttl=600)
def extract_naver_blog_content(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
        )
    }

    try:
        # 단축 URL(naver.me 등) 리다이렉트 추적 후 모바일 URL로 변환
        session = requests.Session()
        res = session.get(url, headers=headers, timeout=10, allow_redirects=True)
        final_url = res.url

        if "blog.naver.com" in final_url and "m.blog.naver.com" not in final_url:
            mobile_url = final_url.replace("blog.naver.com", "m.blog.naver.com")
            res = session.get(mobile_url, headers=headers, timeout=10)

        res.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"블로그 페이지에 접속할 수 없습니다: {e}") from e

    soup = BeautifulSoup(res.text, "html.parser")
    content_div = (
        soup.select_one("div.se-main-container")
        or soup.select_one("#viewTypeSelector")
        or soup.select_one("div#postViewArea")
        or soup.select_one("div.se_component_wrap")
    )

    if not content_div:
        return ""

    # 불필요한 공백 라인 정리
    raw_text = content_div.get_text(separator="\n", strip=True)
    cleaned_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)


# 2. 인스타 캡션 생성 (엄격한 팩트/가격 규칙)
def generate_caption(
    api_key: str,
    blog_text: str,
    sub_title: str,
    address: str,
    model_name: str = "gemini-3.8-flash",
) -> str:
    genai.configure(api_key=api_key)

    prompt = f"""
당신은 인스타그램 맛집 전문 에디터입니다.
제공된 [블로그 본문]을 바탕으로 인스타그램 피드 캡션을 작성하세요.

[원칙]
1. 없는 정보나 임의의 가격/예산 추정 금지.
2. '프로필 링크 확인', '저장/공유' 등 상투적 유도문 배제.
3. [가격 정보]에는 본문에 명시된 메뉴와 가격만 기재 (본문에 가격이 없으면 '매장 문의 또는 메뉴판 참고'로 표기).
4. 문장 사이 줄바꿈과 이모지 활용.

[출력 양식]
[첫 줄 후킹 헤드카피]

[핵심 맛/특징 포인트 요약 2~3단락]

💰 가격 정보
• [메뉴명] : [가격]원
(※ 블로그 본문 기준 메뉴 및 가격)

─
📍 {sub_title}
• {address}

#[키워드 기반 해시태그 10~15개]

[블로그 본문]:
{blog_text[:4000]}
"""
    # 선택한 모델 우선 시도 후, 404(지원 종료 등) 발생 시 사용 가능한 최신 Flash 모델 자동 탐색
    candidate_models = [model_name, "gemini-3.8-flash", "gemini-flash-latest"]
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                short_name = m.name.replace("models/", "")
                if "flash" in short_name and short_name not in candidate_models:
                    candidate_models.append(short_name)
    except Exception:
        pass

    last_err = None
    for candidate in dict.fromkeys(candidate_models):
        try:
            model = genai.GenerativeModel(candidate)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            if "404" in str(e) or "NotFound" in str(e) or "not found" in str(e).lower():
                continue
            raise

    if last_err:
        raise last_err
    return "캡션 생성 결과를 가져오지 못했습니다."


def draw_location_pin(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float) -> None:
    """어떤 폰트에서도 깨지지 않도록 깔끔한 벡터 위치 핀(📍) 아이콘을 그립니다."""
    r = size * 0.34
    head_cy = cy - size * 0.16
    tip_y = cy + size * 0.44

    # 외곽선(검정)
    pad = max(3, int(size * 0.08))
    draw.polygon(
        [
            (cx - r - pad * 0.7, head_cy + r * 0.3),
            (cx + r + pad * 0.7, head_cy + r * 0.3),
            (cx, tip_y + pad),
        ],
        fill="black",
    )
    draw.ellipse(
        (cx - r - pad, head_cy - r - pad, cx + r + pad, head_cy + r + pad),
        fill="black",
    )

    # 핀 본체(레드/코랄)
    pin_color = (255, 68, 88, 255)
    draw.polygon(
        [
            (cx - r * 0.92, head_cy + r * 0.35),
            (cx + r * 0.92, head_cy + r * 0.35),
            (cx, tip_y),
        ],
        fill=pin_color,
    )
    draw.ellipse(
        (cx - r, head_cy - r, cx + r, head_cy + r),
        fill=pin_color,
    )

    # 내부 흰색 원
    inner_r = r * 0.42
    draw.ellipse(
        (cx - inner_r, head_cy - inner_r, cx + inner_r, head_cy + inner_r),
        fill="white",
    )


def draw_sparkle(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float) -> None:
    """깔끔한 벡터 반짝이(✨) 아이콘을 그립니다."""
    def _star(x: float, y: float, r_outer: float, r_inner: float, fill_color, outline_w: int = 3):
        pts = [
            (x, y - r_outer),
            (x + r_inner, y - r_inner),
            (x + r_outer, y),
            (x + r_inner, y + r_inner),
            (x, y + r_outer),
            (x - r_inner, y + r_inner),
            (x - r_outer, y),
            (x - r_inner, y - r_inner),
        ]
        draw.polygon(pts, fill=fill_color, outline="black", width=outline_w)

    # 메인 큰 별 + 우측 상단 작은 별
    _star(cx - size * 0.08, cy + size * 0.06, size * 0.38, size * 0.12, (255, 215, 64, 255), outline_w=max(2, int(size * 0.05)))
    _star(cx + size * 0.26, cy - size * 0.24, size * 0.20, size * 0.07, (255, 235, 130, 255), outline_w=max(2, int(size * 0.04)))


# 3. 4:5 이미지 합성 (EXIF 회전 보정 및 텍스트 자동 맞춤 포함)
def make_cover_image(
    img_file,
    t1: str,
    t2: str,
    addr: str,
    custom_font_bytes: bytes | None = None,
) -> bytes:
    # 스마트폰 사진의 EXIF 회전 정보 보정
    raw_img = Image.open(img_file)
    raw_img = ImageOps.exif_transpose(raw_img).convert("RGBA")

    target_w, target_h = 1080, 1350
    scale = max(target_w / raw_img.width, target_h / raw_img.height)
    new_w, new_h = int(raw_img.width * scale), int(raw_img.height * scale)

    img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img = img.crop(
        (
            (new_w - target_w) // 2,
            (new_h - target_h) // 2,
            (new_w + target_w) // 2,
            (new_h + target_h) // 2,
        )
    )

    # 상/하단 그라데이션 오버레이
    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    for y in range(450):
        draw_ov.line([(0, y), (target_w, y)], fill=(0, 0, 0, int(140 * (1 - y / 450))))
    for y in range(target_h - 250, target_h):
        draw_ov.line(
            [(0, y), (target_w, y)],
            fill=(0, 0, 0, int(160 * ((y - (target_h - 250)) / 250))),
        )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    max_text_width = target_w - 100  # 좌우 여백 50px씩 확보

    # 내용 1 (상단 대형 타이틀)
    f_large = fit_font(draw, t1, initial_size=110, max_width=max_text_width, custom_font_bytes=custom_font_bytes)
    b1 = draw.textbbox((0, 0), t1, font=f_large)
    draw.text(
        ((target_w - (b1[2] - b1[0])) // 2, 100),
        t1,
        font=f_large,
        fill="white",
        stroke_width=10,
        stroke_fill="black",
    )

    # 내용 2 (서브 타이틀 + 좌측 핀 아이콘 + 우측 반짝이 아이콘)
    icon_mid = 76
    gap_mid = 18
    f_mid = fit_font(
        draw,
        t2,
        initial_size=80,
        max_width=max_text_width - (icon_mid * 2 + gap_mid * 2),
        custom_font_bytes=custom_font_bytes,
    )
    b2 = draw.textbbox((0, 0), t2, font=f_mid)
    t2_w = b2[2] - b2[0]
    t2_h = b2[3] - b2[1]
    total_mid_w = icon_mid + gap_mid + t2_w + gap_mid + icon_mid
    start_mid_x = (target_w - total_mid_w) // 2
    y_mid = 250
    center_mid_y = y_mid + b2[1] + t2_h / 2

    draw_location_pin(draw, start_mid_x + icon_mid / 2, center_mid_y, icon_mid)
    draw.text(
        (start_mid_x + icon_mid + gap_mid - b2[0], y_mid),
        t2,
        font=f_mid,
        fill="white",
        stroke_width=8,
        stroke_fill="black",
    )
    draw_sparkle(draw, start_mid_x + icon_mid + gap_mid + t2_w + gap_mid + icon_mid / 2, center_mid_y, icon_mid)

    # 주소 (우측 하단 + 좌측 핀 아이콘)
    icon_small = 42
    gap_small = 10
    f_small = fit_font(
        draw,
        addr,
        initial_size=42,
        max_width=max_text_width - (icon_small + gap_small),
        custom_font_bytes=custom_font_bytes,
    )
    ba = draw.textbbox((0, 0), addr, font=f_small)
    addr_w = ba[2] - ba[0]
    addr_h = ba[3] - ba[1]
    y_addr = target_h - 110
    addr_x = target_w - addr_w - 50
    center_addr_y = y_addr + ba[1] + addr_h / 2

    draw_location_pin(draw, addr_x - gap_small - icon_small / 2, center_addr_y, icon_small)
    draw.text(
        (addr_x - ba[0], y_addr),
        addr,
        font=f_small,
        fill="white",
        stroke_width=5,
        stroke_fill="black",
    )

    out = BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=95)
    return out.getvalue()


# --- 사이드바: API 키 및 폰트 설정 ---
with st.sidebar:
    st.header("⚙️ 환경 설정")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="환경변수(GEMINI_API_KEY)나 secrets.toml이 없을 경우 여기에 입력하세요.",
        )
        if api_key:
            st.session_state["user_api_key"] = api_key
    else:
        st.success("API 키가 설정되어 있습니다.")

    model_choice = st.selectbox(
        "Gemini 모델 선택",
        options=["gemini-3.8-flash", "gemini-3.8-pro", "gemini-flash-latest"],
        index=0,
    )

    custom_font_file = st.file_uploader(
        "커스텀 폰트 (선택, .ttf/.otf)",
        type=["ttf", "otf"],
        help="업로드하지 않으면 폴더 내 font.ttf 또는 시스템 기본 한글 폰트를 사용합니다.",
    )
    custom_font_bytes = custom_font_file.read() if custom_font_file else None


# --- 웹 화면 폼 구성 ---
uploaded_img = st.file_uploader("1. 음식 사진 선택", type=["jpg", "jpeg", "png", "webp"])
blog_url = st.text_input("2. 네이버 블로그 링크", placeholder="https://blog.naver.com/...")
text1 = st.text_input("3. 내용1 (대형 타이틀)", placeholder="일직동 양고기 맛집")
text2 = st.text_input("4. 내용2 (매장 이름)", placeholder="일일양 광명역점")
address = st.text_input("5. 매장 주소", placeholder="경기 광명시 일직로12번길 21 108호")

if st.button("🚀 인스타용 자료 생성하기", use_container_width=True):
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        st.error("좌측 사이드바(모바일은 좌측 상단 `>` 버튼)에서 유효한 Gemini API Key를 입력해 주세요.")
    elif not (uploaded_img and blog_url and text1 and text2 and address):
        st.warning("모든 항목을 입력해 주세요.")
    else:
        try:
            with st.spinner("블로그 본문 분석 및 이미지 제작 중..."):
                # 1. 이미지 생성
                final_img_bytes = make_cover_image(
                    uploaded_img,
                    text1.strip(),
                    text2.strip(),
                    address.strip(),
                    custom_font_bytes=custom_font_bytes,
                )

                # 2. 블로그 크롤링 및 캡션 생성
                blog_txt = extract_naver_blog_content(blog_url)
                if not blog_txt:
                    st.warning("블로그 본문을 추출하지 못했습니다. 링크가 올바른지 확인해 주세요.")
                    caption = ""
                else:
                    caption = generate_caption(
                        api_key=api_key,
                        blog_text=blog_txt,
                        sub_title=text2.strip(),
                        address=address.strip(),
                        model_name=model_choice,
                    )

            # 세션에 결과 저장 (다운로드 버튼 클릭 시 화면 초기화 방지)
            st.session_state["result_img"] = final_img_bytes
            st.session_state["result_caption"] = caption
            st.session_state["result_title"] = text2.strip()

        except Exception as e:
            st.error(f"작업 중 오류가 발생했습니다: {e}")

# --- 결과 출력 영역 (세션 유지) ---
if "result_img" in st.session_state:
    st.divider()
    st.subheader("✅ 1. 완성된 표지 사진")
    st.image(st.session_state["result_img"], use_container_width=True)
    st.download_button(
        label="📥 표지 이미지 다운로드 (JPG)",
        data=st.session_state["result_img"],
        file_name=f"{st.session_state.get('result_title', 'cover')}_인스타표지.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )

    if st.session_state.get("result_caption"):
        st.subheader("✅ 2. 인스타 피드 캡션")
        st.caption("우측 상단 복사 아이콘을 누르면 한 번에 복사됩니다.")
        st.code(st.session_state["result_caption"], language=None)
        st.text_area("캡션 직접 수정하기", value=st.session_state["result_caption"], height=300)
