"""Main Streamlit application for GotYouAI - Conversation Coach."""

import streamlit as st

# Verify XSRF is disabled (for debugging - can remove later)
# st.write("XSRF Protection:", st.get_option("server.enableXsrfProtection"))

import os
import random
import time
import io
from typing import List, Optional, Dict
from PIL import Image
from src.ingestion import ingest_from_screenshots, format_for_analysis
from src.analyzer import analyze_conversation_state
from src.coach import generate_coaching

# Import is_model_loaded with fallback
try:
    from src.coach import is_model_loaded
except ImportError:
    # Fallback function if import fails
    def is_model_loaded() -> bool:
        """Check if model is already loaded (fallback)."""
        return False


# Page configuration
st.set_page_config(
    page_title="GotYouAI - Conversation Coach",
    page_icon="💬",
    layout="wide"
)

# Title and description
st.title("💬 GotYouAI - Your Conversation Coach")
st.markdown("Upload screenshots of your Instagram conversation to get AI-powered coaching and message suggestions.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    # Note: API token no longer required - using local Qwen model
    st.info("💡 Using local Qwen2.5-3B-Instruct model - no API token needed!")
    
    # Keep for backward compatibility but don't require it
    hf_api_token = st.text_input(
        "Hugging Face API Token (not required - using local model)",
        type="password",
        help="Not required - the app uses a local Qwen2.5-3B-Instruct model. This field is kept for backward compatibility only.",
        value=os.getenv("HF_API_TOKEN", "")
    )
    if hf_api_token:
        os.environ["HF_API_TOKEN"] = hf_api_token
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    GotYouAI analyzes your Instagram conversations and provides:
    - **Conversation state** (you cooking 🔥 OR… you cooked 💀)
    - **Coaching feedback** from your fav AI wingman 😈
    - **Message suggestions** to improve your conversations
    """)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'coaching_result' not in st.session_state:
    st.session_state.coaching_result = None
if 'model_loading_shown' not in st.session_state:
    st.session_state.model_loading_shown = False
if 'show_loading_screen' not in st.session_state:
    st.session_state.show_loading_screen = False
if 'proceed_with_analysis' not in st.session_state:
    st.session_state.proceed_with_analysis = False
if 'state' not in st.session_state:
    st.session_state.state = None
if 'messages' not in st.session_state:
    st.session_state.messages = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = None
if 'selected_meme_url' not in st.session_state:
    st.session_state.selected_meme_url = ""

# File uploader and proposition input - hide when analysis is complete
# Use empty placeholders that can be cleared
# Check if loading screen should be shown - if so, hide everything else
if st.session_state.show_loading_screen:
    # Initialize loading start time if not set
    if 'loading_start_time' not in st.session_state:
        st.session_state.loading_start_time = time.time()
        st.session_state.video_phase = True  # Start with video phase
    
    elapsed_time = time.time() - st.session_state.loading_start_time
    video_duration = 6.0  # Show video for exactly 6 seconds
    qr_duration = 9.0  # Show QR code for 5 seconds before proceeding
    
    if st.session_state.get('video_phase', True) and elapsed_time < video_duration:
        # Video phase - show YouTube video
        st.markdown("### ⏳ Loading AI Model...")
        st.info("🔄 Please wait while we load the AI model.")
        
        # Embed YouTube video using HTML
        video_id = "fIxz587N4RM"
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <iframe 
                    width="560" 
                    height="315" 
                    src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&loop=1&playlist={video_id}" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show progress
        progress = min(elapsed_time / video_duration, 0.95)
        st.progress(progress)
        st.caption(f"Loading model... {int(elapsed_time)}/{int(video_duration)} seconds")
        
        # Auto-refresh to update progress
        time.sleep(0.5)
        st.rerun()
    elif elapsed_time >= video_duration:
        # Video finished - show QR code and message
        if st.session_state.get('video_phase', True):
            # Transition to QR phase
            st.session_state.video_phase = False
            st.session_state.qr_start_time = time.time()
            st.rerun()
        
        # QR code phase
        st.markdown("### ☕ Support the Developer")
        
        # QR Code - located in same directory as app.py
        qr_code_path = "bmc_qr.png"
        if os.path.exists(qr_code_path):
            st.image(qr_code_path, width=300)
        else:
            # Create a placeholder QR code area if file not found
            st.markdown("""
            <div style="width: 300px; height: 300px; border: 2px dashed #ccc; 
                        display: flex; align-items: center; justify-content: center; 
                        margin: 0 auto; background-color: #f9f9f9;">
                <p style="color: #666;">QR Code Image<br/>(Add bmc_qr.png)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; 
                    border-left: 4px solid #ffc107; margin-top: 20px;">
            <p style="margin: 0; font-size: 16px;">
                😭 <strong>Sorry for the inconvenience!</strong><br/>
                The dev is poor and does not have money to buy OpenAI's API 😭
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Wait a bit to show QR code, then proceed with analysis
        qr_elapsed = time.time() - st.session_state.get('qr_start_time', time.time())
        if qr_elapsed < qr_duration:
            time.sleep(0.5)
            st.rerun()
        else:
            # QR code shown long enough - proceed with analysis
            st.session_state.show_loading_screen = False
            st.session_state.proceed_with_analysis = True
            if 'loading_start_time' in st.session_state:
                del st.session_state.loading_start_time
            if 'qr_start_time' in st.session_state:
                del st.session_state.qr_start_time
            if 'video_phase' in st.session_state:
                del st.session_state.video_phase
            time.sleep(0.5)
            st.rerun()

uploader_placeholder = st.empty()
screenshots_placeholder = st.empty()
input_placeholder = st.empty()

# Only show upload section if not loading and not complete
if not st.session_state.analysis_complete and not st.session_state.show_loading_screen:
    with uploader_placeholder.container():
        uploaded_files = st.file_uploader(
            "Upload screenshot(s) of your Instagram conversation",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload one or more screenshots containing the last 16 messages"
        )

    # Display uploaded images (folded)
    if uploaded_files:
        with screenshots_placeholder.container():
            with st.expander("📸 Uploaded Screenshots", expanded=False):
                cols = st.columns(min(len(uploaded_files), 3))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 3]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, use_container_width=True)
    else:
        screenshots_placeholder.empty()

    # Optional proposition input
    with input_placeholder.container():
        proposition_input = st.text_input(
            "💭 Optional: Draft message you want to send",
            placeholder="Paste a message you're thinking of sending...",
            help="Get feedback on a message before sending it"
        )
else:
    # Clear all uploader-related widgets when analysis is complete
    uploader_placeholder.empty()
    screenshots_placeholder.empty()
    input_placeholder.empty()
    uploaded_files = None
    proposition_input = ""

# Display analysis results if analysis is complete
if st.session_state.analysis_complete and st.session_state.coaching_result:
    coaching_result = st.session_state.coaching_result
    state = st.session_state.state
    messages = st.session_state.messages
    selected_meme_url = st.session_state.selected_meme_url
    
    # Meme URLs for each state
    state_memes = {
        "cooking": [
            "https://tenor.com/zh-TW/view/basketball-dunk-ball-march-madness-slam-dunk-gif-11946911757704805699",
            "https://tenor.com/zh-TW/view/lebron-gif-4382226072817291094",
            "https://tenor.com/zh-TW/view/kocham-cię-gif-6327772009437970425",
            "https://tenor.com/zh-TW/view/greenfc-chrisdablack-greenfn-cdb-chrisfn-gif-14011322296617422842",
            "https://tenor.com/zh-TW/view/ballin-cat-dunk-basketball-meme-gif-27081377",
            "https://tenor.com/zh-TW/view/basketball-gif-900457100658188470",
            "https://tenor.com/zh-TW/view/mazda-airball-steph-curry-space-airball-space-airball-success-space-ball-successful-fein-airball-gif-4672416707560506981",
            "https://tenor.com/zh-TW/view/airball-gif-11646527062916024729",
            "https://tenor.com/zh-TW/view/greeeen-gif-10411387815657274690",
            "https://tenor.com/zh-TW/view/basketball-basketball-meme-basketball-thumbnail-benjammins-difficult-shot-gif-2839553667802803209"
        ],
        "wait": [
            "https://tenor.com/zh-TW/view/basketball-kid-roblox-xxl-basketball-head-basketball-head-gif-26405889",
            "https://tenor.com/zh-TW/view/airball-nba-brick-gif-15266795546092124798",
            "https://tenor.com/zh-TW/view/tanner-martin-missed-shot-baby-gap-basketball-gif-23675519",
            "https://tenor.com/zh-TW/view/tiktok-shaquille-shaq-timeout-ethical-gif-9311067668442735802",
            "https://tenor.com/zh-TW/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455"
        ],
        "cooked": [
            "https://tenor.com/zh-TW/view/misfired-rocket-shocked-gif-16450198",
            "https://tenor.com/zh-TW/view/fail-funny-basketball-jump-dunk-gif-4586361",
            "https://tenor.com/zh-TW/view/dog-sunset-javgag-gif-6354990519690233079",
            "https://tenor.com/zh-TW/view/chill-dude-chill-dude-im-just-a-chill-dude-just-a-chill-dude-gif-15385961914175037407",
            "https://tenor.com/zh-TW/view/long-tears-gif-11087877242190144149"
        ]
    }
    
    st.success("✅ Analysis complete!")
    
    # State visualization - meme on left, coaching and suggestion in same column on right
    col1, col2 = st.columns([1.2, 2.8])
    with col1:
        state_emoji = {"cooking": "🔥", "wait": "⏳", "cooked": "💀"}
        state_label = state.upper()
        st.markdown(f"### {state_emoji[state]} {state_label}")
        # Display random meme - bigger size
        if selected_meme_url:
            # Extract GIF ID from URL (last number in the URL)
            gif_id = selected_meme_url.split('-')[-1]
            # Use Tenor's embed URL format
            embed_url = f"https://tenor.com/embed/{gif_id}"
            # Display using HTML iframe for better GIF support - increased size
            st.components.v1.html(
                f'<iframe src="{embed_url}" width="100%" height="500" frameBorder="0" allowFullScreen></iframe>',
                height=520
            )
    
    with col2:
        # Coaching and suggested answer in the same column
        st.markdown("### 💡 Coaching")
        st.info(coaching_result['coaching'])
        
        # Suggestions section (moved below coaching) - only show one suggestion
        st.markdown("---")
        st.subheader("💬 Suggested Message")
        
        suggestions = coaching_result['suggestions']
        top_suggestion = suggestions[0] if suggestions else ""
        if suggestions:
            # Use markdown with custom CSS for automatic word wrapping
            st.markdown(
                f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #0084FF;
                    word-wrap: break-word;
                    word-break: break-word;
                    white-space: pre-wrap;
                    overflow-wrap: break-word;
                    max-width: 100%;
                ">
                    {top_suggestion}
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("📋 Copy", key="copy_suggestion", use_container_width=True):
                st.write(f"✅ Copied: {top_suggestion}")
        
        # "Analyze more conversations" button
        st.markdown("---")
        
        if st.button("👀 Analyze more conversations?", use_container_width=True):
            # Reset all session state to initial values (loop back to start)
            st.session_state.analysis_complete = False
            st.session_state.coaching_result = None
            st.session_state.state = None
            st.session_state.messages = None
            st.session_state.uploaded_files_data = None
            st.session_state.selected_meme_url = ""
            st.rerun()

# Analyze button - only show when not in analysis complete state and not loading
if not st.session_state.analysis_complete and not st.session_state.show_loading_screen:
    if st.button("🔍 Analyze Conversation", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one screenshot first.")
        else:
            # Store uploaded files in session state before showing loading screen
            # Read file bytes immediately since UploadedFile objects don't persist across reruns
            st.session_state.uploaded_files_data = []
            for file in uploaded_files:
                # Read the file bytes and store them
                file.seek(0)  # Reset file pointer
                file_bytes = file.read()
                file.seek(0)  # Reset again for potential immediate use
                st.session_state.uploaded_files_data.append({
                    'name': file.name,
                    'bytes': file_bytes,
                    'type': file.type
                })
            
            # Store proposition input if it exists
            if 'proposition_input' in locals() and proposition_input:
                st.session_state.proposition_input_data = proposition_input
            else:
                st.session_state.proposition_input_data = ""
            
            # Check if model needs to be loaded
            model_needs_loading = not is_model_loaded()
            
            if model_needs_loading and not st.session_state.model_loading_shown:
                # Set flag to show we've displayed loading screen
                st.session_state.model_loading_shown = True
                st.session_state.show_loading_screen = True
                st.rerun()
            else:
                # Model already loaded, proceed directly with analysis
                st.session_state.proceed_with_analysis = True
                st.rerun()

# Proceed with analysis if flag is set (after loading or if model already loaded)
if st.session_state.get('proceed_with_analysis', False) and not st.session_state.show_loading_screen:
    if 'proceed_with_analysis' in st.session_state:
        del st.session_state.proceed_with_analysis
    
    # Get uploaded files from session state (stored before loading screen)
    uploaded_files_data = st.session_state.get('uploaded_files_data', None)
    proposition_input = st.session_state.get('proposition_input_data', "")
    
    if not uploaded_files_data:
        st.error("❌ No uploaded files found. Please upload screenshots and try again.")
    else:
        with st.spinner("Processing screenshots and analyzing conversation..."):
            try:
                # Convert stored file bytes back to PIL Images
                # uploaded_files_data is a list of dicts with 'bytes' and 'name' keys
                image_list = []
                for file_data in uploaded_files_data:
                    # Create PIL Image from bytes
                    image = Image.open(io.BytesIO(file_data['bytes']))
                    image_list.append(image)
                
                # Ingest messages from screenshots
                messages = ingest_from_screenshots(image_list)
                
                if not messages:
                    st.error("❌ No messages could be extracted from the screenshots. Please ensure the images are clear and contain visible conversation text.")
                else:
                    # Format for analysis
                    formatted_conversation = format_for_analysis(messages, proposition_input)
                    
                    # Analyze state
                    state = analyze_conversation_state(messages)
                    
                    # Generate coaching (this will load the model if not already loaded)
                    coaching_result = generate_coaching(
                        formatted_conversation,
                        state,
                        hf_api_token=hf_api_token if hf_api_token else None
                    )
                    
                    # Store results in session state
                    st.session_state.analysis_complete = True
                    st.session_state.coaching_result = coaching_result
                    st.session_state.state = state
                    st.session_state.messages = messages
                    # Store uploaded files data for later use
                    if uploaded_files:
                        st.session_state.uploaded_files_data = [file for file in uploaded_files]
                    
                    # Meme URLs for each state
                    state_memes = {
                        "cooking": [
                            "https://tenor.com/zh-TW/view/basketball-dunk-ball-march-madness-slam-dunk-gif-11946911757704805699",
                            "https://tenor.com/zh-TW/view/lebron-gif-4382226072817291094",
                            "https://tenor.com/zh-TW/view/kocham-cię-gif-6327772009437970425",
                            "https://tenor.com/zh-TW/view/greenfc-chrisdablack-greenfn-cdb-chrisfn-gif-14011322296617422842",
                            "https://tenor.com/zh-TW/view/ballin-cat-dunk-basketball-meme-gif-27081377",
                            "https://tenor.com/zh-TW/view/basketball-gif-900457100658188470",
                            "https://tenor.com/zh-TW/view/mazda-airball-steph-curry-space-airball-space-airball-success-space-ball-successful-fein-airball-gif-4672416707560506981",
                            "https://tenor.com/zh-TW/view/airball-gif-11646527062916024729",
                            "https://tenor.com/zh-TW/view/greeeen-gif-10411387815657274690",
                            "https://tenor.com/zh-TW/view/basketball-basketball-meme-basketball-thumbnail-benjammins-difficult-shot-gif-2839553667802803209"
                        ],
                        "wait": [
                            "https://tenor.com/zh-TW/view/basketball-kid-roblox-xxl-basketball-head-basketball-head-gif-26405889",
                            "https://tenor.com/zh-TW/view/airball-nba-brick-gif-15266795546092124798",
                            "https://tenor.com/zh-TW/view/tanner-martin-missed-shot-baby-gap-basketball-gif-23675519",
                            "https://tenor.com/zh-TW/view/tiktok-shaquille-shaq-timeout-ethical-gif-9311067668442735802",
                            "https://tenor.com/zh-TW/view/mr-bean-mrbean-bean-mr-bean-holiday-mr-bean-holiday-movie-gif-3228235746377647455"
                        ],
                        "cooked": [
                            "https://tenor.com/zh-TW/view/misfired-rocket-shocked-gif-16450198",
                            "https://tenor.com/zh-TW/view/fail-funny-basketball-jump-dunk-gif-4586361",
                            "https://tenor.com/zh-TW/view/dog-sunset-javgag-gif-6354990519690233079",
                            "https://tenor.com/zh-TW/view/chill-dude-chill-dude-im-just-a-chill-dude-just-a-chill-dude-gif-15385961914175037407",
                            "https://tenor.com/zh-TW/view/long-tears-gif-11087877242190144149"
                        ]
                    }
                    
                    # Select random meme URL for this state
                    selected_meme_url = ""
                    if state in state_memes:
                        selected_meme_url = random.choice(state_memes[state])
                        st.session_state.selected_meme_url = selected_meme_url
                    
                    # Rerun to show results from session state (hides uploader, shows results)
                    st.rerun()
                        
            except ValueError as e:
                st.error(f"❌ Error processing screenshot: {str(e)}")
                st.info("💡 Tips: Ensure the screenshot is clear, well-lit, and messages are visible. Try adjusting image quality or lighting.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built for SB Hacks 2026 • Powered by Hugging Face 🤗 • Made with ❤️, 💭, and a lot of 🤭"
    "</div>",
    unsafe_allow_html=True
)