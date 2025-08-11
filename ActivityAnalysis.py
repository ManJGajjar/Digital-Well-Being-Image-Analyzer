import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import pytz
import json
import plotly.express as px
from io import BytesIO
import zipfile

# Constants
TIMEZONE = 'Asia/Kolkata'  # Change to your preferred timezone
CONTENT_CATEGORIES = {
    'cars': {'color': '#3498db', 'icon': '🚗'},
    'nudity': {'color': '#e74c3c', 'icon': '⚠️'},
    'violence': {'color': '#f39c12', 'icon': '⚔️'},
    'shopping': {'color': '#9b59b6', 'icon': '🛍️'},
    'food': {'color': '#2ecc71', 'icon': '🍔'},
    'alcohol': {'color': '#e67e22', 'icon': '🍷'},
    'gambling': {'color': '#1abc9c', 'icon': '🎰'},
    'social_media': {'color': '#3498db', 'icon': '📱'},
    'neutral': {'color': '#95a5a6', 'icon': '✅'}
}

# Improved mock detection with confidence scores
def detect_content(image):
    """Improved mock content detection with confidence scores"""
    img = np.array(image)
    h, w = img.shape[:2]
    
    # Initialize results with all categories
    results = {category: {'detected': False, 'confidence': 0} for category in CONTENT_CATEGORIES}
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Food detection (green/brown dominant)
    lower_food = np.array([30, 40, 40])
    upper_food = np.array([90, 255, 255])
    food_mask = cv2.inRange(hsv, lower_food, upper_food)
    food_pixels = np.sum(food_mask > 0)
    if food_pixels > 0.2 * h * w:  # 20% of image
        results['food'] = {'detected': True, 'confidence': min(100, food_pixels / (h * w) * 200)}
    
    # Car detection (blue/metallic colors)
    lower_car = np.array([90, 50, 50])
    upper_car = np.array([130, 255, 255])
    car_mask = cv2.inRange(hsv, lower_car, upper_car)
    car_pixels = np.sum(car_mask > 0)
    if car_pixels > 0.1 * h * w:  # 10% of image
        results['cars'] = {'detected': True, 'confidence': min(100, car_pixels / (h * w) * 150)}
    
    # Nudity detection (skin tones - more precise range)
    lower_skin1 = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 48, 80], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_pixels = np.sum(skin_mask1 > 0) + np.sum(skin_mask2 > 0)
    if skin_pixels > 0.15 * h * w:  # 15% of image
        results['nudity'] = {'detected': True, 'confidence': min(100, skin_pixels / (h * w) * 120)}
    
    #Shopping detection (multiple colorful items)
    print("Image shape:", img.shape)
    if img.shape[0] > 100 and img.shape[1] > 100:  # Ensure image is large enough
        img = cv2.resize(img, (100, 100))  # Resize for uniformity
    else:
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    if len(np.unique(img.reshape(-1, 100), axis=0)) > 50:  # Many different colors
        results['shopping'] = {'detected': True, 'confidence': 10}
    
    return {k: v for k, v in results.items() if v['detected']}

def get_recommendations(detections):
    """Generate detailed recommendations with resources"""
    recommendations = []
    
    for category, data in detections.items():
        if data['detected']:
            conf = data['confidence']
            rec = {
                'type': category,  # Initialize type with the category
                'severity': 'none',
                'message': '',
                'action': '',
                'resources': []
            }
            if category == 'cars':
                rec.update({
                    'type': 'cars',
                    'severity': 'medium' if conf < 70 else 'high',
                    'message': f"Car content detected (confidence: {conf:.0f}%). Frequent exposure to luxury/performance car content can fuel materialistic desires.",
                    'action': "Consider following educational automotive channels instead of luxury car pages.",
                    'resources': [
                        "https://www.psychologytoday.com/us/blog/the-science-behind-behavior/202109/the-psychology-car-enthusiasm",
                        "https://www.mindful.org/how-to-deal-with-cravings/"
                    ]
                })
            elif category == 'nudity':
                rec.update({
                    'type': 'nudity',
                    'severity': 'high',
                    'message': f"Sensitive content detected (confidence: {conf:.0f}%). Be mindful of how this content affects your mood and self-image.",
                    'action': "Consider using platform content filters or limiting exposure time. The 'Digital Wellbeing' settings on your device can help.",
                    'resources': [
                        "https://www.helpguide.org/articles/mental-health/media-and-mental-health.htm",
                        "https://www.commonsensemedia.org/articles/how-media-can-affect-body-image"
                    ]
                })
            elif category == 'food':
                rec.update({
                    'type': 'food',
                    'severity': 'low' if conf < 50 else 'medium',
                    'message': f"Food content detected (confidence: {conf:.0f}%). While not inherently harmful, food content can trigger cravings.",
                    'action': "Be mindful of when you view food content - avoid watching when hungry or dieting.",
                    'resources': [
                        "https://www.healthline.com/nutrition/food-cravings",
                        "https://www.eatingwell.com/article/290412/how-social-media-affects-your-eating-habits/"
                    ]
                })
            recommendations.append(rec)
    
    if not recommendations:
        recommendations.append({
            'type': 'neutral',
            'severity': 'none',
            'message': "No concerning content detected. Maintain healthy browsing habits!",
            'action': "Consider setting daily screen time goals to maintain balance.",
            'resources': []
        })
    
    return recommendations

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'image_name', 'detections', 'recommendations', 
        'image_data', 'analysis_time', 'user_notes'
    ])

# App UI
st.set_page_config(layout="wide", page_title="Digital Well-being Image Analyzer", page_icon="🖼️")
st.title("📱 Sattava Path Digital Well-being Image Analyzer")
st.markdown("""
    <style>
        .stExpander .stMarkdown { font-size: 16px; }
        .severity-high { color: #e74c3c; font-weight: bold; }
        .severity-medium { color: #f39c12; font-weight: bold; }
        .severity-low { color: #2ecc71; font-weight: bold; }
        .severity-none { color: #95a5a6; }
    </style>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown("""
    This tool helps you analyze images from your digital life to:
    - Identify potentially addictive or harmful content patterns
    - Get personalized recommendations for healthier digital habits
    - Track your content consumption over time
    - Export your data for personal reflection or professional consultation
    """)

# Main analysis section
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an image from your saved content", 
                                   type=["jpg", "jpeg", "png"],
                                   help="Upload screenshots from social media, shopping apps, etc.")

with col2:
    user_notes = st.text_area("Add personal notes (optional)", 
                             help="Record how you felt about this content or why you saved it")

if uploaded_file is not None:
    # Display and process image
    image = Image.open(uploaded_file)
    timestamp = datetime.now(pytz.timezone(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
    
    with st.spinner('Analyzing content...'):
        start_time = datetime.now()
        detections = detect_content(image)
        recommendations = get_recommendations(detections)
        analysis_time = (datetime.now() - start_time).total_seconds()
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "📝 Recommendations", "📂 Export Options"])
    
    with tab1:
        st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True)
        
        # Detection badges
        st.subheader("Content Analysis")
        cols = st.columns(4)
        detected_cats = [k for k, v in detections.items() if v['detected']]
        for i, cat in enumerate(detected_cats[:4]):
            with cols[i]:
                st.markdown(f"""
                    <div style="border-radius: 10px; padding: 10px; background-color: {CONTENT_CATEGORIES[cat]['color']}20; 
                                border-left: 5px solid {CONTENT_CATEGORIES[cat]['color']}; margin: 5px 0;">
                        <p style="margin: 0; font-size: 14px;"><b>{CONTENT_CATEGORIES[cat]['icon']} {cat.capitalize()}</b></p>
                        <p style="margin: 0; font-size: 12px;">Confidence: {detections[cat]['confidence']:.0f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        
        if len(detected_cats) > 4:
            st.warning(f"+ {len(detected_cats)-4} more categories detected")
        
        # Detection visualization
        if detections:
            # Create DataFrame with correct structure
            df_data = [
                {
                    'Category': cat,
                    'Confidence': data['confidence']
                }
                for cat, data in detections.items()
            ]
            df = pd.DataFrame(df_data)
            
            if not df.empty:
                fig = px.bar(df, 
                    x='Category', 
                    y='Confidence',
                    color='Category',
                    title='Detection Confidence Levels',
                    color_discrete_map={k: v['color'] for k, v in CONTENT_CATEGORIES.items()}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if recommendations:
            st.subheader("Personalized Recommendations")
            for rec in recommendations:
                with st.expander(f"{CONTENT_CATEGORIES[rec['type']]['icon']} {rec['type'].capitalize()} - Severity: {rec['severity']}", expanded=True):
                    st.markdown(f"**{rec['message']}**")
                    st.info(f"💡 **Actionable Tip:** {rec['action']}")
                    
                    if rec['resources']:
                        st.markdown("**📚 Learn more:**")
                        for resource in rec['resources']:
                            st.markdown(f"- [{resource}]({resource})")
    
    with tab3:
        # Prepare data for export
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_data = image_bytes.getvalue()
        
        analysis_data = {
            'timestamp': timestamp,
            'image_name': uploaded_file.name,
            'detections': detections,
            'recommendations': recommendations,
            'analysis_time_seconds': analysis_time,
            'user_notes': user_notes
        }
        
        # Multiple export options
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.download_button(
                label="📥 Download Analysis Report (JSON)",
                data=json.dumps(analysis_data, indent=2),
                file_name=f"content_analysis_{timestamp.replace(':', '-')}.json",
                mime='application/json'
            )
            
            st.download_button(
                label="📊 Download All Data (CSV)",
                data=st.session_state.history.to_csv(index=False),
                file_name="content_analysis_full_history.csv",
                mime='text/csv'
            )
        
        with export_col2:
            # Create zip with image and analysis
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr(f"analysis_{timestamp.replace(':', '-')}.json", json.dumps(analysis_data, indent=2))
                zip_file.writestr(uploaded_file.name, image_data)
            
            st.download_button(
                label="🗄️ Download Complete Package (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"content_analysis_package_{timestamp.replace(':', '-')}.zip",
                mime='application/zip'
            )
    
    # Save to history
    new_entry = {
        'timestamp': timestamp,
        'image_name': uploaded_file.name,
        'detections': json.dumps(detections),
        'recommendations': json.dumps(recommendations),
        'image_data': image_data,
        'analysis_time': analysis_time,
        'user_notes': user_notes
    }
    
    st.session_state.history = pd.concat([
        st.session_state.history, 
        pd.DataFrame([new_entry])
    ], ignore_index=True)

# History and Trends section
st.sidebar.title("📅 Analysis History")
if not st.session_state.history.empty:
    # Convert history for display
    display_history = st.session_state.history.copy()
    display_history['timestamp'] = pd.to_datetime(display_history['timestamp'])
    display_history['date'] = display_history['timestamp'].dt.date
    display_history['time'] = display_history['timestamp'].dt.time
    
    # Date filter
    min_date = display_history['date'].min()
    max_date = display_history['date'].max()
    selected_dates = st.sidebar.date_input(
        "Filter by date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(selected_dates) == 2:
        display_history = display_history[
            (display_history['date'] >= selected_dates[0]) & 
            (display_history['date'] <= selected_dates[1])
        ]
    
    # Category filter
    all_categories = set()
    for detections in display_history['detections']:
        detections_dict = json.loads(detections)
        all_categories.update(detections_dict.keys())
    
    selected_categories = st.sidebar.multiselect(
        "Filter by content type",
        options=sorted(all_categories),
        default=[]
    )
    if selected_categories:
        mask = display_history['detections'].apply(
            lambda x: any(cat in json.loads(x) for cat in selected_categories)
        )
        display_history = display_history[mask]
    
    # Display filtered history
    # Display filtered history
    st.sidebar.write(f"Showing {len(display_history)} analyses")
    
    for idx, row in display_history.iterrows():
        with st.sidebar.expander(f"{row['timestamp']} - {row['image_name']}", expanded=False):
            st.image(row['image_data'], use_container_width=True)
            
            detections = json.loads(row['detections'])
            if detections:
                st.write("**Detected Content:**")
                cols = st.columns(3)
                for i, (cat, data) in enumerate(detections.items()):
                    with cols[i % 3]:
                        st.markdown(f"{CONTENT_CATEGORIES[cat]['icon']} {cat} ({data['confidence']:.0f}%)")
            
            if row['user_notes']:
                st.write("**Your Notes:**")
                st.info(row['user_notes'])
            
            if st.button(f"Delete ❌", key=f"del_{idx}"):
                st.session_state.history = st.session_state.history.drop(index=idx)
                st.rerun()
    
    # Trends visualization
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Content Trends")
    
    try:
        # Prepare trend data
        trend_data = []
        for _, row in display_history.iterrows():
            detections = json.loads(row['detections'])
            for cat, data in detections.items():
                trend_data.append({
                    'date': row['date'],
                    'category': cat,
                    'confidence': data['confidence'],
                    'count': 1
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            # Daily count by category
            daily_counts = trend_df.groupby(['date', 'category']).size().unstack(fill_value=0)
            st.sidebar.write("**Daily Content Frequency**")
            st.sidebar.bar_chart(daily_counts)
            
            # Confidence trends
            avg_confidence = trend_df.groupby('category')['confidence'].mean().sort_values(ascending=False)
            st.sidebar.write("**Average Detection Confidence**")
            st.sidebar.bar_chart(avg_confidence)
    except:
        st.sidebar.warning("Couldn't generate trends with current filters")
    
    # Clear history
    if st.sidebar.button("🧹 Clear All History"):
        st.session_state.history = pd.DataFrame(columns=[
            'timestamp', 'image_name', 'detections', 'recommendations', 
            'image_data', 'analysis_time', 'user_notes'
        ])
        st.rerun()
else:
    st.sidebar.info("No analysis history yet. Upload images to get started.")

# Additional features section
#st.markdown("---")
#st.subheader("🔮 Future Features Roadmap")
#future_features = """
#1. **Real AI Integration**: Replace mock detection with actual ML models (TensorFlow/PyTorch)
#2. **Multi-image Analysis**: Upload folders or multiple images at once for batch processing
#3. **Emotion Tracking**: Correlate content types with mood ratings
#4. **Content Blocking**: Integration with browser extensions to block detected harmful content
#5. **Personalized Challenges**: Generate weekly digital wellbeing challenges based on your patterns
#6. **Community Insights**: Anonymous aggregation of user data to show broader trends
#7. **Mobile App Version**: Standalone app with background monitoring capabilities
#8. **Professional Reports**: Generate clinician-ready reports for therapy sessions
#9. **Real-time Alerts**: Notifications when harmful patterns are detected
#10. **Integration with Screen Time APIs**: Direct connection to iOS/Android digital wellbeing tools
#"""

#st.markdown(future_features)