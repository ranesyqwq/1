#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shanghai CityWalk Sentiment Analysis System
Analyze: Extract Landmarks -> Sentiment Analysis -> Overall Scoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from collections import defaultdict
import warnings
import os
import sys

# Set Chinese font
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def extract_landmarks_from_data(df):
    """从数据中自动提取所有出现的地名"""
    landmark_keywords = [
        '外滩', '南京路', '豫园', '城隍庙', '田子坊', '新天地',
        '武康路', '安福路', '思南公馆', '静安寺', '陆家嘴',
        '迪士尼', '朱家角', '枫泾', '七宝', 'M50',
        '上生新所', '愚园路', '淮海路', '甜爱路', '多伦路',
        '徐家汇', '龙华寺', '长乐路', '乌鲁木齐路', '陕西南路',
        '复兴路', '嘉陵路', '淮海中路', '黄陂南路', '泰康路',
        '湖南路', '天平路', '衡山路', '山阴路', '霍山路',
        '茅台路', '永康路', '汾阳路', '巨鹿路', '富民路',
        '建国西路', '建国中路', '建国路', '复兴中路', '复兴西路',
        '陕西北路', '西康路', '威海路', '万航渡路', '昭化路',
        '铜仁路', '华山路', '东平路', '古美路', '南阳路',
        '凯旋路', '百乐门', '福州路', '兆丰路', '冠生园',
        '北京东路', '北京西路', '人民广场', '人民公园',
        '东方明珠', '世纪大道', '浦东', '浦西', '浦北',
        '北外滩', '外白渡桥', '皇家园林', '四川北路',
        '共青团', '虹口', '黄浦江', '长风',
        '瑞金医院', '长海医院', '静安别墅', '常德公馆',
        '仁德里', '三十二弄', '吴昌硕公园', '长宁公园',
        '江南造船厂', '文采里', '恒丰路',
    ]
    
    all_content = ' '.join(df['content'].dropna().astype(str).tolist())
    found_landmarks = set()
    
    for keyword in landmark_keywords:
        if keyword in all_content:
            found_landmarks.add(keyword)
    
    return sorted(list(found_landmarks))


class SimpleSentimentAnalyzer:
    """Simple Chinese Sentiment Analyzer - Keyword-based"""
    
    def __init__(self):
        self.positive_words = {
            '很棒': 0.9, '很好': 0.85, '很美': 0.85, '很漂亮': 0.9, '不错': 0.8,
            '值得': 0.85, '推荐': 0.9, '喜欢': 0.85, '满意': 0.8, '开心': 0.85,
            '舒服': 0.8, '优雅': 0.85, '特色': 0.75, '有趣': 0.85, '完美': 0.95,
            '精妙': 0.85, '精致': 0.8, '亮点': 0.75, '亮丽': 0.8, '生机': 0.8,
            '壮观': 0.8, '雄伟': 0.85, '古朴': 0.75, '气息': 0.7, '浓厚': 0.7,
            '独特': 0.75, '创意': 0.8, '艺术': 0.75, '文化': 0.7, '历史': 0.7,
            '安静': 0.75, '清幽': 0.8, '宁静': 0.8, '祥和': 0.85, '浪漫': 0.85,
            '繁华': 0.7, '热闹': 0.7, '活力': 0.75, '欢乐': 0.85, '有意思': 0.8,
            '亲近': 0.75, '底蕴': 0.7, '品味': 0.75, '迷人': 0.85, '梦幻': 0.85,
            '高级': 0.75, '设计感': 0.8, '韵味': 0.8, '风情': 0.75, '气质': 0.75,
        }
        
        self.negative_words = {
            '很差': 0.15, '不好': 0.2, '很丑': 0.1, '讨厌': 0.05, '失望': 0.25,
            '后悔': 0.15, '浪费': 0.2, '不满': 0.25, '难过': 0.2, '伤心': 0.15,
            '生气': 0.2, '不舒服': 0.25, '拥挤': 0.3, '排队': 0.35, '费钱': 0.3,
            '太高': 0.35, '过度': 0.3, '贵': 0.35, '昂贵': 0.3, '坑': 0.15,
            '骗': 0.1, '缺少': 0.35, '没有': 0.4, '无': 0.4, '冷清': 0.35,
            '荒凉': 0.25, '破旧': 0.2, '陈旧': 0.35, '落后': 0.3, '不方便': 0.3,
        }
        
        self.negation_words = {'不', '没', '无', '别', '莫'}
    
    def analyze(self, text):
        """Analyze sentiment score (0-1)"""
        text = str(text)
        # For Chinese text, we don't use .lower() and split by space
        # Instead, we check each word in our dictionary directly in the text
        
        positive_score = 0
        negative_score = 0
        
        # Check for positive words
        for word, score in self.positive_words.items():
            if word in text:
                positive_score += score
        
        # Check for negative words
        for word, score in self.negative_words.items():
            if word in text:
                negative_score += score
        
        total = positive_score + negative_score
        if total == 0:
            return 0.5
        
        sentiment = positive_score / total
        return min(1.0, max(0.0, sentiment))


def preprocess_text(text):
    """Text preprocessing"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data():
    """Load data from Excel file"""
    data_path = r'c:\Users\27885\Desktop\citywalk\去重后的数据.xlsx'
    
    try:
        df = pd.read_excel(data_path)
        if len(df) > 0:
            return df
    except Exception as e:
        print(f"Error loading file: {e}")
    
    # Fallback to sample data
    return pd.DataFrame({
        'content': [
            '武康路上的老洋房充满历史气息，散步很舒服，值得一来',
            '新天地的建筑很有设计感，但消费太高，有点失望',
            '豫园是上海的标志性景点，园林设计精妙，很值得看',
        ]
    })


def main():
    """Main analysis function"""
    
    print("="*70)
    print("Shanghai CityWalk Sentiment Analysis System")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    
    # Rename columns if needed
    if 'content' not in df.columns and len(df.columns) >= 2:
        old_col = df.columns[1]
        df.rename(columns={old_col: 'content'}, inplace=True)
    
    print(f"Loaded {len(df)} comments")
    
    # Preprocess
    print("Preprocessing...")
    df['processed'] = df['content'].apply(preprocess_text)
    valid_count = len(df[df['processed'] != ''])
    print(f"Valid: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")
    
    # Extract landmarks
    print("\nExtracting landmarks...")
    landmarks = extract_landmarks_from_data(df)
    
    if not landmarks:
        print("No landmarks found")
        return
    
    print(f"Found {len(landmarks)} landmarks")
    
    # Extract comments for each landmark
    print("Processing landmarks...")
    landmark_data = defaultdict(list)
    landmark_raw = defaultdict(list)
    
    for idx, row in df.iterrows():
        content = row['content']
        for landmark in landmarks:
            if landmark in content:
                landmark_data[landmark].append(content)
                landmark_raw[landmark].append(content)
    
    sorted_landmarks = sorted(landmark_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Analyze sentiment
    print(f"\nAnalyzing sentiment for {len(sorted_landmarks)} landmarks...")
    analyzer = SimpleSentimentAnalyzer()
    results = []
    
    for idx, (landmark, comments) in enumerate(sorted_landmarks, 1):
        sentiments = [analyzer.analyze(c) for c in comments]
        
        avg_sentiment = np.mean(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.6)
        negative_count = sum(1 for s in sentiments if s < 0.4)
        positive_rate = positive_count / len(sentiments) if sentiments else 0
        
        if sentiments:
            best_idx = np.argmax(sentiments)
            sample_text = comments[best_idx][:40]
        else:
            sample_text = ""
        
        if avg_sentiment > 0.7:
            grade = "Excellent"
        elif avg_sentiment > 0.6:
            grade = "Good"
        elif avg_sentiment > 0.4:
            grade = "Average"
        else:
            grade = "Poor"
        
        results.append({
            'Landmark': landmark,
            'Score': f"{avg_sentiment:.2f}",
            'Grade': grade,
            'Positive': positive_count,
            'Negative': negative_count,
            'PosRate': f"{positive_rate:.1%}",
            'Count': len(sentiments),
            'Sample': sample_text
        })
        
        if idx <= 10 or idx % 5 == 0:
            print(f"  [{idx}/{len(sorted_landmarks)}] {landmark}: {avg_sentiment:.2f}")
    
    # Save CSV
    print("\nSaving results...")
    output_dir = r'c:\Users\27885\Desktop\citywalk\情感分析'
    csv_path = os.path.join(output_dir, 'citywalk_analysis_results.csv')
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV saved: {csv_path}")
    
    # Create visualization
    print("Creating charts...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Shanghai CityWalk Sentiment Analysis', fontsize=16, fontweight='bold')
    
    landmarks_top = result_df.head(10)
    
    # Chart 1: Score
    ax = axes[0, 0]
    scores = [float(x) for x in landmarks_top['Score']]
    ax.barh(landmarks_top['Landmark'], scores, color='skyblue')
    ax.set_xlabel('Sentiment Score')
    ax.set_title('Top 10 Landmarks - Score')
    ax.set_xlim(0, 1)
    
    # Chart 2: Review Count
    ax = axes[0, 1]
    counts = landmarks_top['Count'].astype(int)
    ax.bar(range(len(landmarks_top)), counts, color='lightcoral')
    ax.set_xticks(range(len(landmarks_top)))
    ax.set_xticklabels(landmarks_top['Landmark'], rotation=45, ha='right')
    ax.set_ylabel('Review Count')
    ax.set_title('Top 10 Landmarks - Review Count')
    
    # Chart 3: Positive Rate
    ax = axes[1, 0]
    pos_rates = [float(x.rstrip('%'))/100 for x in landmarks_top['PosRate']]
    ax.bar(range(len(landmarks_top)), pos_rates, color='lightgreen')
    ax.set_xticks(range(len(landmarks_top)))
    ax.set_xticklabels(landmarks_top['Landmark'], rotation=45, ha='right')
    ax.set_ylabel('Positive Rate')
    ax.set_title('Top 10 Landmarks - Positive Rate')
    ax.set_ylim(0, 1)
    
    # Chart 4: Distribution
    ax = axes[1, 1]
    all_scores = [float(x) for x in result_df['Score']]
    ax.hist(all_scores, bins=10, color='orange', edgecolor='black')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title('All Landmarks - Score Distribution')
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'citywalk_analysis_results.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved: {png_path}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  1. {csv_path}")
    print(f"  2. {png_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
