"""
上海CityWalk打卡点情感分析系统 (无外部依赖版本)
功能：提取打卡点 -> 情感分析 -> 综合评分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from collections import defaultdict
import warnings
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class SimpleSentimentAnalyzer:
    """简单的中文情感分析器 - 基于关键词"""
    
    def __init__(self):
        # 积极词汇
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
            '素质': 0.7, '修养': 0.7, '优质': 0.8, '顶级': 0.85
        }
        
        # 否定词汇
        self.negative_words = {
            '很差': 0.15, '不好': 0.2, '很丑': 0.1, '讨厌': 0.05, '失望': 0.25,
            '后悔': 0.15, '浪费': 0.2, '不满': 0.25, '难过': 0.2, '伤心': 0.15,
            '生气': 0.2, '不舒服': 0.25, '拥挤': 0.3, '排队': 0.35, '费钱': 0.3,
            '太高': 0.35, '过度': 0.3, '贵': 0.35, '昂贵': 0.3, '坑': 0.15,
            '骗': 0.1, '缺少': 0.35, '没有': 0.4, '无': 0.4, '没': 0.4,
            '冷清': 0.35, '荒凉': 0.25, '破旧': 0.2, '陈旧': 0.35, '落后': 0.3,
            '不方便': 0.3, '不舒适': 0.3, '难受': 0.25, '累': 0.3, '疲惫': 0.3,
            '反感': 0.15, '厌烦': 0.2, '烦': 0.25, '讨厌': 0.15, '厌': 0.25,
            '不': 0.4, '没': 0.4, '没有': 0.4, '无': 0.4, '别': 0.35
        }
        
        # 否定修饰词
        self.negation_words = {'不', '没', '无', '别', '莫'}
    
    def analyze(self, text):
        """分析文本情感得分 (0-1)"""
        if not text or not isinstance(text, str):
            return 0.5
        
        text = text.lower()
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text)
        
        positive_score = 0
        negative_score = 0
        
        for i, word in enumerate(words):
            # 检查积极词汇
            if word in self.positive_words:
                score = self.positive_words[word]
                # 检查是否被否定
                if i > 0 and words[i-1] in self.negation_words:
                    negative_score += score
                else:
                    positive_score += score
            
            # 检查否定词汇
            elif word in self.negative_words:
                score = self.negative_words[word]
                negative_score += score
        
        # 计算综合得分
        total = positive_score + negative_score
        if total == 0:
            return 0.5
        
        sentiment = positive_score / total
        return min(1.0, max(0.0, sentiment))


def preprocess_text(text):
    """文本预处理"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data():
    """加载数据"""
    print("=" * 70)
    print("🌆 上海CityWalk打卡点情感分析系统".center(70))
    print("=" * 70)
    
    possible_paths = [
        '去重后的数据.csv',
        '去重后的数据.xlsx',
        os.path.expanduser('~/Desktop/去重后的数据.xlsx'),
        os.path.expanduser('~/Desktop/原数据.xlsx'),
    ]
    
    df = None
    for path in possible_paths:
        try:
            if path.endswith('.xlsx'):
                df = pd.read_excel(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path, encoding='utf-8')
            
            if df is not None and len(df) > 0:
                print(f"\n✅ 成功加载数据: {path}")
                return df
        except:
            continue
    
    print("\n⚠️  未找到实际数据文件，使用示例数据演示")
    return create_sample_data()


def create_sample_data():
    """创建示例数据"""
    sample_data = {
        'content': [
            '武康路上的老洋房充满历史气息，散步很舒服，值得一来',
            '新天地的建筑很有设计感，但消费太高，有点失望',
            '豫园是上海的标志性景点，园林设计精妙，很值得看',
            '外滩看浦江夜景很美，但人太多了，拍照不方便',
            '田子坊有很多特色小店，创意十足，逛了半天还没逛够',
            '城隍庙的传统建筑保护得很好，感受到了老上海的文化',
            '陆家嘴的高楼很壮观，但缺少人文气息',
            '安福路很安静，古树很多，特别适合散步',
            '思南公馆的环境很优雅，适合拍照',
            '静安寺很宁静祥和，虽然人多但很有氛围',
            '朱家角的古镇风情很浓厚，水乡景色很美',
            '上生新所的老建筑新玩法很有意思，很有趣',
            '愚园路的生活气息很浓，感觉很亲近',
            '多伦路的文化底蕴很深，值得细细品味',
            '武康路和田子坊都很推荐，各有特色',
            '外滩的景色一般，感觉没想象中那么好',
            '迪士尼乐园很欢乐，小朋友很开心，但排队太久',
            '枫泾古镇比较冷清，但古色古香，有历史感',
            '七宝有江南水乡的韵味，很不错',
            '淮海路是购物天堂，很繁华',
            '徐家汇天主教堂庄严肃穆，建筑很有特色',
            '龙华寺的历史悠久，环境很清幽',
            '1933老场坊改造得很特别，有艺术气息',
            '甜爱路很浪漫，适合情侣打卡',
            'M50创意园区很有艺术范儿',
        ]
    }
    return pd.DataFrame(sample_data)


def main():
    """主函数"""
    
    # 1. 加载数据
    df = load_data()
    
    if 'content' not in df.columns:
        cols = [c for c in df.columns if '内容' in c or '评论' in c or '文本' in c]
        if cols:
            df.rename(columns={cols[0]: 'content'}, inplace=True)
        else:
            print(f"❌ 错误：无法找到内容列\n可用字段: {df.columns.tolist()}")
            return
    
    print(f"📊 数据量: {len(df)} 条评论\n")
    
    # 2. 文本预处理
    print("🔄 正在预处理文本...")
    df['processed'] = df['content'].apply(preprocess_text)
    valid_count = len(df[df['processed'] != ''])
    print(f"✓ 有效文本: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)\n")
    
    # 3. 打卡点库
    landmarks = [
        '外滩', '南京路', '豫园', '城隍庙', '田子坊', '新天地',
        '武康路', '安福路', '思南公馆', '静安寺', '陆家嘴',
        '迪士尼', '朱家角', '枫泾', '七宝', 'M50', '1933',
        '上生新所', '愚园路', '淮海路', '甜爱路', '多伦路',
        '徐家汇', '龙华寺', '共青森林公园', '东平国家森林公园'
    ]
    
    # 4. 提取打卡点
    print("🔍 正在提取打卡点...")
    landmark_data = defaultdict(list)
    landmark_raw = defaultdict(list)
    
    for idx, row in df.iterrows():
        processed = row['processed']
        original = row['content']
        for landmark in landmarks:
            if landmark in processed:
                landmark_data[landmark].append(processed)
                landmark_raw[landmark].append(original)
    
    if not landmark_data:
        print("❌ 未找到任何打卡点")
        return
    
    # 按数量排序
    sorted_landmarks = sorted(landmark_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n✓ 识别到 {len(sorted_landmarks)} 个打卡点:")
    for i, (lm, comments) in enumerate(sorted_landmarks[:10], 1):
        print(f"   {i:2d}. {lm:10s} ({len(comments):3d} 条评论)")
    if len(sorted_landmarks) > 10:
        print(f"   ... 等共 {len(sorted_landmarks)} 个")
    
    # 5. 情感分析
    print("\n" + "=" * 70)
    print("🚀 执行情感分析...".center(70))
    print("=" * 70 + "\n")
    
    analyzer = SimpleSentimentAnalyzer()
    results = []
    
    for idx, (landmark, comments) in enumerate(sorted_landmarks, 1):
        print(f"[{idx}/{len(sorted_landmarks)}] 分析 {landmark:12s}", end=" ", flush=True)
        
        # 分析每条评论
        sentiments = [analyzer.analyze(c) for c in comments]
        
        # 统计指标
        avg_sentiment = np.mean(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.6)
        negative_count = sum(1 for s in sentiments if s < 0.4)
        positive_rate = positive_count / len(sentiments) if sentiments else 0
        
        # 找最具代表性的评论
        best_idx = np.argmax(sentiments)
        sample_text = landmark_raw[landmark][best_idx][:45]
        
        # 情感等级
        if avg_sentiment >= 0.7:
            grade = '强正面'
        elif avg_sentiment >= 0.6:
            grade = '正面'
        elif avg_sentiment >= 0.4:
            grade = '中立'
        else:
            grade = '负面'
        
        results.append({
            '打卡点': landmark,
            '情感得分': round(avg_sentiment, 4),
            '情感等级': grade,
            '积极评论数': positive_count,
            '负面评论数': negative_count,
            '积极率': round(positive_rate, 3),
            '样本量': len(comments),
            '示例': sample_text
        })
        
        # 进度条
        print(f"✓ {avg_sentiment:.3f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('情感得分', ascending=False)
    
    # 6. 保存结果
    print("\n" + "=" * 70)
    print("💾 保存结果".center(70))
    print("=" * 70 + "\n")
    
    csv_path = '打卡点情感分析结果.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf_8_sig')
    print(f"✅ CSV 文件: {csv_path}")
    
    # 7. 显示表格结果
    print("\n📋 情感分析结果汇总：\n")
    print(f"{'排名':^4} | {'打卡点':^12} | {'得分':^6} | {'等级':^6} | {'积极率':^7} | {'样本':^5} | {'示例':^20}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:4d} | {row['打卡点']:12s} | {row['情感得分']:6.3f} | {row['情感等级']:6s} | {row['积极率']:6.1%} | {row['样本量']:5d} | {row['示例']:20s}")
    
    # 8. Top5推荐
    print("\n" + "=" * 70)
    print("🏆 最值得推荐的TOP5打卡点".center(70))
    print("=" * 70 + "\n")
    
    top5 = results_df.head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        stars = "⭐" * int(row['情感得分'] * 5)
        print(f"{i}. {row['打卡点']:12s} | 得分: {row['情感得分']:.3f} {stars} | 评论数: {row['样本量']}")
    
    # 9. 可视化
    print("\n📊 生成可视化图表...", end=" ", flush=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🌆 上海CityWalk打卡点情感分析报告', fontsize=18, fontweight='bold')
    
    # 图1：情感得分排行
    ax1 = axes[0, 0]
    cmap = plt.cm.get_cmap('RdYlGn')
    colors = [cmap(s) for s in results_df['情感得分']]
    bars = ax1.barh(results_df['打卡点'], results_df['情感得分'], color=colors, edgecolor='grey', linewidth=1.5)
    ax1.set_xlabel('情感得分', fontsize=11)
    ax1.set_title('📊 情感得分排行', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.3, 1.0)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar, score in zip(bars, results_df['情感得分']):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.3f}', 
                va='center', fontsize=8)
    
    # 图2：样本量对比
    ax2 = axes[0, 1]
    ax2.bar(range(len(results_df)), results_df['样本量'], color='skyblue', edgecolor='navy', linewidth=1.5)
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels(results_df['打卡点'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('评论数量', fontsize=11)
    ax2.set_title('📈 评论数量分布', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 图3：积极率
    ax3 = axes[1, 0]
    ax3.bar(range(len(results_df)), results_df['积极率'], color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    ax3.set_xticks(range(len(results_df)))
    ax3.set_xticklabels(results_df['打卡点'], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('积极评论比例', fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.set_title('😊 积极评论率', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 图4：等级分布饼图
    ax4 = axes[1, 1]
    grade_counts = results_df['情感等级'].value_counts()
    colors_pie = ['#2ca02c', '#ffdd57', '#ff7f0e', '#d62728']
    grade_order = ['强正面', '正面', '中立', '负面']
    grade_counts = grade_counts.reindex([g for g in grade_order if g in grade_counts.index])
    ax4.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.0f%%',
            colors=colors_pie[:len(grade_counts)], startangle=90)
    ax4.set_title('🎯 情感等级分布', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    png_path = '打卡点情感分析结果.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓")
    print(f"✅ PNG 文件: {png_path}\n")
    
    # 10. 深度洞察
    print("=" * 70)
    print("💡 深度洞察分析".center(70))
    print("=" * 70 + "\n")
    
    overall_score = results_df['情感得分'].mean()
    overall_positive_rate = results_df['积极率'].mean()
    
    if overall_score >= 0.7:
        desc = "🌟 高度推荐"
    elif overall_score >= 0.6:
        desc = "😊 值得体验"
    elif overall_score >= 0.5:
        desc = "😐 一般"
    else:
        desc = "😞 需谨慎"
    
    print(f"📊 整体评估:")
    print(f"   • 综合情感得分: {overall_score:.3f}/1.0 - {desc}")
    print(f"   • 整体积极率: {overall_positive_rate:.1%}")
    print(f"   • 分析打卡点: {len(results_df)} 个")
    print(f"   • 总评论数: {results_df['样本量'].sum()} 条")
    
    print(f"\n🏅 排名概览:")
    print(f"   • 🥇 最佳: {results_df.iloc[0]['打卡点']} ({results_df.iloc[0]['情感得分']:.3f})")
    if len(results_df) > 1:
        most_comments = results_df.nlargest(1, '样本量').iloc[0]
        print(f"   • 🔥 热门: {most_comments['打卡点']} ({most_comments['样本量']} 条评论)")
        print(f"   • ❌ 需改: {results_df.iloc[-1]['打卡点']} ({results_df.iloc[-1]['情感得分']:.3f})")
    
    print(f"\n📈 按等级统计:")
    for grade in ['强正面', '正面', '中立', '负面']:
        items = results_df[results_df['情感等级'] == grade]
        if len(items) > 0:
            names = items['打卡点'].tolist()
            print(f"   • {grade:6s}: {', '.join(names[:5])}", end="")
            if len(names) > 5:
                print(f" 等 ({len(names)} 个)")
            else:
                print()
    
    print("\n" + "=" * 70)
    print("✨ 分析完成！".center(70))
    print("=" * 70)
    print(f"\n📁 生成的文件:")
    print(f"   1. 详细结果: {csv_path}")
    print(f"   2. 可视化: {png_path}")
    print(f"\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键关闭程序...")
